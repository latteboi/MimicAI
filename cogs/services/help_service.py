import hashlib
import os
import asyncio
from typing import Optional

from ..utils.constants import defaultConfig, DOCS_DIR, DEFAULT_HELP_MODE_INJECTION
from ..utils.content import DEFAULT_HELP_DOCS
from ..utils.menu_map import build_menu_map
from ..managers.memory_manager import encode_embedding_b64, decode_embedding_b64
from .api_service import get_embedding_vector

# Written beside the shards. Maps each default shard to the SHA-256 of the text this
# build ships for it, which is what lets an upgrade tell "the operator rewrote this"
# apart from "this is last release's wording".
_MANIFEST_FILENAME = ".defaults_manifest.json"

# Bumping the embedding model or its dimensionality invalidates every cached vector.
# Stored in the cache so a corpus edit and a format change both force a rebuild.
_CACHE_FORMAT = "v2-256d"

# Retrieval shape. The threshold is deliberately high: below it, an ordinary
# roleplay line starts dragging documentation into a turn that never asked for it.
_RELEVANCE_THRESHOLD = 0.60
_TOP_DOCUMENTS = 5


def _digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


class HelpService:
    """Owns the Help Mode documentation corpus: writing the default shards to disk,
    embedding them, and answering a query with the closest ones.

    Holds a back-reference to the parent cog for shared instance caches and cross-manager lookups,
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    def _ensure_docs_directory(self) -> bool:
        """Writes the bundled documentation shards to disk, and returns whether the
        on-disk corpus changed.

        The old version wrote a shard only when the file was absent, which meant a
        release could revise the documentation and no existing instance would ever see
        it -- the shards on disk stayed at whatever the first boot wrote, forever.
        Blindly overwriting instead is no better: operators edit these through
        `/mod` -> Docs and would lose that work on every upgrade.

        So each shard we write is recorded in a manifest by content hash. On boot a
        shard is rewritten only when the file still matches the hash we last wrote for
        it -- proof that nobody has touched it since. A shard the operator has edited
        no longer matches, and is left exactly as it is.
        """
        manifest_path = os.path.join(DOCS_DIR, _MANIFEST_FILENAME)

        manifest = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "rb") as f:
                    import orjson
                    manifest = orjson.loads(f.read()) or {}
            except Exception as e:
                # A corrupt manifest must not cost the operator their edits, so treat
                # it as "everything is customised" and only fill in missing files.
                print(f"Documentation manifest unreadable ({e}); leaving existing shards untouched.")
                manifest = {}

        changed = False
        for rel_path, content in DEFAULT_HELP_DOCS.items():
            filepath = os.path.join(DOCS_DIR, rel_path)
            shipped = content.strip()
            shipped_digest = _digest(shipped)

            recorded = manifest.get(rel_path)

            if not os.path.exists(filepath):
                # Absent and previously written means the operator deleted it through
                # /mod -> Docs. Recreating it on the next boot would make that button
                # do nothing that survives a restart, so a known shard stays deleted;
                # only a genuinely new one is written.
                if recorded is not None:
                    continue
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(shipped)
                manifest[rel_path] = shipped_digest
                changed = True
                continue

            if recorded == shipped_digest:
                continue  # Already current.

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    on_disk = f.read().strip()
            except Exception as e:
                print(f"Failed to read documentation shard {rel_path}: {e}")
                continue

            if recorded is not None and _digest(on_disk) != recorded:
                # Operator-edited since we wrote it. Their copy wins, and we stop
                # tracking it so we never reconsider overwriting it.
                continue

            if on_disk == shipped:
                manifest[rel_path] = shipped_digest
                continue

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(shipped)
            manifest[rel_path] = shipped_digest
            changed = True

        try:
            import orjson
            os.makedirs(DOCS_DIR, exist_ok=True)
            with open(manifest_path, "wb") as f:
                f.write(orjson.dumps(manifest))
        except Exception as e:
            print(f"Failed to write the documentation manifest: {e}")

        return changed

    def _rebuild_doc_matrix(self):
        """Collapses `doc_vectors` into one pre-normalised (N, dims) float32 matrix.

        The search path used to rebuild this on *every query*: np.array over a list of
        lists, plus a norm per row, for a corpus that is static between reloads -- and
        it did so inline on the event loop, not in a thread.

        Rows are stored unit length so a query is a bare dot product. Once the matrix
        exists the per-document `emb` is dropped: 256 boxed Python floats is roughly
        8 KB against 1 KB for the same row of the matrix, and nothing reads it again.
        """
        import numpy as np

        self.cog.doc_matrix = None
        vectors = getattr(self.cog, "doc_vectors", None)
        if not vectors:
            return

        try:
            matrix = np.array([doc["emb"] for doc in vectors], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(vectors):
                raise ValueError(f"ragged document embeddings: {matrix.shape}")
        except Exception as e:
            # A cache written at a different dimensionality would land here. Better to
            # disable help RAG with a warning than to raise inside /help at query time,
            # which is where this would previously have surfaced.
            print(f"Failed to build documentation matrix ({e}); help RAG disabled until reload.")
            self.cog.doc_vectors = []
            return

        norms = np.linalg.norm(matrix, axis=1)
        # A zero row stays zero and so scores 0.0, matching the old 1e-10 guard.
        matrix /= np.where(norms == 0.0, 1.0, norms)[:, None]
        self.cog.doc_matrix = matrix

        for doc in vectors:
            doc.pop("emb", None)

    async def _load_and_embed_docs(self):
        """Reads all .txt shards, tags them semantically, and builds the vector database."""
        cache_path = os.path.join(DOCS_DIR, "embedded_docs_cache.json.gz")

        def _sync_walk_docs():
            chunks = []
            # Sorted so the corpus fingerprint depends on the documents rather than on
            # the order the filesystem happens to hand them back.
            for root, dirs, files in sorted(os.walk(DOCS_DIR)):
                dirs.sort()
                for file in sorted(files):
                    if file.endswith(".txt"):
                        filepath = os.path.join(root, file)
                        category = os.path.basename(root).replace("_", " ").title()
                        doc_name = file.replace(".txt", "").replace("_", " ").title()

                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                                if content:
                                    # Prepend semantic tags based on file architecture
                                    tagged_content = f"[Category: {category} - {doc_name}]\n{content}"
                                    chunks.append(tagged_content)
                        except Exception as e:
                            print(f"Failed to read documentation file {filepath}: {e}")
            return chunks

        def _sync_ensure_and_load_cache():
            self._ensure_docs_directory()

            chunks = _sync_walk_docs()

            # Fingerprint of the corpus as it now stands on disk. The cache was
            # previously reused whenever the file existed, so an edited shard -- or a
            # shard a new build revised -- kept answering from the vectors of the text
            # it replaced, with nothing to indicate the documentation had moved on.
            fingerprint = _digest(_CACHE_FORMAT + "\n" + "\n\x00\n".join(chunks))

            if os.path.exists(cache_path):
                try:
                    cached = self.cog.storage_manager._load_json_gzip(cache_path, encrypted=False)
                    if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
                        items = cached.get("docs") or []
                        for item in items:
                            item["emb"] = decode_embedding_b64(item["emb_b64"])
                        return items, chunks, fingerprint
                    # A bare list is a cache from before fingerprinting. It cannot be
                    # verified against the current corpus, so it is rebuilt once.
                    if isinstance(cached, dict):
                        print("Documentation has changed since the vector cache was written; re-embedding.")
                    else:
                        print("Documentation vector cache predates fingerprinting; re-embedding once.")
                except Exception as e:
                    print(f"Failed to load documentation vectors from cache: {e}")

            return None, chunks, fingerprint

        cached_data, chunks, fingerprint = await asyncio.to_thread(_sync_ensure_and_load_cache)
        if cached_data is not None:
            self.cog.doc_vectors = cached_data
            self._rebuild_doc_matrix()
            print(f"Loaded {len(self.cog.doc_vectors)} embedded documentation shards from local cache.")
            return

        self.cog.doc_vectors = []

        api_key = self.cog.storage_manager._get_api_key_for_user(int(defaultConfig.DISCORD_OWNER_ID), "gemini")
        if not api_key:
            print("Warning: No Bot Owner Google API Key found. Skipping documentation vector generation.")
            return

        cache_to_save = []
        for chunk in chunks:
            try:
                emb = await get_embedding_vector(api_key, chunk, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=256, timeout=5.0)
                if emb is None:
                    continue
                b64_emb = encode_embedding_b64(emb)
                self.cog.doc_vectors.append({"text": chunk, "emb": emb, "emb_b64": b64_emb})
                cache_to_save.append({"text": chunk, "emb_b64": b64_emb})
            except Exception as e:
                print(f"Failed to embed documentation chunk: {e}")

        # Only fingerprint a complete corpus. If any shard failed to embed, the cache
        # would otherwise be stamped as current and the missing documents would never
        # be retried.
        if cache_to_save and len(cache_to_save) == len(chunks):
            try:
                payload = {"fingerprint": fingerprint, "docs": cache_to_save}
                await asyncio.to_thread(self.cog.storage_manager._atomic_json_save_gzip, payload, cache_path, encrypted=False)
            except Exception as e:
                print(f"Failed to save documentation embedding cache: {e}")
        elif cache_to_save:
            print(f"Embedded {len(cache_to_save)}/{len(chunks)} documentation shards; cache not written so the rest are retried next boot.")

        self._rebuild_doc_matrix()
        print(f"Loaded and embedded {len(self.cog.doc_vectors)} documentation shards.")

    async def _get_relevant_help_context(self, query: str, guild_id: Optional[int], force_always_respond: bool = False) -> Optional[str]:
        """Performs vector search and returns a strict Protocol Override XML block."""
        if not hasattr(self.cog, 'doc_vectors') or not self.cog.doc_vectors:
            if force_always_respond:
                return self._render_injection("No relevant documentation found.")
            return None
            
        emb = await self.cog.memory_manager._get_embedding(query, guild_id if guild_id else 0, "RETRIEVAL_QUERY")
        if not emb:
            if force_always_respond:
                return self._render_injection("No relevant documentation found.")
            return None
            
        import numpy as np

        doc_matrix = getattr(self.cog, "doc_matrix", None)
        if doc_matrix is None:
            # doc_vectors exists but the matrix does not, so this instance loaded its
            # documents before the matrix was introduced (or a rebuild failed). Build
            # it now rather than falling back to the old per-query path.
            self._rebuild_doc_matrix()
            doc_matrix = getattr(self.cog, "doc_matrix", None)
            if doc_matrix is None:
                return None

        prompt_vec = np.array(emb, dtype=np.float32)
        prompt_norm = np.linalg.norm(prompt_vec)
        prompt_unit = prompt_vec / (prompt_norm if prompt_norm != 0 else 1e-10)

        # Rows are already unit length, so cosine is a bare dot product.
        similarities = doc_matrix @ prompt_unit

        # Only the top few are ever used, so partition for them rather than building a
        # dict per document and sorting the whole corpus -- this runs on the turn path
        # for every Help Mode profile, not just on /help.
        wanted = min(_TOP_DOCUMENTS, similarities.shape[0])
        top_idx = np.argpartition(similarities, -wanted)[-wanted:]
        top_idx = top_idx[np.argsort(similarities[top_idx])[::-1]]

        # High relevance threshold ensures we only answer actual technical questions
        top_chunks = [self.cog.doc_vectors[i]["text"]
                      for i in top_idx if similarities[i] >= _RELEVANCE_THRESHOLD]

        if top_chunks:
            docs = "\n---\n".join(top_chunks)
        elif force_always_respond:
            docs = "No relevant documentation found."
        else:
            return None # Return None to let standard roleplay characters continue standard chat

        return self._render_injection(docs)

    def _render_injection(self, docs: str) -> str:
        """Fills the Help Mode template, which the operator may have overridden via /mod.

        Plain replacement rather than .format(), because documentation and personas
        routinely contain braces and a KeyError here would cost the turn. An override
        that omits a placeholder simply keeps whatever text it has in that position --
        an operator who deleted {menu_map} wanted it gone.
        """
        template = self.cog.global_prompts.get("HELP_MODE_INJECTION", DEFAULT_HELP_MODE_INJECTION)
        rendered = template.replace("{docs}", docs)
        if "{menu_map}" in rendered:
            rendered = rendered.replace("{menu_map}", build_menu_map())
        return rendered
