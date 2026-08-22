import os
import asyncio
from typing import Optional

from ..utils.constants import defaultConfig, DOCS_DIR, DEFAULT_HELP_MODE_INJECTION
from ..managers.memory_manager import encode_embedding_b64, decode_embedding_b64
from .api_service import get_embedding_vector


class HelpService:
    """Owns RAG guide-vector generation and search for /help and documentation lookups.

    Holds a back-reference to the parent cog for shared instance caches and cross-manager lookups,
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog

    async def _ensure_guide_vectors(self, guild_id: Optional[int]):
        if hasattr(self.cog, 'guide_vectors') and self.cog.guide_vectors:
            return
            
        api_key = None
        if guild_id:
            api_key = self.cog.storage_manager._get_api_key_for_guild(guild_id)
        if not api_key:
            owner_id = int(defaultConfig.DISCORD_OWNER_ID)
            api_key = self.cog.storage_manager._get_api_key_for_user(owner_id, "gemini")
        
        if not api_key: return
        
        try:
            from ..utils.content import DOC_CATEGORIES, HELP_CATEGORIES
            chunks = []
            for d in [DOC_CATEGORIES, HELP_CATEGORIES]:
                for cat, pages in d.items():
                    for page, text in pages.items():
                        chunks.append(f"[{cat} - {page}]\n{text}")
            
            self.cog.guide_vectors = []
            for chunk in chunks:
                try:
                    emb = await get_embedding_vector(api_key, chunk, task_type="RETRIEVAL_DOCUMENT", output_dimensionality=256, timeout=5.0)
                    if emb is None:
                        continue
                    self.cog.guide_vectors.append({"text": chunk, "emb": emb})
                except Exception as e:
                    print(f"Guide embedding failed: {e}")
        except Exception as e:
            print(f"Failed to generate guide vectors overall: {e}")

    
    def _ensure_docs_directory(self):
        """Creates the documentation directory structure and writes default shards if missing."""
        for rel_path, content in DEFAULT_HELP_DOCS.items():
            filepath = os.path.join(DOCS_DIR, rel_path)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content.strip())

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

        def _sync_ensure_and_load_cache():
            self._ensure_docs_directory()
            if os.path.exists(cache_path):
                try:
                    cached_data = self.cog.storage_manager._load_json_gzip(cache_path, encrypted=False)
                    if cached_data and isinstance(cached_data, list):
                        for item in cached_data:
                            item["emb"] = decode_embedding_b64(item["emb_b64"])
                        return cached_data
                except Exception as e:
                    print(f"Failed to load documentation vectors from cache: {e}")
            return None

        cached_data = await asyncio.to_thread(_sync_ensure_and_load_cache)
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

        def _sync_walk_docs():
            chunks = []
            for root, dirs, files in os.walk(DOCS_DIR):
                for file in files:
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

        chunks = await asyncio.to_thread(_sync_walk_docs)

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

        if cache_to_save:
            try:
                await asyncio.to_thread(self.cog.storage_manager._atomic_json_save_gzip, cache_to_save, cache_path, encrypted=False)
            except Exception as e:
                print(f"Failed to save documentation embedding cache: {e}")

        self._rebuild_doc_matrix()
        print(f"Loaded and embedded {len(self.cog.doc_vectors)} documentation shards.")

    async def _get_relevant_help_context(self, query: str, guild_id: Optional[int], force_always_respond: bool = False) -> Optional[str]:
        """Performs vector search and returns a strict Protocol Override XML block."""
        if not hasattr(self.cog, 'doc_vectors') or not self.cog.doc_vectors:
            if force_always_respond:
                docs = "No relevant documentation found."
                template = self.cog.global_prompts.get("HELP_MODE_INJECTION", DEFAULT_HELP_MODE_INJECTION)
                return template.replace("{docs}", docs)
            return None
            
        emb = await self.cog.memory_manager._get_embedding(query, guild_id if guild_id else 0, "RETRIEVAL_QUERY")
        if not emb:
            if force_always_respond:
                docs = "No relevant documentation found."
                template = self.cog.global_prompts.get("HELP_MODE_INJECTION", DEFAULT_HELP_MODE_INJECTION)
                return template.replace("{docs}", docs)
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

        sims = [{"text": doc["text"], "sim": float(similarities[i])} for i, doc in enumerate(self.cog.doc_vectors)]
        sims.sort(key=lambda x: x["sim"], reverse=True)
        
        # High relevance threshold ensures we only answer actual technical questions
        top_chunks = [x["text"] for x in sims if x["sim"] >= 0.60][:5]
        
        if top_chunks:
            docs = "\n---\n".join(top_chunks)
        elif force_always_respond:
            docs = "No relevant documentation found."
        else:
            return None # Return None to let standard roleplay characters continue standard chat
            
        template = self.cog.global_prompts.get("HELP_MODE_INJECTION", DEFAULT_HELP_MODE_INJECTION)
        # String replacement used instead of .format() to prevent KeyError if documentation contains curly braces
        protocol_block = template.replace("{docs}", docs)
        return protocol_block

# --- Default Help Document Shards ---
DEFAULT_HELP_DOCS = {
    # PROFILES CATEGORY
    "profiles/identifiers_and_pids.txt": (
        "Concept: While users see aesthetic names (e.g., 'Detective'), the system identifies profiles exclusively via 16-character Profile IDs (PIDs).\n"
        "Prefixes: Personal Profiles use 'A' (e.g., `A0A1B2C3...`), Borrowed Profiles use 'B' (e.g., `B0A1B2C3...`), and System Profiles use 'X' (e.g., `X0A1B2C3...`).\n"
        "Limits: You can maintain a maximum of 100 Personal Profiles on your account.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Name already exists' when renaming or creating. Fix: Every name is tied to an immutable PID. Choose a unique local name to map to your profile."
    ),
    "profiles/personal_vs_borrowed.txt": (
        "Concept: Every profile has a Profile ID (PID) whose first letter is its class. A = Personal Profile, fully editable and owned by you. B = a profile you borrowed through a private share code. C = a profile you borrowed from the Public Library. X = a System Profile provided by the bot operator. Borrowed profiles (B and C) are read-only links pointing back to the creator's master profile.\n"
        "B versus C: the letter records where the borrow came from, not what it is now. A PID never changes, so if the owner later unpublishes a profile your borrow keeps its C. Borrows created before the C class existed are all B regardless of origin, so the letter is a hint for reading IDs rather than something the bot decides behaviour from.\n"
        "Limits: You can borrow a maximum of 100 profiles from other users.\n"
        "Mechanism: If the original owner deletes or renames their Personal Profile, the Cascade Deletion Protocol instantly severs all borrowed links. The borrowed variant will be automatically deleted from your list to prevent database corruption.\n"
        "Naming: Your own profiles always win a name clash. If you create a personal profile with the same name as a System Profile, yours is the one that runs; the System Profile is only reached when you have no profile of your own by that name."
    ),
    "profiles/sharing_and_cloning.txt": (
        "Concept: 'Sharing' generates a temporary 5-minute Share Code allowing others to borrow a read-only link. 'Cloning' generates a 5-minute Clone Code to copy the configuration into a brand-new, independent Class A profile.\n"
        "Limitations: Cloning severs the link to the original, allowing full editing. However, Long-Term Memories (LTM) and Child Bot configurations are deliberately scrubbed during clones to protect privacy."
    ),
    "profiles/public_hub_publishing.txt": (
        "Concept: Publishing to the global Public Library (`/profile hub`) allows any user to borrow your profile.\n"
        "Mechanism: Before hitting the public index, an automated safety validation (`AUTO_MODERATOR`) evaluates the profile name, display name, and avatar. Explicit or graphic profiles, and any profile rated 'Adult 18+', are blocked from global publishing -- only 'General' profiles can be published. A profile whose Content Rating is still 'Pending' is also refused: publishing is the one place that will not proceed without a verdict. Open the profile dashboard to run the check, then publish."
    ),
    "profiles/child_bot_sync.txt": (
        "Concept: Override options allow you to customise Webhook appearances. If linked to a Child Bot, the system synchronises this identity by updating the actual bot application.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Child bot appearance changed too frequently.' Fix: Discord strictly limits application avatar updates (2 changes per 10 minutes). The system enforces cooldowns to prevent API bans. Wait 10 minutes before trying again."
    ),
    "profiles/safety_and_nsfw.txt": (
        "Concept: A profile carries exactly one Content Rating, and everything else follows from it -- which channels it may run in, which provider harm thresholds apply, and whether it may be published. The separate 'Safety Level' setting was removed; it was a second field for the same decision and the two could disagree. Profiles still carrying the old 'Unrestricted 18+' value are migrated to an owner-declared Adult rating on first read, so no 18+ profile is released into general channels by the change.\n"
        "Content Rating states: 'General' runs in any channel. 'Adult 18+' runs only in age-restricted channels. 'Pending' means no verdict has been reached yet and behaves as General, unless the operator enabled fail-closed mode. 'Exempt' is a bot-operator setting that skips classification entirely.\n"
        "Declaring 18+ yourself: The profile dashboard Home tab has a 'Declare Adult Content 18+' action. It sets the Adult rating directly, with no classifier call and no waiting, and you can withdraw it again at any time -- withdrawing hands the profile back to the classifier for a fresh verdict. Use it when you already know a profile belongs in age-restricted channels. Declared profiles are not sent to the classifier at all, since Adult is already the strictest verdict it could return.\n"
        "Content Classification: An LLM classifier reads the profile persona, instructions, name, display name and image prompt, and returns General or Adult. An Adult verdict forces the profile to age-restricted channels regardless of what its owner wanted. The classifier can only ever make a profile stricter -- it can never unlock an 18+ profile, and it cannot withdraw an owner's own declaration. Verdicts are keyed to a hash of the profile content, so editing the persona re-runs the check automatically and deleting and recreating an identical profile reaches the same verdict. Classification runs in the background after an edit and never blocks a reply.\n"
        "Why a profile was rated Adult: The dashboard shows a broad category -- 'Explicit sexual content', 'Sexually-focused persona', 'Graphic violence', 'Minor safety', or a generic 'Adult themes' -- and never a description of the persona itself. The classifier is constrained to return one of those categories rather than prose, so nothing it writes about a profile is stored or shown back.\n"
        "Disputing a verdict: A profile owner cannot clear a classifier Adult verdict; only the bot operator can, via the mod view of the profile. (An Adult rating you declared yourself is different -- that one you can always withdraw.) A clearance is tied to the exact content that was flagged, so it lapses if the persona is edited afterwards.\n"
        "NSFW Gating: A profile rated Adult 18+ is checked against `is_nsfw()` on the destination channel before the prompt is sent. For a borrowed profile the rating is taken from the original owner's profile, so a borrower cannot downgrade it.\n"
        "Channel Content Policy: In any channel that is NOT age-restricted, the system appends a `<content_policy>` note to the system instruction asking the model to keep the response suitable for a general audience. This applies to every provider, including OpenRouter and Ollama, and is editable by the bot operator via `/mod` as the CONTENT_POLICY prompt.\n"
        "Provider Filters: On Google models the harm thresholds follow the destination channel, not the profile: an age-restricted channel sends BLOCK_NONE, every other channel sends BLOCK_ONLY_HIGH. Since only cleared profiles ever reach an age-restricted channel, the filter and the channel gate can no longer disagree. The single exception is an operator-set Exempt profile, which sends BLOCK_NONE everywhere. Non-Google providers ignore these thresholds entirely, which is why the content policy note exists.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Adult profile is silent or throwing errors.' Fix: Adult profiles are strictly blocked in standard channels. You MUST set the channel to Age-Restricted (NSFW) in your Discord settings to use them.\n"
        "- Symptom: 'The Declare Adult Content 18+ action is missing from the dropdown.' Fix: It is hidden for borrowed profiles, for profiles published to the Public Library, and while a classifier Adult verdict or an operator Exemption stands. Unpublish the profile first, or edit the original if it is borrowed.\n"
        "- Symptom: 'My profile still says Restricted or Unrestricted 18+ somewhere.' Fix: That wording is gone. The old stored value is folded into the Content Rating the first time the profile is read and written out the next time it is saved.\n"
        "- Symptom: 'My profile was flagged Adult and I cannot change it back.' Fix: The declaration action is withheld while a classifier Adult verdict stands. Edit the persona to remove the adult material, which re-runs the check, or ask the bot operator to clear the verdict.\n"
        "- Symptom: 'Content Rating is stuck on Pending.' Fix: Classification needs an OpenRouter or Google API key -- your own, or the bot operator's. Without one no verdict can be reached and the profile stays Pending, which behaves as General unless the operator enabled fail-closed mode. Publishing is refused while Pending."
    ),

    # APIS CATEGORY
    "apis/google_gemini.txt": (
        "Requirements: A Google GenAI API Key submitted via the `/settings` DM command.\n"
        "Capabilities: Powers standard text generation. It is the ONLY provider that natively supports Google Search Grounding and direct URL fetching.\n"
        "Image Generation Constraints: Generating images (`!image`) requires an active billing account configured in Google AI Studio. Google's free-tier keys completely block image model execution.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The bot isn't responding in the server' or 'Bot is silent'. Fix: Ensure the server admin has added a Google API Key via `/settings` -> Server Primary Key, or that you have added your own Personal Key.\n"
        "- Symptom: 'Image generation failed' or 'Paid Key Required'. Fix: You are likely using an unpaid free-tier Google API key. You must configure billing in Google AI Studio to unlock image models."
    ),
    "apis/openrouter.txt": (
        "Requirements: An OpenRouter API Key submitted via the `/settings` DM command.\n"
        "Capabilities: Allows users to access non-Google models like Anthropic's Claude, Meta's Llama, and xAI's Grok.\n"
        "Limitations: OpenRouter models do NOT have native access to Google Search or URL fetching. To use Grounding or URL Context with OpenRouter, you MUST go into `/profile manage` -> Tools -> and set Grounding/URL Context to RAG Mode.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'My Claude profile is hallucinating web links' or 'Grounding failed with Claude'. Fix: You must set your Grounding mode to 'RAG'. Native grounding only works with Google.\n"
        "- Symptom: 'Rate Limit Error.' Fix: Your OpenRouter account lacks sufficient credit balance to process the request."
    ),
    "apis/ollama.txt": (
        "Requirements: Ollama installed on your local machine, and the specific model downloaded via your terminal (e.g., `ollama run llama3`).\n"
        "Setup: Go to `/profile manage` -> Params -> Models. Click the 'API' button until it says 'Ollama'. Click 'Set Host URL'.\n"
        "Remote Hosting: If your Discord bot is hosted on a cloud server, it cannot see your home PC's 127.0.0.1 address. You MUST expose your local Ollama port using a secure SSH tunnel (e.g., `ssh -R 80:localhost:11434 nokey@localhost.run`) and paste the resulting HTTPS link into the Host URL setting.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Ollama is offline' or 'Network Error'. Fix: Ensure the Ollama app is actively running on your PC, and that the Host URL in the bot matches your tunnel address."
    ),
    "apis/rate_limits_and_errors.txt": (
        "Concept: API execution errors occur when an external inference endpoint rejects a generation payload. The bot automatically parses these HTTP status codes to provide diagnostics.\n"
        "Status Code Meanings:\n"
        "- 401 (Unauthorized): The API key submitted is invalid, expired, or has been revoked by the provider.\n"
        "- 429 (Resource Exhausted / Rate Limited): The key has hit its Requests-Per-Minute (RPM) or Daily Quota limit. For Google, this is common on free accounts. For OpenRouter, it means your credit balance is empty.\n"
        "- 403 (Access Forbidden): The model is restricted, or the content violated the provider's native safety moderation filters.\n"
        "- 404 (Model Not Found): The selected model has been deprecated by the provider.\n"
        "Failover Protocol: When a 429 or network timeout occurs, the bot temporarily marks the key as cooling down and immediately redirects the payload to your designated Fallback Model."
    ),
    
    # SESSIONS CATEGORY
    "sessions/session_config.txt": (
        "Command: `/session config`\n"
        "Capabilities: Allows server administrators or session owners to configure multi-profile chat sessions. You can add or remove participants, set a global Master Prompt (Director's Note), and change execution modes.\n"
        "Execution Modes: 'Sequential' forces participants to speak in a strict order. 'Random' shuffles the speaker order every round."
    ),
    "sessions/session_swap.txt": (
        "Command: `/session swap [profile] [use_child_bot] [slot]`\n"
        "Capabilities: Allows users to dynamically inject, remove, or swap characters inside an active session without interrupting the conversation. Users can assign specific slots (1-200) to override a specific participant.\n"
        "Delivery Method: The `use_child_bot` parameter forces the profile to reply using a dedicated Discord bot application (Child Bot) instead of a Webhook."
    ),
    "sessions/session_controls.txt": (
        "Capabilities: Users can dynamically control the flow of a chat session using specific message reactions.\n"
        "- Regenerate Emoji (🔁): Edits the bot's message and forces it to generate a new response, effectively performing a time travel edit.\n"
        "- Next Speaker Emoji (⏯️): Triggers the next participant in the cast list to respond.\n"
        "- Mute Turn Emoji (🔇): Hides the targeted message from the bot's memory transcript, making it invisible to the AI.\n"
        "- Skip Participant Emoji (❌): Suspends a specific profile from responding in the session until unskipped."
    ),
    "sessions/response_modes.txt": (
        "Setup: Configured via `/profile manage` -> Tools -> Cycle Response Mode.\n"
        "Modes:\n"
        "- Regular: The bot replies normally without pings.\n"
        "- Mention: The bot prepends the user's Discord ping to the message payload.\n"
        "- Reply: The bot uses the native Discord Reply feature to visually link its response to the user's prompt.\n"
        "- Mention+Reply: A hybrid behavior combining both."
    ),
    "sessions/proactivity_and_director.txt": (
        "Setup: Enabled via `/session config` -> Proactivity.\n"
        "Mechanism: When Proactivity is active, an asynchronous system loop monitors the channel. Based on your configured Trigger Chance (0-100%) and Cooldown, the bot can autonomously initiate conversation.\n"
        "AI Director: If configured, the system uses a secondary model to read the last 10 messages of the conversation. It generates a brief environmental change or sudden event (e.g., 'A loud noise is heard outside').\n"
        "Payload: This scene update is injected as an <internal_note> directly to the cast list, forcing the AI characters to dynamically react to the new situation autonomously."
    ),

    # MEMORY CATEGORY
    "memory/stm_buffer.txt": (
        "Concept: Short-Term Memory (STM) is the volatile context buffer for the immediate conversation.\n"
        "Limits: The STM Length defines how many previous conversational turns are appended to the context window (Max 50). Exceeding this can degrade the AI's adherence to persona instructions.\n"
        "Command: `/refresh` instantly clears the immediate conversational short-term memory buffer for the channel, effectively wiping recent context without deleting long-term records. This is useful if the bot becomes confused."
    ),
    "memory/ltm_archive.txt": (
        "Concept: Long-Term Memory (LTM) is a persistent, vector-embedded archive of past conversations.\n"
        "Creation: The bot tracks the exchange volume. Once the 'Creation Interval' is met, an auxiliary model extracts the conversation and condenses it into a third-person semantic summarisation.\n"
        "Retrieval: When a user speaks, their prompt is embedded and compared against the LTM archive via Cosine Similarity. If the score exceeds the 'Relevance Threshold', the memory is injected as `<archive_context>`.\n"
        "Management: Users can manually add, edit, or delete LTM summaries via `/profile data manage`."
    ),
    "memory/training_examples.txt": (
        "Concept: Stylistic training examples are explicit input-output templates used to dictate a profile's behavioral style, formatting, and tone.\n"
        "Retrieval: Like LTMs, these templates are embedded and retrieved via vector search. Upon matching the relevance threshold, the templates are injected into the system prompt to force the AI to mimic the designated style.\n"
        "Management: Users can configure these style guides via `/profile data manage` -> Training."
    ),
    "memory/context_metadata_and_xml.txt": (
        "Concept: MimicAI utilises a strict XML Partitioning Protocol to keep background technical context completely isolated from the conversational chat.\n"
        "Identity Headers: Every message injected into the AI's history is prefixed with a hardcoded header: `<Name> [ID: PID] [Timestamp]:`. This guarantees that the model maintains perfect chronological awareness and never confuses which character sent which message."
    ),

    # FEATURES CATEGORY
    "features/image_generation.txt": (
        "Commands: `!image` or `!imagine`.\n"
        "Capabilities: Profiles can generate `.png` visuals based on user prompts. The system will automatically inject the profile's 'Appearance' context if second-person pronouns are detected in the prompt.\n"
        "Symptom: Paid Key Required. Image Generation models are completely blocked by Google on free-tier API keys. A valid billing account must be active in Google AI Studio to use this feature."
    ),
    "features/speech_tts.txt": (
        "Concept: Text-To-Speech (TTS) utilises a generative audio model to synthesize voice responses.\n"
        "Configuration: The Director's Desk (`/profile manage` -> Persona -> TTS Instructions) allows users to define the Vocal Archetype, Accent, Dynamics, and Pacing of the voice.\n"
        "Hardware: The Voice Name (e.g., Aoede, Kore) and Speech Model can be selected in the profile's Speech Settings. A lower temperature produces stable audio, while high temperatures may result in audio artefacts."
    ),
    "features/neuro_engine.txt": (
        "Concept: The Neuro-Endocrine Engine simulates biological hormonal states to produce dynamic emotional characterisation.\n"
        "Variables: The system tracks Dopamine (Motivation/Joy), Cortisol (Stress/Anxiety), Oxytocin (Trust/Empathy), and Adrenaline (Urgency).\n"
        "Execution: At the end of every response, the AI evaluates its interaction and outputs a hidden `<neuro_update>` tag, permanently updating its emotional state for the next turn."
    ),
    "features/grounding_and_rag.txt": (
        "Concept: Web Grounding enables real-time Google Search grounding. URL Fetching allows the bot to scrape and read website links provided in chat.\n"
        "Native Mode: Highly accurate and supports inline footnotes. Native search ONLY works with official Google Gemini models.\n"
        "RAG Mode Grounding: Uses an ephemeral Gemini model to scrape and summarise data. RAG mode is entirely model-agnostic and MUST be used if the profile is powered by an OpenRouter or custom AI model."
    ),
    "features/typing_simulation.txt": (
        "Concept: Realistic Typing simulates human interface delays before sending a message.\n"
        "Configuration: Users can define the Typing CPS (Characters Per Second) and Max Delay constraints. The system parses the AI's output, calculates the delay based on sentence length, and streams the text to Discord dynamically."
    ),
    "features/repetition_critic.txt": (
        "Concept: The Anti-Repetition Critic is a secondary AI evaluation layer designed to prevent Model Collapse.\n"
        "Mechanism: Before sending a message, the Critic analyzes the conversation transcript to detect robotic transitions or repetitive linguistic loops. If detected, it forces an explicit negative constraint into the AI's system prompt (e.g., 'Do not use the word acknowledge').\n"
        "Tradeoff: Enabling the critic significantly reduces repetition but adds notable latency to the generation time."
    ),
    "features/multimodal_media_handling.txt": (
        "Concept: Multimodal processing allows profiles to analyze media files (images, audio, video) attached to your Discord messages.\n"
        "Vision Processing: Models with native Vision support (e.g., Gemini Flash/Pro) can analyze image attachments. If a user replies to an image, the image is dynamically injected into the profile's conversational history.\n"
        "Audio & Video: Models can process direct `.wav` or `.mp3` audio files and video files.\n"
        "Limitations & Errors: If you attach an image or audio file to a profile powered by a model that lacks multimodal support (such as many custom OpenRouter or local Ollama models), the API call will fail. The bot will catch this and display: 'Unsupported File Format (Model lacks Vision/Audio support).'"
    )
}