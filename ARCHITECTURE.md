# Architecture

How MimicAI is put together, and why. Read the [Performance contract](#the-performance-contract)
before changing anything that runs per message or per turn.

---

## The deployment target dictates the engineering

Production is a **GCP e2-micro: 1 GB RAM, 0.25 vCPU baseline (burstable), zram + swap**,
running 24/7 under systemd. Almost every non-obvious decision in this codebase traces back
to that box. It is not a hypothetical constraint being designed around — it is the machine
the bot actually lives on, hosting every session, every child bot and every model call in a
single Python process.

Three consequences run through everything below:

1. **Memory that grows is memory that never comes back.** A 24/7 process on 1 GB has no
   quiet period to recover in. Anything keyed by channel, user or profile is bounded.
2. **CPU is a shared, throttled resource.** At 0.25 vCPU, anything over ~10 ms of pure
   Python stalls Discord's heartbeat. `asyncio.to_thread` does not rescue GIL-bound work —
   it only helps for genuinely blocking syscalls.
3. **Dependencies cost resident set even when idle.** Vendor SDKs were removed in favour of
   hand-rolled REST adapters, which is why `api_service.py` is 1,600 lines.

---

## Process model

Everything is one process, one event loop.

```
BotManager.py
  └── tune_allocator()          glibc M_ARENA_MAX / M_MMAP_THRESHOLD, before any import
  └── commands.Bot              parent gateway connection, max_messages=None
       └── MimicCog             the god-cog: all shared state, all slash commands
            ├── managers        own persisted state
            ├── services        own operations
            ├── listeners       gateway events, inherited by MimicCog
            └── ChildBotManager
                 └── commands.Bot × N     child bots, as asyncio tasks
```

`tune_allocator()` runs at the very top of `BotManager.py`, before any other import. This is
deliberate: `M_ARENA_MAX` only governs arenas that do not yet exist, and pinning
`M_MMAP_THRESHOLD` early keeps large buffers on the `mmap` path where freeing actually
returns pages to the OS. See `cogs/utils/memory_tuning.py` for the full argument. It is a
no-op off glibc.

The same module exposes `maybe_trim_malloc()`, a rate-limited `malloc_trim` for the call
sites that churn the allocator hardest — media transfers especially. It is rate-limited
precisely so that a four-participant round sharing one attachment trims once rather than
four times.

`cogs/utils/loop_probe.py` measures the thing CPU percentage cannot see: how long the loop
thread is actually held. The bot averages a fraction of a percent of a CPU and can still
block for hundreds of milliseconds inside one synchronous step, and it is the block that
drops a gateway heartbeat. The probe sleeps a fixed interval and reports the overshoot,
with RSS sampled alongside. Off unless `MIMIC_LOOP_PROBE` is set — see the module docstring
for the four environment variables.

Secrets are read once. `constants._get_gcp_client()` builds the Secret Manager client on
first use and `_release_gcp_client()` drops it the moment `defaultConfig` is built, because
its gRPC channel holds roughly ten threads — timers, four `event_engine` workers, a
lifeguard — for the life of the process, in a bot that caps its own executor at two. The
client is lazy rather than eager so that a later caller rebuilds it instead of silently
receiving the default.

`max_messages=None` disables discord.py's 1,000-`Message` cache. Nothing reads it — every
delete and edit listener uses the `on_raw_*` variants, which consult the gateway payload
rather than the cache. Left on, that deque is the one baseline term that grows forever.

### Child bots are not subprocesses

`ChildBotManager.launch_bot` constructs an additional `commands.Bot` and runs it as an
asyncio task **in the main event loop**. No subprocess, no socket IPC — `cog.manager_queue`
is a plain in-process `asyncio.Queue`.

This is the single most important thing to understand before touching child bots. Every
child is another gateway connection heartbeating inside the same 1 GB process. They are
created with `Intents.none()` plus guilds, and `max_messages=None` for the same reason as
the parent, which matters more here because the cost would be paid once per child.

### Instance locking

`MimicCog` writes a lock file at boot. An instance that fails to acquire it starts in
**INACTIVE** mode and polls to reacquire, rather than competing for the same data
directory. This is what makes a restart-during-shutdown survivable.

---

## Module layout

```
cogs/
  MimicCog.py              god-cog: all caches, all slash commands, LRUCache definition
  managers/
    storage_manager.py     IOManager: Fernet + zstd + orjson persistence primitives
    profile_manager.py     profile CRUD, personal/borrowed/system resolution, sharing
    session_manager.py     hydration/dehydration, eviction, history derivation
    memory_manager.py      LTM, embeddings, cosine/MMR retrieval, training examples
    server_manager.py      per-guild index, webhooks, global prompts
    child_bot_manager.py   child-bot client lifecycle and command dispatch
  services/
    api_service.py         provider adapters + model instantiation/routing
    generation_service.py  _multi_profile_worker — the core turn-rotation engine
    generation/            mixins: heartbeat, prompt_builder, delivery, regeneration,
                           speak, global_chat, whisper, triggers, image_round
    media_service.py       TTS and image-generation queue workers
    tools_service.py       web grounding, URL context, anti-repetition critic
    help_service.py        RAG over bundled documentation
  listeners/               gateway events; inherited by MimicCog
  gui/                     Discord UI views and modals
  utils/                   constants, helpers, content, fuzzy, http_client, memory_tuning,
                           loop_probe
```

**Managers own persisted state. Services own operations.** Both take a back-reference to
the cog in `__init__` — a transitional dependency-injection pattern, not a finished design.
State still lives on `MimicCog`; the managers and services are extraction seams rather than
independent components.

`GenerationService` is assembled from mixins in `cogs/services/generation/`. The split is by
generation *mode* — a whisper, a regeneration, a global chat and a multi-profile round each
have their own history-assembly and delivery path — rather than by layer.

---

## State and bounded caches

`MimicCog.__init__` is where every shared cache is declared, and nearly all of them are
`LRUCache` — an `OrderedDict` subclass (defined in `MimicCog.py`) that moves keys on read
and evicts the oldest past `max_size`.

```python
self.user_indices          = LRUCache(max_size=20)
self.server_indices        = LRUCache(max_size=50)
self.decrypted_key_cache   = LRUCache(max_size=100)
self.content_rating_cache  = LRUCache(max_size=512)
self.channel_models        = LRUCache(max_size=CHANNEL_MODEL_CACHE_MAX_SIZE)
...
```

**The rule: any dict keyed by channel, user or profile must be an `LRUCache` or have an
explicit eviction path.** A plain dict on any of those keys grows for the life of the
process. `channel_models` and `channel_model_last_profile_key` were both plain dicts once;
that is why they are called out here.

Sessions have their own eviction: `session_last_accessed` plus an `eviction_heap`, swept by
`SessionManager.evict_inactive_sessions_task`. An evicted session is *dehydrated* to disk,
not lost — `_ensure_session_hydrated` reloads it on the next trigger.

---

## Storage

```
users/<user_id>/
  index.json                          plaintext: name -> pid maps (personal/borrowed/system)
  keys.json.gz                        API key slots + provider assignments
  shares.json.gz                      incoming profile shares
  profiles/<pid>/
    profile.json.gz                   unified: {name, config, prompts, child_bot}
    ltm.json.gz                       long-term memories + b64 float16 embeddings
    training.json.gz                  few-shot examples + embeddings
    global_chat.json.gz               `/profile global_chat` log, keyed (host, profile)
servers/<guild_id>/
  index.json                          active sessions, user profile prefs, key pointers
  api_keys.json.gz
  webhooks.json.gz
  sessions/<channel_id>/multi/session_log.json.gz
public_profiles/                      the shared library index
borrows.json                          plaintext reverse index: source profile -> borrowers
mod/                                  blacklist, global prompt overrides, docs
```

`index.json` also carries `key_file_stamp`, the `[mtime_ns, size]` of `keys.json.gz` that
`has_personal_key` was computed from, so the boot-and-hourly consistency check verifies the
flag with a stat rather than a decrypt.

### The store answers lookups, not queries

Every read is "give me this path": `user_id -> pid -> shard`. There are no joins and no
ad-hoc queries, which is why a filesystem is the right shape for it and why SQL would buy
a query planner nothing uses while costing per-shard encryption, `rmtree` deletion and the
blast radius of a single corrupt file.

The exception is the question a key-value store cannot answer: *who points at this?*
`is_profile_distributed` and `_cascade_delete_borrowed_profiles` both ask it, and both used
to walk every user directory and decrypt every borrowed profile config. `borrows.json` is
the index built for that one question -- see CLAUDE.md for the invariants that keep it
honest. A new question of that shape gets its own derived index; it does not get a scan.

### Every `.json.gz` is Fernet-encrypted zstd

Not gzip. The extension is historical and nothing depends on it. `index.json` files are
plaintext orjson, since they hold only name-to-PID mappings.

The reader detects the format rather than trusting the name: `read_json_gzip` decrypts
first, attempts `zstd.decompress`, and falls back to `gzip.decompress` on `ZstdError` — so
archives written by older builds still load.

Writes are atomic: temp file, then `os.replace`.

### zstd contexts are thread-local, and must stay that way

`ZstdCompressor` and `ZstdDecompressor` each own a native `ZSTD_CCtx` / `ZSTD_DCtx`, and the
C backend releases the GIL while working on it. Two threads sharing one instance therefore
run libzstd on the same context concurrently and corrupt it.

Every `IOManager` read and write is reached through `asyncio.to_thread`, so *any*
module-level compressor singleton is a live data race. Use the `_get_compressor()` /
`_get_decompressor()` helpers. Never hoist a context to module scope.

All persistence goes through `IOManager` / `StorageManager`. There is no direct file I/O
elsewhere, and adding some would bypass both the encryption and the atomic-write guarantee.

---

## The turn engine

`GenerationService._multi_profile_worker(channel_id)` is one long-lived task per active
channel. Its shape:

1. **Hydrate.** `_ensure_session_hydrated` pulls the session from memory or disk, and
   lazily validates every participant (once per distinct owner, not once per participant —
   that scan decrypts profile files).
2. **Block on the queue.** `session['task_queue'].get()` waits for a trigger.
3. **Batch.** Everything already queued is drained into one round, so a burst of messages
   produces one round rather than one per message.
4. **Yield.** A queued whisper or an in-flight purge/regeneration takes precedence, using
   flag counters rather than polling — see the comments around `whisper_waiting` for why
   the naive version starves.
5. **Normalise triggers** (`triggers.py`) into the round's user-side history: messages,
   reactions, replies, proactive kicks.
6. **Optionally generate one image** (`image_round.py`) *before* any participant speaks, so
   every turn in the round can see it.
7. **Rotate.** Each participant in turn: build its prompt, call its model, deliver.

### Histories are derived, never maintained

There is exactly one `unified_log` per session. Each participant's view of the conversation
is computed from it by `SessionManager._build_history_for_participant`, which walks the
log's tail and, for each turn, assigns `role: 'model'` when `speaker_pid` matches this
participant and `role: 'user'` otherwise — so every profile sees its own past messages as
its own and everyone else's as input. The same pass filters private turns to their owner
(a `whisper` reaches only its `target_pid`, a `private_response` only its speaker), merges
consecutive same-role turns into one entry, and attaches per-profile context — URL
documents and grounding summaries — only if that profile has the corresponding tool
enabled.

**Do not add per-participant history objects.** With a cast of up to 200, storing a
per-participant history is a multiplicative memory cost for data that is a pure function of
the log.

---

## Providers

`APIService._instantiate_model` routes a raw model name to an adapter:

| Prefix | Adapter | Notes |
|---|---|---|
| `GOOGLE/` | `GoogleRESTModel` | Hand-rolled REST over `httpx` |
| `OPENROUTER/` | `OpenRouterModel` | OpenAI-compatible chat completions |
| `OLLAMA/` | `OllamaModel` | User-supplied host URL, per profile |
| *(bare)* | heuristic | A `/` in the name, or `grok`/`anthropic`, implies OpenRouter |

Prefixes are **case-sensitive**, because OpenRouter hosts models under lowercase creator
namespaces like `google/gemini-2.5-flash` and the two must not collide.

### No vendor SDKs

`google-genai` was removed and replaced with the REST adapter. This dropped roughly 70 MB
of import baseline, plus the `websockets` / `requests` / `pydantic` the SDK pulled in for
Live API and Vertex paths the bot never touched. `HarmCategory` and `HarmBlockThreshold`
survive as plain attribute holders in `constants.py` wrapping the same bare strings the REST
API accepts, so existing call sites read unchanged.

If you are tempted to add an SDK for a new provider: write the adapter instead.

### One shared HTTP client

`cogs/utils/http_client.get_shared_client()` returns a single process-wide
`httpx.AsyncClient`. Constructing one is expensive on the target — a fresh
`ssl.SSLContext`, the certifi CA bundle parsed into OpenSSL X509 objects, roughly 14 ms and
~0.8 MB of native allocation. Ten-odd call sites were each doing that per request, paying
for it twice: the transient buffers, and the heap fragmentation left behind by allocating
and freeing at that rate. It also discarded connection reuse.

Use the shared client. Do not construct `httpx.AsyncClient` in a request path.

---

## Memory and embeddings

Long-term memories and training examples are stored with their vectors inline, as
**base64 of `float16`** (`encode_embedding_b64`), decoded to `float32` for maths.

Vectors are **256-dimensional**, truncated from the embedding model's native output using
Matryoshka Representation Learning. The quality loss is small; the disk and RAM saving is
not.

Similarity is computed by decoding the whole candidate set into one stacked `(N, dims)`
matrix and issuing **a single BLAS call**:

```python
matrix = np.frombuffer(raw_bytes, dtype=np.float16).reshape(len(b64_embs), -1).astype(np.float32)
```

**Never loop cosine per item.** Retrieval also runs a vectorised MMR pass for diversity,
operating on the already-decoded matrix rather than re-decoding base64 per candidate.

---

## Prompt assembly

Context reaches models as XML-ish tags: `<persona_profile>`, `<archive_context>`,
`<whisper_context>`, `<training_data>`, `<neuro_endocrine_engine>`, and others. The full
list is `SYSTEM_XML_TAGS` in `constants.py`.

`_scrub_response_text` strips these from model output, along with reasoning blocks,
identity headers, timestamps and generation metadata, using regexes compiled once at import
from that same list.

**A new tag must be registered in `SYSTEM_XML_TAGS`, or it leaks into user-visible
messages.** This is the most common way to ship a visible bug here.

### Where a block goes

Two destinations, and the choice is not stylistic.

**The system instruction** (`_construct_system_instructions`) carries standing context —
what is true for the whole scene. It is assembled in three bands:

| Band | Blocks | Why there |
|---|---|---|
| stable | `scene_prompt`, `persona_profile`, `character_instructions` | changes rarely, so it is the cacheable prefix |
| volatile | `session_synopsis`, `game_context`, `neuro_endocrine_engine`, `time_context`, `training_data`, `archive_context`, `negative_constraints` | changes per turn or per minute |
| trailing | `context_rules`, `content_policy` | output-format and hard-content rules, last for recency |

Providers cache on a shared prefix, so **the first block that changes invalidates every
token after it**. `<time_context>` is formatted to the minute; with the persona behind it
the largest stable part of every prompt was re-billed uncached every time the clock ticked.
A new block goes in `volatile_parts` unless it is genuinely per-profile-stable.

**The final user turn** carries per-turn context — what is true for *this* round:
`whisper_context` recaps, `external_context`, `document_context`, help context, media, and
image notes. Retrieval that serves the turn sits next to the turn.

Recency is real. `<content_policy>` is last deliberately, and `<rewrite_request>` — the
in-character `/speak` directive — goes in the **final user turn, after the whole
transcript**, because every other block tells the model to continue the conversation as
itself. Moved anywhere earlier it loses, and the character answers the transcript instead
of re-voicing the author's line. No error, correct-looking output, wrong message.

### What goes *in* a block

State facts; do not give stage directions. A block is read by a character who already has
a persona, training examples, an emotional state and the whole transcript — so a sentence
telling it *how to behave* is competing with all of that, from the highest-attention
position in the prompt, and it wins by flattening the character.

`<image_context>` is the worked example. "You have just generated the following image
based on the prompt 'X'" is a fact the model cannot derive. The sentence that used to
follow it — "Present it with a comment." — could not be what causes a turn (the round is
already generating one); what it supplied was a *register*, and *present* and *comment*
are gallery-attendant words. Every profile answered its own image in the same obliging
voice, which is the assistant voice the rest of the stack exists to suppress.

Three sentences survive the test:

- **A fact the model cannot derive** — that it made the image, that a whisper is private,
  that training examples are not conversation history.
- **A protocol constraint** — word caps, "dialogue only, no XML tags", the
  `<neuro_update>` format spec. These are parser contracts, not personality; deleting one
  breaks something quietly.
- **A directive fighting a structural attractor** — `<rewrite_request>`, and "reply
  directly to this whisper" when the whisper competes with a whole public transcript.

Everything else is flab, and three shapes of it recur. A **menu** ("gloat, sulk,
congratulate, blame the deck") is worse than a bare directive: it anchors on its examples
however it is hedged afterwards. **Shouted compliance** (`You MUST`, `CRITICAL:`, `STRICT
ADHERENCE REQUIRED:`) pulls a persona toward assistant-compliance, and when six blocks are
critical none are. And **"in character"** only parses if the model conceives of itself as
an actor who could also *not* be in character — it posits the out-of-character default and
then asks politely for the other one.

When a directive still feels necessary, look for the missing fact instead.

This applies to the character-facing blocks. Utility prompts — the classifier, the
summariser, the critic, the grounding researchers — are talking to a model acting as a
tool, where there is no character to distort and the shouting is holding a parsed output
format in place.

---

## The optional native core

`mimic_core` (Rust/C) is probed at import for three functions:

- `calculate_similarities_b64` — `memory_manager.py`
- `count_tokens` — `helpers.py`
- `scan_repetition` — `helpers.py`

It is optional. Every call site is guarded by `_HAS_NATIVE_CORE` and `hasattr`, with a
NumPy or pure-Python fallback beside it. **Keep both paths working and behaviourally
identical** — most installs will not have the extension built.

---

## The performance contract

Before adding anything to a per-message or per-turn path, ask:

**Does it allocate per call what could be allocated once?**
Client objects, SSL contexts, compiled regexes and parsed config belong in a cache, not the
hot path.

**Does it block the event loop?**
Anything over ~10 ms of pure CPU stalls Discord heartbeats. `asyncio.to_thread` helps only
for genuinely blocking syscalls, not for GIL-bound work.

**Is the growth bounded?**
Any dict keyed by channel, user or profile must be an `LRUCache` or have an eviction path.

### Do not reintroduce these

- **Unbounded plain dicts** for `channel_models` / `channel_model_last_profile_key`.
- **Per-participant history objects.** Derive from `unified_log`.
- **Inline media download-and-upload.** Go through `_resolve_media_uri` — a TTL'd, bounded
  cache with single-flight, so concurrent rounds share one upload.
- **`gc.collect()` per participant per round.** That was ~11 ms of GIL-held CPU per
  participant. Refcounting frees image buffers when the last reference drops.
- **A module-level zstd compressor or decompressor.** See above — it is a data race, not a
  style preference.
- **Free/premium user tiering.** Removed 2026-08-19. Limits now use the former premium
  values as the baseline: 100 profiles, 100 borrowed, 5,000 LTM entries, 100 training
  examples.

---

## Conventions

- **`orjson` throughout**, not stdlib `json`. It returns `bytes` — open files in binary
  mode.
- **`utils/helpers.Timeout` uses `signal.alarm`** — main thread only, whole-second
  granularity. Never call it inside `asyncio.to_thread`.
- **Australian/British spelling** in user-facing strings and comments ("synchronised",
  "behaviour"). Match the surrounding file.
- **Long explanatory comments are the house style.** Where a decision looks strange, the
  reasoning is written down next to it. Preserve those comments when refactoring; they are
  the record of what was already tried.

---

## Provenance

The codebase was generated by AI under human direction, beginning with Gemini 2.5 Pro in
Google AI Studio. That history is visible in the structure: a god-cog that grew before it
was split, mixins layered onto services rather than a redesign, and comments that argue at
length for their own code. The architecture notes above describe what is there now, not
what a clean-sheet design would have produced.
