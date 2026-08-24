"""User-facing documentation copy.

Two corpora live here, and they are deliberately not the same text:

* `HELP_CATEGORIES` backs the `/guide` dropdown. It is browsable prose, read by a
  human who is scrolling to learn how something works.
* `DEFAULT_HELP_DOCS` backs Help Mode and `/help ask`. Each entry is written on disk
  as a shard under `mod/docs/`, embedded, and retrieved by vector search. It is
  written for retrieval against a user's *problem*, which is why the shards are terse
  and carry explicit `Symptom:` / `Fix:` pairs -- those lines are what a frustrated
  user's question actually matches on.

They were previously split across two modules, alongside a third structure
(`DOC_CATEGORIES`) that nothing read at all. Keeping both here means a feature change
touches one file, and the drift between what `/guide` says and what Help Mode answers
is visible in a single diff.

The operator may edit the on-disk shards via `/mod` -> Docs. `HelpService` tracks
which shards it wrote and leaves edited ones alone on upgrade.
"""

HELP_CATEGORIES = {
    "1. Getting Started": {
        "What MimicAI Is": (
            "MimicAI runs AI characters -- called **profiles** -- inside Discord.\n\n"
            "A profile holds a persona, a set of instructions, a model choice, sampling parameters, and its own memory. "
            "Profiles speak either through **webhooks** (a distinct name and avatar with no extra bot application) or as a "
            "**child bot** (a real Discord application with its own presence in the member list).\n\n"
            "Profiles talk inside **sessions**. A session is bound to one channel, holds a cast of profiles, and keeps a single "
            "shared transcript that each participant sees from its own point of view.\n\n"
            "Use the dropdowns below to browse the documentation. For a question you would rather just ask, use `/help ask:<your question>`."
        ),
        "First Steps": (
            "**1. Add an API key.** Open a DM with the bot and run `/settings` -> **API Keys**. Pick an empty slot, click **Submit Key**, "
            "then use the assignment dropdown to point it at **Personal** (for DM chats) and at any server you administrate. "
            "Click **Save Assignments** -- nothing is stored until you do.\n\n"
            "**2. Create a profile.** `/profile create profile_name:detective`. If you would rather start from a concept than a blank form, "
            "`/profile generate` drafts a whole profile for you.\n\n"
            "**3. Shape it.** `/profile manage profile_name:detective` opens the dashboard. **Persona** -> Edit Persona for backstory and traits; "
            "Edit Instructions for speech and formatting; Edit Appearance for a display name and avatar URL.\n\n"
            "**4. Start talking.** `/session config` -> add the profile to the cast -> **Start / Update Session**. Then just send a message in the channel."
        ),
        "API Keys and Where They Apply": (
            "You hold four key slots: two Google Gemini, two OpenRouter. Each slot can be assigned to two kinds of scope:\n\n"
            "• **Personal:** used for your DM conversations (`/profile global_chat`) and for your own profiles' background work.\n"
            "• **A server:** used for every session in that server. Only administrators of that server can assign a key to it.\n\n"
            "A server needs *someone* to have assigned a key to it, otherwise its profiles cannot generate at all. "
            "If several people assign keys, the server index records the pointer -- the last assignment wins.\n\n"
            "**Google keys are strongly recommended to be paid tier.** Long-term memory, training retrieval, grounding, the "
            "content classifier and image generation all make their own calls. Free-tier rate limits throttle them badly, and image "
            "generation is blocked on free keys entirely."
        ),
        "Profile Classes (PIDs)": (
            "Every profile has an immutable 16-character **Profile ID**. Its first letter records the class:\n\n"
            "• **A** -- Personal. Yours, fully editable.\n"
            "• **B** -- borrowed through a private share code.\n"
            "• **C** -- borrowed from the Public Library.\n"
            "• **X** -- a System Profile provided by the bot operator.\n\n"
            "Borrowed profiles (B and C) are read-only links back to the creator's master profile. You can override local behaviour "
            "such as timezone and response mode, but the identity stays synchronised with the original -- and if the owner deletes or "
            "renames it, your borrow is severed automatically.\n\n"
            "The letter records where a borrow *came from*, not what it is now. A PID never changes, so a borrow of a profile that was "
            "later unpublished keeps its C, and borrows predating the C class are all B. Read it as a hint, not as a rule the bot enforces.\n\n"
            "**Limits:** 100 personal profiles, 100 borrowed."
        ),
    },
    "2. Writing a Character": {
        "Persona vs Instructions": (
            "The two halves of a profile do different jobs, and mixing them up is the most common cause of a character that will not behave.\n\n"
            "**Persona** answers *who is this*. Backstory, personality traits, likes, dislikes, appearance. It is descriptive. "
            "The Appearance field does double duty: the image generator reads it when the request is for a picture of the character itself.\n\n"
            "**Instructions** answer *how should it write*. Formatting rules, message length, what never to do. It is imperative, and it "
            "takes priority over persona bias during generation. There are four separate instruction blocks so long rulesets do not hit a "
            "single field's character limit."
        ),
        "Instruction Style by Model Type": (
            "How much instruction a model wants depends on what kind of model it is.\n\n"
            "**Non-reasoning models** behave as next-token predictors. They benefit from explicit structure: formatting rules, negative "
            "constraints, worked examples, step-by-step direction.\n\n"
            "**Reasoning models** deliberate internally before answering. Overloading them with micromanaged formatting and long few-shot "
            "sets disrupts that phase and causes intent drift. Suggest the destination; do not force the journey. Keep instructions short, "
            "direct and minimal, and let the model plan.\n\n"
            "If a reasoning-model profile feels stiff or keeps ignoring your persona, try deleting instructions rather than adding them."
        ),
        "Training Examples": (
            "Training Examples are the most direct tool for pinning down a character's voice. Each is an explicit `input -> response` pair "
            "that you author by hand.\n\n"
            "They are not injected wholesale. Each is embedded, and when you speak, your message is matched against the library; only the "
            "examples that clear the relevance threshold are injected for that turn. So a large library stays cheap -- you pay for the "
            "handful that are actually relevant.\n\n"
            "This makes them ideal for situational voice: how the character greets someone, how they react to being insulted, how they "
            "handle a question they cannot answer. Manage them at `/profile manage` -> **Memory** -> Manage Training Examples, and tune "
            "the match with **Set Training Parameters**.\n\n"
            "**Limit:** 100 examples per profile."
        ),
        "Generating a Profile with AI": (
            "`/profile generate` drafts a complete profile -- persona fields and instructions -- from a short concept.\n\n"
            "Treat the result as a first draft. It gives you a structurally complete profile to react to, which is usually faster than "
            "filling in an empty form, but the voice will be generic until you edit it. Generated profiles are ordinary Class A profiles "
            "with no special status."
        ),
    },
    "3. Models and Parameters": {
        "Choosing Models": (
            "Set at `/profile manage` -> **Params** -> Set Models. You choose a **Primary Model** and a **Fallback Model**.\n\n"
            "**Providers:**\n"
            "• **Google Gemini** -- the only provider with native Grounding and native URL fetching.\n"
            "• **OpenRouter** -- Anthropic, Meta, xAI, DeepSeek and many others through one key.\n"
            "• **Ollama** -- models running on your own machine, free to run. Set the host URL from the same menu.\n\n"
            "**Fallback** is not optional infrastructure you can ignore. If the primary request fails -- rate limit, timeout, safety block -- "
            "the payload is immediately re-sent to the fallback. Choose something cheap and fast; a profile with a good fallback stays alive "
            "through an outage that would otherwise silence it."
        ),
        "Core Sampling": (
            "At `/profile manage` -> **Params** -> Set Generation Parameters & STM.\n\n"
            "• **Temperature (0.0 - 2.0):** overall randomness. Around 0.3 is analytical and predictable; around 1.5 is chaotic and inventive. "
            "Most roleplay sits between 0.8 and 1.2.\n"
            "• **Top K:** hard-limits the pool to the K most likely next tokens, cutting off the tail of nonsense.\n"
            "• **Top P (nucleus):** takes tokens in probability order until their combined mass reaches P. The pool shrinks when the model is "
            "unsure and widens when it is confident.\n"
            "• **STM Length (0 - 50):** how many previous turns are sent as context. Higher is more coherent and more expensive; very high "
            "values can dilute the model's adherence to its instructions."
        ),
        "Advanced Sampling and Penalties": (
            "At `/profile manage` -> **Params** -> Set Advanced Parameters. These are honoured by OpenRouter models; Google models ignore them.\n\n"
            "• **Min P (0.0 - 1.0):** an adaptive floor scaled to the top token's confidence. If the best token is at 50% and Min P is 0.1, "
            "anything under 5% is dropped. It generally holds coherence better than Top P at high temperature.\n"
            "• **Top A (0.0 - 1.0):** truncates based on the square of the top token's probability -- prunes hard when the model is confident, "
            "stays permissive when it is not.\n"
            "• **Frequency Penalty (-2.0 - 2.0):** scales with how often a token has already appeared. Pushes vocabulary diversity.\n"
            "• **Presence Penalty (-2.0 - 2.0):** a flat penalty for any token used at all. Pushes topic change.\n"
            "• **Repetition Penalty:** a multiplicative penalty on recent tokens. The blunt instrument for breaking a stuttering loop."
        ),
        "Reasoning and Thinking Budget": (
            "At `/profile manage` -> **Params** -> Set Thinking Parameters. Applies to models with an explicit reasoning phase.\n\n"
            "• **Reasoning Level:** how much effort to spend deliberating (None through XHigh).\n"
            "• **Token Budget:** a hard cap on internal thought, for models that accept one.\n"
            "• **Thinking Summary:** whether the reasoning trace is delivered alongside the reply.\n"
            "• **Signatures:** preserves the provider's thought-signature between turns where supported, so a multi-turn chain of reasoning "
            "stays coherent.\n\n"
            "Reasoning is usually the single largest contributor to latency. Lowering the level is the first thing to try on a sluggish profile."
        ),
        "Making a Profile Faster": (
            "In rough order of effect:\n\n"
            "• **Lower the Reasoning Level**, or cap the token budget.\n"
            "• **Disable the Anti-Repetition Critic.** It is an extra model call before every reply.\n"
            "• **Set Grounding and URL Context to OFF**, or to Native if your model supports it. RAG mode is an extra call each.\n"
            "• **Reduce STM Length**, and reduce LTM context size.\n"
            "• **Switch to a lighter primary model.** A flash-lite variant answers near-instantly at some cost in reasoning depth.\n\n"
            "In a multi-profile session these costs multiply by the number of participants in the round."
        ),
    },
    "4. Memory": {
        "The Three Layers": (
            "**Short-Term Memory** is the running conversation. One shared transcript per session; each profile's view is derived from it at "
            "generation time, so private turns stay private. Depth is the profile's STM Length. `/refresh` clears it for the channel without "
            "touching anything persistent.\n\n"
            "**Long-Term Memory** is a permanent, searchable archive the bot writes for itself.\n\n"
            "**Training Examples** are voice guides you write by hand. See *Writing a Character*.\n\n"
            "The three are independent: clearing STM does not touch LTM, and deleting a memory does not affect training examples."
        ),
        "How Long-Term Memory Works": (
            "**Creation.** The profile counts exchanges. When the **Creation Interval** is reached, the last N turns are sent to a background "
            "model, which condenses them into a short third-person factual summary. You can rewrite the instruction it uses via "
            "**Set LTM Summarisation Prompt**.\n\n"
            "**Retrieval.** Your message is embedded and compared against the archive. Memories scoring above the **Relevance Threshold** are "
            "injected as `<archive_context>`, up to the LTM context size.\n\n"
            "**Where a memory applies.** Every memory belongs to the server it was formed in, and is only ever recalled there. A profile "
            "used in two servers keeps two separate archives, and neither is visible in global chat.\n\n"
            "Turn creation off entirely with **Toggle LTM Auto-Creation** if you would rather curate the archive by hand at "
            "`/profile manage` -> **Memory**.\n\n"
            "**Limit:** 5,000 memories per profile."
        ),
        "Embeddings and Retrieval": (
            "Both memories and training examples are stored as vectors -- numeric fingerprints of their meaning -- so retrieval matches on "
            "sense rather than on exact words. Asking about 'the argument at the docks' finds a memory written as 'a heated confrontation "
            "near the harbour'.\n\n"
            "Vectors are truncated to 256 dimensions using Matryoshka Representation Learning, which keeps almost all of the retrieval quality "
            "at a fraction of the storage. This is by far the cheapest call the bot makes.\n\n"
            "**Relevance thresholds** are the dial that matters. Too high and nothing is ever recalled; too low and irrelevant memories crowd "
            "the context. If a profile keeps dragging in unrelated history, raise the threshold before you start deleting memories."
        ),
    },
    "5. Sessions": {
        "Starting and Shaping a Session": (
            "`/session config` (administrators) opens the session dashboard.\n\n"
            "**Cast** adds and removes participants -- personal, borrowed, or child bots. Up to 200.\n\n"
            "**Config** sets how the round runs: **Toggle Execution** switches between sequential and random turn order, **Edit Master Prompt** "
            "sets the scene every participant sees, **Set Response Limit** caps replies per round, and **Toggle TTS** turns on audio.\n\n"
            "`/session swap` changes the cast live without interrupting the conversation, including into a specific slot. "
            "`/session view` dumps the current configuration and participant status. `/session trigger` forces a round. "
            "`/session audit` reports token usage and diagnostics for the active session."
        ),
        "Reaction Controls": (
            "React to a message to steer the session without typing a command:\n\n"
            "• **🔁 Regenerate** -- discard that reply and generate a new one from the same context.\n"
            "• **⏯️ Next Speaker** -- make the next participant respond.\n"
            "• **🍿 Continue Round** -- run a fresh round for the whole cast.\n"
            "• **🔇 Mute Turn** -- hide that message from the transcript. It stays in the channel but becomes invisible to the profiles.\n"
            "• **❌ Skip Participant** -- suspend one profile from responding until you unskip it.\n\n"
            "Mute is the useful one for repair work: if a reply took the scene somewhere you did not want, mute it and the characters will "
            "carry on as though it never happened."
        ),
        "Reactivity and Proactivity": (
            "**Reactivity** decides whether a profile interjects when it is not its turn. Each participant has a **Chance** percentage rolled "
            "against every user message, and a list of **Wakewords** -- an exact match jumps the queue immediately. Configured under "
            "`/session config` -> Reactivity.\n\n"
            "**Proactivity** lets the cast start talking with nobody prompting them. A timer rolls against your configured chance and cooldown; "
            "when it fires, the **AI Director** -- a separate model -- reads recent history and writes a short environmental beat ('a loud noise "
            "outside'), injected as an `<internal_note>` that the cast reacts to.\n\n"
            "Proactivity is the setting most likely to surprise you on the bill: it generates rounds with no user input. Start with a low chance "
            "and a long cooldown."
        ),
        "Response Modes and Delivery": (
            "**Response Mode** (`/profile manage` -> Tools -> Cycle Response Mode) controls how a reply is attached to your message:\n\n"
            "• **Regular** -- a plain message.\n"
            "• **Mention** -- prepends your ping.\n"
            "• **Reply** -- uses Discord's native reply reference.\n"
            "• **Mention + Reply** -- both.\n\n"
            "**Realistic Typing** (Tools -> Toggle Realistic Typing) splits the reply on sentence boundaries and streams it with a delay based "
            "on length and your configured characters-per-second, instead of posting one block."
        ),
        "Private and One-Off Speech": (
            "**`/whisper`** sends a private message to one participant in an active session. Its reply is ephemeral -- only you see it -- and both "
            "sides stay hidden from the other profiles' view of the transcript. Useful for directing a character without the rest of the cast "
            "reacting to your instruction.\n\n"
            "**`/speak`** (administrators) posts a message as one of your profiles with no generation at all. You supply the text; the profile "
            "delivers it in its own name and avatar.\n\n"
            "**`/profile global_chat`** opens a persistent conversation with one profile, kept separately from any server session and "
            "carried with you across servers. The profile needs a rating of **General**, or an operator **Exemption** -- it does not "
            "have to be published to the Public Library, and never did have to be listed to be talked to privately.\n\n"
            "It is not a DM and not a stream of messages. The whole conversation lives in **one embed**, posted wherever you ran the "
            "command and rewritten in place: the profile's latest reply is the body, your last message is the footer. **Reply** opens a "
            "text box, **Play** sends what is queued, and the 🔒 button decides who may use them. Locked (the default) means you alone; "
            "unlocked means anyone who can see the message, which is how a channel full of people holds one conversation with a "
            "profile. Everyone who replies inside the same ten-second window is queued, and Play sends them as a single turn."
        ),
    },
    "6. Tools and Media": {
        "Web Grounding": (
            "`/profile manage` -> **Tools** -> Toggle Grounding. Cycles **OFF -> NATIVE -> RAG**.\n\n"
            "• **Native** -- Google's own search tool. Accurate, gives inline citations, and works *only* on Google Gemini models.\n"
            "• **RAG** -- a background model searches and summarises before your profile sees the result. Works with every provider, including "
            "OpenRouter and Ollama. Costs one extra call per message.\n\n"
            "If your profile runs on anything other than a Google model, RAG is the only mode that will do anything."
        ),
        "URL Context": (
            "`/profile manage` -> **Tools** -> Toggle URL Context Fetching. Also cycles **OFF -> NATIVE -> RAG**.\n\n"
            "• **Native** -- Google fetches the link server-side, including large files and PDFs. Google models only.\n"
            "• **RAG** -- the bot fetches the page, strips it to text, and injects it as `<document_context>`. Works everywhere.\n\n"
            "Fetched text is truncated to keep it from swallowing the context window."
        ),
        "Image Generation": (
            "Send `!image` or `!imagine` followed by a prompt in a channel where the profile has image generation enabled.\n\n"
            "**What happens:** the image model renders the picture, then the image is shown to the profile's *own* text model, which writes an "
            "in-character comment to send with it. The character presents the image as something it made or found.\n\n"
            "**Self-portraits:** if the prompt asks for a picture of the character, its Appearance text is injected into the image prompt "
            "automatically, so it looks consistent between generations.\n\n"
            "**Edits:** reply to an existing image with `!image` and a new instruction to iterate on it.\n\n"
            "**Output** (`/profile manage` -> Tools -> Set Image Output) sets aspect ratio, resolution and thinking level. Only the options the "
            "profile's image model accepts are offered, and 4K is deliberately absent -- it usually exceeds Discord's attachment limit.\n\n"
            "**Requires a paid-tier Google key.** Image models are blocked on free-tier keys entirely."
        ),
        "Speech and the Director's Desk": (
            "TTS is a generative model, not a synthesiser reading text -- so it is directed with description rather than tuned with sliders.\n\n"
            "**Director's Desk** (`/profile manage` -> Persona -> TTS Instructions) takes plain English: **Archetype** ('a grizzled detective'), "
            "**Accent** ('standard British'), **Dynamics** ('whispering in a cavern'), **Pacing and Style** ('rapid, with a subtle smile').\n\n"
            "**Voice** (Params -> Choose TTS Voice) offers all thirty prebuilt Gemini voices, grouped by gender and described by Google's own "
            "one-word character -- Kore is firm and female, Enceladus breathy and male, Sulafat warm and female. **Set Speech Settings** holds the on/off switch and temperature; keep temperature "
            "near default, since far from it produces audible artefacts. The speech model itself is under Set Models.\n\n"
            "**Audio tags** steer delivery mid-line: `[whispers]`, `[shouting]`, `[laughs]`, `[sighs]`. There is no fixed list.\n\n"
            "**Session audio** (`/session config` -> Config -> Toggle TTS) can deliver text plus audio, audio only, or hold every participant's "
            "audio until the round ends and stitch it into one file."
        ),
        "Attachments the Bot Can Read": (
            "Profiles can read images, audio and video attached to your messages, provided the model behind them supports it. Replying to a "
            "message with an image pulls that image into context too.\n\n"
            "Many OpenRouter and Ollama models are text-only. Attaching media to one of those fails at the provider and surfaces as "
            "'Unsupported File Format (Model lacks Vision/Audio support)'. Switch the profile's model, or leave the attachment off."
        ),
        "Anti-Repetition Critic": (
            "`/profile manage` -> **Tools** -> Toggle Anti-Repetition Critic.\n\n"
            "Before a reply is sent, a lightweight model reviews recent output for structural repetition -- the same opening phrase every turn, "
            "the same sentence rhythm, a verbatim loop. If it finds one, it writes a negative constraint into the prompt banning that specific "
            "pattern for the coming turn.\n\n"
            "It works, and it costs an extra model call per message. Worth it on a long-running session where a character has started to sound "
            "like itself on repeat; not worth it on a fast, casual profile."
        ),
        "Neuro-Endocrine Engine": (
            "`/profile manage` -> **Tools** -> Toggle Neuro-Endocrine Engine.\n\n"
            "The profile carries four variables between turns, each 0-100: **Dopamine** (joy, motivation), **Cortisol** (stress, frustration), "
            "**Oxytocin** (bonding, trust), and **Adrenaline** (urgency).\n\n"
            "Its current state is described in the prompt, and at the end of each reply the model emits a hidden update reflecting how the "
            "exchange affected it. The next turn reads that state back. The effect is a character whose mood carries and shifts over a "
            "conversation rather than resetting each message. Nothing about the mechanism is visible in chat."
        ),
        "Time Awareness": (
            "`/profile manage` -> **Tools** -> Set Time & Timezone. Enable tracking and give the profile an IANA timezone "
            "(`Australia/Sydney`, `Europe/London`).\n\n"
            "The profile's local time is injected as `<time_context>`, and every message in history carries a timestamp. Together these give the "
            "character a real sense of when things happened -- that it is late at night where it is, or that you have not spoken in three days."
        ),
    },
    "7. Sharing and Publishing": {
        "Sharing, Borrowing and Cloning": (
            "There are two ways to give someone a profile, and they are not the same thing.\n\n"
            "**Share** produces a 5-minute code that creates a *borrow*: a read-only link back to your master profile. The borrower gets your "
            "character, and any edit you make reaches them. If you delete or rename the original, their copy is severed.\n\n"
            "**Clone** produces a 5-minute code that creates an *independent copy* -- a new Class A profile the recipient owns and can edit "
            "freely, with no link back to you.\n\n"
            "Cloning deliberately scrubs Long-Term Memories and any child-bot configuration. Memories are conversation history and are not "
            "yours to hand on; child-bot config contains a token.\n\n"
            "Both are generated from `/profile hub`, and both require the profile to be rated. An **Adult 18+** profile can be shared "
            "privately but not published; an **Unrated** one cannot be either until you rate it."
        ),
        "The Public Library": (
            "`/profile hub` -> **Public Library** lists every profile published on this instance. Browse or search it, and borrow anything you "
            "like -- a library borrow is the same read-only link a share code produces.\n\n"
            "**Publishing** puts your profile in that index. It is instant: it reads the Content Rating you already hold rather than running "
            "a fresh check, so there is nothing to wait for.\n\n"
            "Only profiles rated **General** can be published. Adult profiles are refused -- they can still be shared privately -- and so are "
            "Unrated ones. Rate the profile first from `/profile manage` -> Home -> **Content Safety**.\n\n"
            "Unpublish at any time from **Manage My Shares**. Existing borrows survive; the profile just stops being listed."
        ),
        "Content Ratings": (
            "Every profile carries one **Content Rating**. It decides whether the profile can be shared, published or used "
            "in Global Chat -- and, for Adult, which channels it can run in.\n\n"
            "• **Unrated** -- where every new profile starts. It runs normally in your own servers, but cannot be shared, "
            "published, or used in Global Chat.\n"
            "• **Pending** -- submitted, waiting on a verdict. Usually a few seconds.\n"
            "• **General** -- runs anywhere, can be shared, published and used in Global Chat.\n"
            "• **Adult 18+** -- runs only in age-restricted channels. Can be shared privately, but not published to the "
            "Public Library or used in Global Chat (a Global Chat can be opened in any channel, so none of them is "
            "guaranteed age-restricted).\n"
            "• **Exempt** -- set by the bot operator only. Treated as rated: shareable, publishable and usable in Global Chat.\n\n"
            "**Nothing is classified unless you ask.** Open `/profile manage` -> Home -> **Content Safety** and choose "
            "**Submit for Rating**. That dashboard also shows exactly what your profile can and cannot do, and why.\n\n"
            "**Declaring it yourself.** From the same dashboard, **Declare Adult 18+** sets the Adult rating immediately, "
            "with no check and no waiting. It counts as a rating, so a declared profile can be shared privately straight "
            "away. Withdraw it whenever you like -- that returns the profile to Unrated.\n\n"
            "**What gets checked:** the profile name, display name, avatar image, persona and AI instructions. Never your "
            "long-term memories, training examples, or any conversation.\n\n"
            "**If you edit a rated profile,** its rating no longer describes it, and you will be asked whether to re-check "
            "it or set it back to Unrated. If other people have borrowed the profile, it is re-checked automatically so "
            "their copy keeps an accurate rating.\n\n"
            "**Disputes.** A classifier Adult verdict can only be cleared by the bot operator. Editing the persona lets you "
            "submit again, which is usually faster."
        ),
        "Child Bots": (
            "A child bot gives a profile its own Discord application -- its own member-list entry, presence and status -- instead of a webhook.\n\n"
            "Registration is restricted to the bot operator, since it means handing over a bot token. Configure at `/settings` -> **Child Bots**.\n\n"
            "Each child runs inside the same process as the main bot, sharing its event loop; there is no separate program to run. The parent "
            "synchronises the profile's display name and avatar to the application automatically.\n\n"
            "Discord limits application avatar changes to roughly two per ten minutes. The bot enforces its own cooldown to stay under that, so "
            "an appearance edit may not appear immediately."
        ),
        "Export, Import and Deletion": (
            "**`/export`** (DM only) writes selected profiles and their memories to a plaintext `.mimic` file. **`/import`** reads one back, "
            "validating the schema and renaming around any name collisions.\n\n"
            "The export is *plaintext*: it is decrypted on the way out so that it can be moved between instances, which also means anyone "
            "holding the file can read the personas and memories in it. Treat it accordingly.\n\n"
            "**`/privacy`** covers the other direction -- reviewing and deleting your data, up to removing your account's data entirely."
        ),
    },
    "8. Getting Help and Fixing Things": {
        "Help Mode and /help": (
            "**`/help ask:<question>`** answers a technical question about the bot directly, retrieving from this documentation. Nobody else "
            "sees the answer.\n\n"
            "**`/help`** with no question toggles **MimicGuide**, a built-in system profile, in and out of the current session, so you can ask "
            "follow-ups conversationally.\n\n"
            "**Help Mode** (`/profile manage` -> Tools -> Toggle Help Mode) lets one of *your* profiles answer bot questions in character. When "
            "someone asks something technical, the relevant documentation is retrieved and injected for that turn only; the profile answers as "
            "itself. If the question is not technical, nothing is injected and the profile just talks normally."
        ),
        "When a Profile Says Nothing": (
            "In order of likelihood:\n\n"
            "**No API key reaches this server.** An administrator must assign a key to it via `/settings` -> API Keys. Check that "
            "**Save Assignments** was clicked.\n\n"
            "**The profile is rated Adult 18+ and the channel is not age-restricted.** It is blocked before generation. Mark the channel NSFW "
            "in Discord's settings, or use a different profile.\n\n"
            "**Response Mode is set to Mention or Reply.** The profile is waiting to be addressed that way.\n\n"
            "**The session was suspended or the channel purged.** Run `/session view` to see whether a session is actually live.\n\n"
            "**Generation is stuck.** `/cancel` stops whatever the bot is doing in this channel."
        ),
        "Reading API Errors": (
            "• **401 Invalid API Key** -- the key is wrong, expired or revoked. Re-submit it.\n"
            "• **429 Rate Limited** -- on Google, usually a free-tier quota; on OpenRouter, an empty credit balance. The bot cools the key down "
            "and retries through your Fallback Model.\n"
            "• **402 Insufficient Credits** -- OpenRouter balance exhausted.\n"
            "• **403 Access Forbidden / Moderated** -- the model is restricted to your account, or the provider's own safety filter refused the "
            "content.\n"
            "• **404 Model Not Found** -- the model has been deprecated or renamed by the provider. Pick a current one.\n"
            "• **Capability Mismatch** -- no endpoint on OpenRouter serves that model with the features requested.\n"
            "• **Ollama Unreachable** -- the Ollama server is not running, or the host URL no longer matches your tunnel.\n"
            "• **Empty Response** -- the model returned nothing, usually a safety block. The fallback model is tried automatically.\n\n"
            "You can set what users see on failure at `/profile manage` -> Home -> Custom Error Message."
        ),
        "When a Character Feels Wrong": (
            "**Repetitive.** Enable the Anti-Repetition Critic; raise temperature; add repetition penalties on OpenRouter models; add Training "
            "Examples showing variety.\n\n"
            "**Ignoring its persona.** Reduce STM Length -- a very long context dilutes system instructions. Check whether your Instructions "
            "contradict your Persona. On a reasoning model, try cutting instructions down rather than adding more.\n\n"
            "**Breaking character to be helpful.** Add an explicit negative constraint in Instructions. Assistant-tuned models need to be told.\n\n"
            "**Recalling irrelevant things.** Raise the LTM Relevance Threshold, or prune the archive at `/profile manage` -> Memory.\n\n"
            "**Confused about who said what.** `/refresh` clears the short-term buffer. In a large cast, raise STM Length so a full round fits "
            "in context.\n\n"
            "**Talking about being an AI or leaking tags.** Usually a model that is not suited to roleplay. Try a different one."
        ),
        "Command Reference": (
            "**Profiles:** `/profile create`, `/profile generate`, `/profile manage`, `/profile list`, `/profile bulk manage`, `/profile hub`, "
            "`/profile global_chat`\n\n"
            "**Sessions:** `/session config`, `/session swap`, `/session view`, `/session trigger`, `/session audit`\n\n"
            "**In-channel:** `/whisper`, `/speak`, `/refresh`, `/cancel`, `/suspend`, `/purge`, `/clear`\n\n"
            "**Setup and data:** `/settings`, `/export`, `/import`, `/privacy`, `/terms`, `/invite`, `/whoami`, `/viewavatar`\n\n"
            "**Documentation:** `/guide` (this browser), `/help`\n\n"
            "**Operator:** `/mod`, `/shutdown`\n\n"
            "`/suspend` and `/purge` require administrator permission; `/mod` and `/shutdown` are bot-owner only."
        ),
    },
}


# --- Help Mode / `/help ask` retrieval corpus -------------------------------------
#
# Written to disk as `mod/docs/<category>/<name>.txt` on boot, embedded once, and
# searched by cosine similarity per query. Style rules, because they are what makes
# retrieval land:
#
#   * One shard, one subject. A shard covering two things matches neither well.
#   * Lead with `Concept:` or `Command:` so the first sentence carries the topic.
#   * End with `Symptom:` / `Fix:` pairs. Users describe problems, not features, and
#     these lines are what their phrasing actually matches against.
#   * Spell out the full dashboard path. The model quotes these back verbatim.
#
# Editing an entry here updates the on-disk shard on next boot *unless* the operator
# has edited that shard themselves -- see HelpService._ensure_docs_directory.
DEFAULT_HELP_DOCS = {
    # --- PROFILES ---
    "profiles/identifiers_and_pids.txt": (
        "Concept: While users see aesthetic names (e.g., 'Detective'), the system identifies profiles exclusively via 16-character Profile IDs (PIDs).\n"
        "Prefixes: 'A' is a Personal Profile, 'B' a profile borrowed via a private share code, 'C' a profile borrowed from the Public Library, and 'X' a System Profile provided by the bot operator.\n"
        "Limits: A maximum of 100 Personal Profiles and 100 borrowed profiles per account.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Name already exists' when renaming or creating. Fix: Every name is tied to an immutable PID. Choose a unique local name to map to your profile."
    ),
    "profiles/personal_vs_borrowed.txt": (
        "Concept: Every profile has a Profile ID (PID) whose first letter is its class. A = Personal Profile, fully editable and owned by you. B = a profile you borrowed through a private share code. C = a profile you borrowed from the Public Library. X = a System Profile provided by the bot operator. Borrowed profiles (B and C) are read-only links pointing back to the creator's master profile.\n"
        "B versus C: the letter records where the borrow came from, not what it is now. A PID never changes, so if the owner later unpublishes a profile your borrow keeps its C. Borrows created before the C class existed are all B regardless of origin, so the letter is a hint for reading IDs rather than something the bot decides behaviour from.\n"
        "What a borrower may change: local-only settings such as timezone, response mode and generation visual. Persona and instructions stay with the owner, and the Persona tab is hidden entirely on a borrowed profile.\n"
        "Mechanism: If the original owner deletes or renames their Personal Profile, the Cascade Deletion Protocol instantly severs all borrowed links. The borrowed variant will be automatically deleted from your list to prevent database corruption.\n"
        "Naming: Your own profiles always win a name clash. If you create a personal profile with the same name as a System Profile, yours is the one that runs; the System Profile is only reached when you have no profile of your own by that name.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'A borrowed profile vanished from my list.' Fix: The owner deleted or renamed the original. Borrows are links, not copies; ask them to share it again.\n"
        "- Symptom: 'I cannot edit the persona of a profile I borrowed.' Fix: That is by design. Use Profile Cloning in `/profile hub` instead, which produces an independent Class A copy you own."
    ),
    "profiles/sharing_and_cloning.txt": (
        "Concept: 'Sharing' generates a temporary 5-minute Share Code allowing others to borrow a read-only link. 'Cloning' generates a 5-minute Clone Code to copy the configuration into a brand-new, independent Class A profile.\n"
        "Choosing between them: share when you want the recipient to keep receiving your edits; clone when you want them to have their own copy to change freely.\n"
        "Limitations: Cloning severs the link to the original, allowing full editing. However, Long-Term Memories (LTM) and Child Bot configurations are deliberately scrubbed during clones -- memories are conversation history that is not the cloner's to receive, and child bot config contains a bot token.\n"
        "Location: Both are generated from `/profile hub`, or from `/profile manage` -> Home -> Share Profile.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'My share code says expired or invalid.' Fix: Codes live for 5 minutes only. Generate a fresh one.\n"
        "- Symptom: 'I cloned a profile but it has no memories.' Fix: Intended. LTM is never transferred by a clone."
    ),
    "profiles/public_hub_publishing.txt": (
        "Concept: Publishing to the global Public Library (`/profile hub`) allows any user to borrow your profile.\n"
        "Mechanism: Publishing reads the Content Rating the profile already holds. It runs no check of its own and makes no API call, so it is instant. Only 'General' profiles can be published; 'Adult 18+' is refused (though it can still be shared privately), and so is anything Unrated or Pending.\n"
        "Getting rated: `/profile manage` -> Home -> Content Safety -> Submit for Rating.\n"
        "Unpublishing: Done from `/profile hub` -> Manage My Shares. Existing borrows keep working; the profile simply stops being listed.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Publishing was refused because the profile is Unrated.' Fix: Submit it for a rating from the Content Safety dashboard, then publish.\n"
        "- Symptom: 'Publishing used to fail with an avatar or URL error.' Fix: That download is gone. Publishing no longer fetches the avatar.\n"
        "- Symptom: 'The Declare Adult 18+ action disappeared after I published.' Fix: It is hidden for published profiles. Unpublish first."
    ),
    "profiles/child_bot_sync.txt": (
        "Concept: A Child Bot gives a profile its own Discord application, so it appears in the member list with its own presence rather than speaking through a webhook. Registration is restricted to the bot operator because it requires a bot token. Configure at `/settings` -> Child Bots.\n"
        "Sync: The parent instance automatically pushes the profile's display name and avatar to the child application whenever they change.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Child bot appearance changed too frequently.' Fix: Discord strictly limits application avatar updates (2 changes per 10 minutes). The system enforces cooldowns to prevent API bans. Wait 10 minutes before trying again.\n"
        "- Symptom: 'I cannot create a child bot.' Fix: Only the bot operator can register child bot applications. Ask them, or use webhook delivery, which needs no extra application."
    ),
    "profiles/generating_profiles.txt": (
        "Command: `/profile generate`\n"
        "Concept: Drafts a complete profile -- persona fields and AI instructions -- from a short concept sentence, so you start from a structurally complete character rather than an empty form.\n"
        "Status: The result is an ordinary Class A Personal Profile with no special properties. It counts against your 100-profile limit and is edited exactly like any other.\n"
        "Expectation: Treat it as a first draft. The structure will be sound; the voice will be generic until you edit the Persona and Instructions yourself, and add Training Examples.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Generated profiles all sound the same.' Fix: Expected. Generation gives you a scaffold. Distinctive voice comes from Training Examples and hand-edited Instructions."
    ),
    "profiles/content_ratings.txt": (
        "Concept: Every profile carries exactly one Content Rating, and every distribution gate is derived from it -- whether the profile can be shared, published, or used in Global Chat -- plus, for Adult, which channels it may run in.\n"
        "States: 'Unrated' is where every new profile starts; it runs normally but cannot be shared, published or used in Global Chat. 'Pending' means submitted and awaiting a verdict. 'General' can do everything. 'Adult 18+' runs only in age-restricted channels and can be shared privately but never published or used in Global Chat. 'Exempt' is a bot-operator setting that skips classification and disables provider content filtering; it carries the same capabilities as General, so an exempt profile can be shared, published and used in Global Chat.\n"
        "Nothing is classified unless asked: A profile is only ever sent to the classifier when its owner submits it, from `/profile manage` -> Home -> Content Safety -> Submit for Rating. Editing a persona does not trigger a check, and neither does creating a profile.\n"
        "Unrated is not restricted: An Unrated profile behaves exactly like a General one at runtime -- same channels, same provider filtering. The rating governs distribution, not execution. It is only barred from being handed to other people.\n"
        "Declaring 18+ yourself: The Content Safety dashboard has a Declare Adult 18+ action that sets the Adult rating directly, with no classifier call and no waiting. It counts as a rating, so a declared profile can be shared privately immediately. Withdrawing it returns the profile to Unrated.\n"
        "What is judged: The profile name, display name, avatar image, persona and AI instructions. Never long-term memories, training examples or conversation logs. The avatar is judged as an image; if it cannot be downloaded the text is judged alone rather than the profile being refused.\n"
        "Editing a rated profile: The rating no longer describes the profile, so you are asked whether to re-check it or set it back to Unrated. If other people have borrowed the profile it is re-checked automatically, so their copy keeps an accurate rating.\n"
        "Why a profile was rated Adult: The dashboard shows a broad category -- 'Explicit sexual content', 'Sexually-focused persona', 'Graphic violence', 'Minor safety', or a generic 'Adult themes' -- and never a description of the persona itself.\n"
        "Disputing a verdict: Only the bot operator can clear a classifier Adult verdict, from the mod view of the profile. An Adult rating you declared yourself is different -- you can always withdraw that. A clearance is tied to the exact content that was flagged, so it lapses if the persona is edited afterwards.\n"
        "NSFW Gating: A profile rated Adult 18+ is checked against the destination channel before the prompt is sent. For a borrowed profile the rating is taken from the original owner's profile, so a borrower cannot downgrade it.\n"
        "Channel Content Policy: In any channel that is NOT age-restricted, a content policy note is appended to the system instruction asking the model to keep the response suitable for a general audience. This applies to every provider, including OpenRouter and Ollama, and is editable by the bot operator via `/mod`.\n"
        "Provider Filters: On Google models the harm thresholds follow the destination channel, not the profile: an age-restricted channel sends BLOCK_NONE, every other channel sends BLOCK_ONLY_HIGH. An operator-set Exempt profile sends BLOCK_NONE everywhere. Non-Google providers ignore these thresholds entirely, which is why the content policy note exists.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'I cannot share or publish my profile.' Fix: It is Unrated. Open `/profile manage` -> Home -> Content Safety and choose Submit for Rating.\n"
        "- Symptom: 'Global Chat says my profile cannot be used.' Fix: Global Chat needs a rating of General, or an operator Exemption; publishing to the Public Library is not required and is not a substitute. Unrated profiles must be submitted first; Adult profiles cannot be used there at all, because a Global Chat can be opened in any channel and none of them is guaranteed age-restricted.\n"
        "- Symptom: 'My profile used to be published and now it is not.' Fix: The bot operator reset every content rating on this instance to establish a clean baseline, because ratings reached automatically by older builds were never something the owner asked for. Submit the profile again from Content Safety and re-publish it.\n"
        "- Symptom: 'Adult profile is silent or throwing errors.' Fix: Adult profiles are blocked in standard channels. Set the channel to Age-Restricted (NSFW) in Discord's settings.\n"
        "- Symptom: 'Content Rating is stuck on Pending.' Fix: Classification needs an OpenRouter or Google API key -- your own, or the bot operator's. Without one no verdict can be reached. Check `/settings`.\n"
        "- Symptom: 'The Declare Adult 18+ action is missing.' Fix: It is hidden for borrowed profiles, for published profiles, and while a classifier Adult verdict or an operator Exemption stands. Unpublish the profile first, or edit the original if it is borrowed.\n"
        "- Symptom: 'Publishing failed with a URL or avatar error.' Fix: That check is gone. Publishing no longer downloads anything; it reads the rating you already hold."
    ),

    # --- APIS ---
    "apis/key_slots_and_assignment.txt": (
        "Command: `/settings` -> API Keys (DM only)\n"
        "Concept: You hold four key slots -- Google Gemini 1 and 2, OpenRouter 1 and 2. Select a slot, click 'Submit Key' to store a key in it, then use the assignment dropdown to choose where that slot applies.\n"
        "Scopes: 'Personal' applies the key to your DM conversations and your own profiles' background work. A server entry applies it to every session in that server, and only appears for servers where you hold administrator permission.\n"
        "Important: An assignment is not saved until 'Save Assignments' is clicked. Selecting scopes in the dropdown alone changes nothing.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'I added a key but the bot still will not respond in my server.' Fix: Storing a key and assigning it are two separate steps. Select the slot, tick the server in the assignment dropdown, then click Save Assignments.\n"
        "- Symptom: 'The server I want is not in the assignment dropdown.' Fix: Only servers where you have administrator permission are listed. Ask an administrator of that server to assign their own key.\n"
        "- Symptom: 'Where do I run this?' Fix: `/settings` is DM-only. Open a direct message with the bot."
    ),
    "apis/google_gemini.txt": (
        "Requirements: A Google API key from Google AI Studio, submitted via the `/settings` DM command.\n"
        "Capabilities: Powers standard text generation. It is the ONLY provider that natively supports Google Search Grounding and direct URL fetching, and the only one that can generate images or speech.\n"
        "Free versus paid tier: Free-tier keys work for basic chat but rate-limit hard. Long-term memory summarisation, training retrieval, grounding, content classification and image generation all make their own calls, so a free key degrades the whole experience. Image generation is blocked outright on free-tier keys.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The bot isn't responding in the server' or 'Bot is silent'. Fix: Ensure an administrator has assigned a Google API key to this server via `/settings` -> API Keys, and clicked Save Assignments.\n"
        "- Symptom: 'Image generation failed' or 'Paid Key Required'. Fix: You are using a free-tier Google API key. You must configure billing in Google AI Studio to unlock image models."
    ),
    "apis/openrouter.txt": (
        "Requirements: An OpenRouter API Key submitted via the `/settings` DM command.\n"
        "Capabilities: Allows users to access non-Google models like Anthropic's Claude, Meta's Llama, DeepSeek and xAI's Grok. OpenRouter models are also the only ones that honour the advanced sampling parameters (Min P, Top A, and the frequency, presence and repetition penalties).\n"
        "Limitations: OpenRouter models do NOT have native access to Google Search or URL fetching, and cannot generate images or speech. To use Grounding or URL Context with OpenRouter, you MUST go into `/profile manage` -> Tools -> and set Grounding/URL Context to RAG Mode.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'My Claude profile is hallucinating web links' or 'Grounding failed with Claude'. Fix: You must set your Grounding mode to 'RAG'. Native grounding only works with Google.\n"
        "- Symptom: 'Insufficient Credits' or '402'. Fix: Your OpenRouter account has no remaining credit balance.\n"
        "- Symptom: 'Capability Mismatch' or 'No endpoints found'. Fix: No provider on OpenRouter serves that model with the features requested. Pick a different model."
    ),
    "apis/ollama.txt": (
        "Requirements: Ollama installed on your local machine, and the specific model downloaded via your terminal (e.g., `ollama run llama3`).\n"
        "Setup: Go to `/profile manage` -> Params -> Set Models. Click the 'API' button until it says 'Ollama'. Click 'Set Host URL'.\n"
        "Cost: Ollama models run on your own hardware and cost nothing per message. They cannot generate images or speech, and must use RAG mode for Grounding and URL Context.\n"
        "Remote Hosting: If your Discord bot is hosted on a cloud server, it cannot see your home PC's 127.0.0.1 address. You MUST expose your local Ollama port using a secure SSH tunnel (e.g., `ssh -R 80:localhost:11434 nokey@localhost.run`) and paste the resulting HTTPS link into the Host URL setting.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Ollama is offline' or 'Network Error'. Fix: Ensure the Ollama app is actively running on your PC, and that the Host URL in the bot matches your tunnel address. Tunnel URLs usually change every time the tunnel restarts.\n"
        "- Symptom: 'The Set Host URL button is red.' Fix: Red means the bot could not reach that URL. Green means it responded."
    ),
    "apis/rate_limits_and_errors.txt": (
        "Concept: API execution errors occur when an external inference endpoint rejects a generation payload. The bot parses these and shows a short diagnostic.\n"
        "Status Code Meanings:\n"
        "- 401 (Invalid API Key): The API key submitted is invalid, expired, or has been revoked by the provider.\n"
        "- 429 (Rate Limited): The key has hit its Requests-Per-Minute or daily quota. For Google, common on free accounts. For OpenRouter, it usually means an empty credit balance.\n"
        "- 402 (Insufficient Credits): OpenRouter balance exhausted.\n"
        "- 403 (Access Forbidden / Moderated): The model is restricted for your account, or the content tripped the provider's own safety moderation.\n"
        "- 404 (Model Not Found): The selected model has been deprecated or renamed by the provider.\n"
        "- 413 (File Too Large): An attachment exceeded what the provider accepts.\n"
        "- Empty Response: The model returned no text, most often a silent safety block.\n"
        "Failover Protocol: When a request fails, the bot temporarily marks the key as cooling down and immediately redirects the payload to your designated Fallback Model. Setting a cheap, reliable fallback at `/profile manage` -> Params -> Set Models is what keeps a profile alive through an outage.\n"
        "Custom wording: `/profile manage` -> Home -> Custom Error Message sets what users see when generation fails.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The bot keeps replying with my error message.' Fix: Both the primary and fallback models failed. Check the key is valid and has quota, and that the model names are current."
    ),

    # --- SESSIONS ---
    "sessions/session_config.txt": (
        "Command: `/session config` (server administrators)\n"
        "Capabilities: Configures the multi-profile chat session in this channel. The Cast tab adds and removes participants (up to 200, from personal, borrowed and child bot profiles). The Config tab sets execution order, the Master Prompt, TTS and the per-round response limit.\n"
        "Execution Modes: 'Sequential' forces participants to speak in a strict order. 'Random' shuffles the speaker order every round.\n"
        "Master Prompt: A scene prompt shared by every participant in the session -- the setting, the situation, what is happening. Distinct from any individual profile's persona.\n"
        "Response Limit: Caps how many profiles reply in a single round, so a large cast does not answer every message all at once.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'I configured a session but nothing happens.' Fix: Click 'Start / Update Session' after building the cast. Selecting profiles alone does not start it.\n"
        "- Symptom: 'Only some of my cast replies each round.' Fix: That is the Response Limit. Raise it in the Config tab."
    ),
    "sessions/session_swap.txt": (
        "Command: `/session swap [profile] [use_child_bot] [slot]`\n"
        "Capabilities: Dynamically inject, remove, or swap characters inside an active session without interrupting the conversation. Users can assign specific slots to override a specific participant.\n"
        "Delivery Method: The `use_child_bot` parameter forces the profile to reply using a dedicated Discord bot application (Child Bot) instead of a Webhook."
    ),
    "sessions/session_controls.txt": (
        "Capabilities: Users can dynamically control the flow of a chat session using specific message reactions.\n"
        "- Regenerate (🔁): Discards that reply and generates a new one from the same context.\n"
        "- Next Speaker (⏯️): Triggers the next participant in the cast list to respond.\n"
        "- Continue Round (🍿): Triggers a fresh round for every participant.\n"
        "- Mute Turn (🔇): Hides the targeted message from the bot's memory transcript, making it invisible to the AI while leaving it in the channel.\n"
        "- Skip Participant (❌): Suspends a specific profile from responding in the session until unskipped.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'A reply took the scene in a direction I did not want.' Fix: React with 🔇 to mute that turn. The characters carry on as though it never happened, without deleting the message.\n"
        "- Symptom: 'One character keeps dominating the round.' Fix: React with ❌ on one of its messages to skip it for a while."
    ),
    "sessions/response_modes.txt": (
        "Setup: Configured via `/profile manage` -> Tools -> Cycle Response Mode.\n"
        "Modes:\n"
        "- Regular: The bot replies normally without pings.\n"
        "- Mention: The bot prepends the user's Discord ping to the message payload.\n"
        "- Reply: The bot uses the native Discord Reply feature to visually link its response to the user's prompt.\n"
        "- Mention+Reply: A hybrid behavior combining both.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The profile only responds sometimes, or seems to ignore me.' Fix: Check the response mode. Mention and Reply modes wait to be addressed that way."
    ),
    "sessions/proactivity_and_director.txt": (
        "Setup: Enabled via `/session config` -> Proactivity.\n"
        "Mechanism: When Proactivity is active, an asynchronous system loop monitors the channel. Based on your configured Trigger Chance (0-100%) and Cooldown, the bot can autonomously initiate conversation with no user message at all.\n"
        "AI Director: If configured, the system uses a secondary model to read recent conversation and generate a brief environmental change or sudden event (e.g., 'A loud noise is heard outside').\n"
        "Payload: This scene update is injected as an <internal_note> directly to the cast list, forcing the AI characters to dynamically react to the new situation autonomously.\n"
        "Cost: Proactive rounds consume API quota without anyone prompting them, and the Director is an extra call on top. Start with a low chance and a long cooldown.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The bot keeps talking to itself and burning my quota.' Fix: Lower the Trigger Chance or raise the Cooldown under `/session config` -> Proactivity, or toggle Proactivity off entirely."
    ),
    "sessions/reactivity_and_wakewords.txt": (
        "Setup: `/session config` -> Reactivity.\n"
        "Concept: Reactivity governs whether a profile interjects when it is not its turn in the rotation. Each participant carries a Chance percentage, rolled against user messages, and a list of Wakewords.\n"
        "Wakewords: An exact match in a user's message causes that profile to bypass normal turn order and respond immediately. Useful for calling a specific character by name in a large cast.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Profiles keep interrupting each other.' Fix: Lower the Chance percentage for those participants.\n"
        "- Symptom: 'My wakeword does not trigger.' Fix: Wakeword matching is exact. Check spelling and that the word actually appears in the message."
    ),
    "sessions/whisper_and_speak.txt": (
        "Commands: `/whisper` and `/speak`\n"
        "Whisper: Sends a private message to one participant in an active multi-profile session. The reply is ephemeral -- only you see it -- and neither your whisper nor the reply appears in the other profiles' view of the transcript. Use it to direct a character without the rest of the cast reacting to your instruction.\n"
        "Speak: Posts a message as one of your profiles with no generation at all. You supply the text and the profile delivers it under its own name and avatar. Requires administrator permission, and webhook or child bot delivery.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Whisper says there is no active session.' Fix: Whisper targets a participant in a running multi-profile session. Start one with `/session config` first.\n"
        "- Symptom: 'Other characters reacted to something I whispered.' Fix: They should not. Whisper turns are filtered out of every other participant's history; if the content also appeared as a normal message, that is what they saw."
    ),
    "sessions/global_chat.txt": (
        "Command: `/profile global_chat`\n"
        "Concept: Opens a persistent conversation with a single profile. It has its own history, kept separately from any server session, and follows you across servers and into DMs.\n"
        "Surface: Not a DM and not a channel of messages -- the entire conversation is rendered into a single embed, posted where the command was run and edited in place. The profile's latest reply is the embed body and your last message is its footer, so the message never scrolls away. Reply opens a modal to write in, Play submits the queue, and the lock button controls access.\n"
        "Who can join: A session is locked to the host by default. Unlocking it lets anyone who can see the embed press Reply and Play, and the history stays the host's. Replies landing inside the same ten-second window are queued together and delivered as one turn, so a group is answered once rather than once each.\n"
        "Keys: Global chat runs on your Personal key assignment rather than a server's key, so you need a key assigned to 'Personal' in `/settings` -> API Keys.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Global chat says no API key.' Fix: Assign one of your key slots to the 'Personal' scope in `/settings` -> API Keys and click Save Assignments.\n"
        "- Symptom: 'The profile does not remember our server conversation in global chat.' Fix: Correct, and there is no setting for it. Global chat keeps a separate history of its own, and every long-term memory belongs to the server it formed in, so no server memory is ever recalled there."
    ),
    "sessions/maintenance_commands.txt": (
        "Commands: `/refresh`, `/cancel`, `/suspend`, `/purge`, `/clear`, `/session view`, `/session trigger`, `/session audit`\n"
        "- `/refresh`: Clears the short-term conversation buffer for this channel. Long-term memories and training examples are untouched. Use when a profile has become confused about recent events.\n"
        "- `/cancel`: Stops whatever generation or typing indicator is currently running in this channel.\n"
        "- `/suspend`: Administrators. Ends the session in this channel and stops the bot responding until it is configured again.\n"
        "- `/purge`: Administrators. Deletes messages and the associated session memory, or gracefully dehydrates the session.\n"
        "- `/clear`: Clears the bot's own messages from a DM channel.\n"
        "- `/session view`: Shows the current session configuration and participant status.\n"
        "- `/session trigger`: Forces a new round immediately without waiting for a user message.\n"
        "- `/session audit`: Reports token usage and diagnostics for the active session. Use it when you want to know what is actually filling the context window.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The bot seems stuck typing forever.' Fix: Run `/cancel` in that channel.\n"
        "- Symptom: 'Responses are getting expensive or slow and I do not know why.' Fix: Run `/session audit` to see the token breakdown per participant."
    ),

    # --- MEMORY ---
    "memory/stm_buffer.txt": (
        "Concept: Short-Term Memory (STM) is the running conversation buffer. The session keeps one shared transcript, and each participant's view is derived from it at generation time, so private turns stay visible only to the profile they involved.\n"
        "Limits: STM Length defines how many previous conversational turns are appended to the context window (maximum 50). Set it at `/profile manage` -> Params -> Set Generation Parameters & STM.\n"
        "Tradeoff: A longer buffer is more coherent and more expensive, and a very long one can dilute the model's adherence to its persona and instructions.\n"
        "Command: `/refresh` clears the buffer for the channel without deleting anything persistent.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The profile has stopped following its instructions in a long conversation.' Fix: Lower the STM Length. A very long context crowds out system instructions.\n"
        "- Symptom: 'In a big cast, profiles do not see what other characters said.' Fix: Raise STM Length so at least one full round fits in the context window."
    ),
    "memory/ltm_archive.txt": (
        "Concept: Long-Term Memory (LTM) is a persistent, vector-embedded archive of past conversations. Maximum 5,000 memories per profile.\n"
        "Creation: The bot tracks the exchange volume. Once the 'Creation Interval' is met, an auxiliary model extracts the recent conversation and condenses it into a third-person factual summary. Toggle this off with 'Toggle LTM Auto-Creation' if you would rather curate by hand.\n"
        "Retrieval: When a user speaks, their prompt is embedded and compared against the LTM archive. If the score exceeds the 'Relevance Threshold', the memory is injected as `<archive_context>`.\n"
        "Where a memory applies: Every memory belongs to the server it formed in and is recalled only there. A profile used in several servers keeps a separate archive per server, and none of them reach global chat.\n"
        "Management: Add, edit and delete memories at `/profile manage` -> Memory -> Manage Long-Term Memories. Tune recall with 'Set LTM Parameters', and rewrite the summarisation instruction with 'Set LTM Summarization Prompt'.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The profile keeps bringing up irrelevant old events.' Fix: Raise the LTM Relevance Threshold in Set LTM Parameters before deleting memories.\n"
        "- Symptom: 'The profile never remembers anything.' Fix: Check LTM Auto-Creation is on, that the Creation Interval is not set very high, and that the Relevance Threshold is not too strict.\n"
        "- Symptom: 'Memories are not being created.' Fix: Summarisation needs a working API key on the server. Check `/settings` -> API Keys."
    ),
    "memory/training_examples.txt": (
        "Concept: Training Examples are explicit input-output pairs you author by hand to dictate a profile's voice, formatting and tone. Maximum 100 per profile.\n"
        "Retrieval: They are not all injected. Each is embedded, and your current message is matched against the library; only examples clearing the relevance threshold are injected for that turn. A large library therefore stays cheap.\n"
        "Best use: Situational voice -- how the character greets someone, reacts to an insult, or handles a question it cannot answer.\n"
        "Management: `/profile manage` -> Memory -> Manage Training Examples. Tune matching with 'Set Training Parameters'.\n"
        "Availability: Training Examples belong to the profile owner. They cannot be edited on a borrowed profile.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'My training examples never seem to apply.' Fix: Lower the relevance threshold in Set Training Parameters, or rewrite the example input to resemble how users actually phrase things.\n"
        "- Symptom: 'The character copies my examples word for word.' Fix: Provide several varied examples for the same situation rather than one, and raise temperature slightly."
    ),
    "memory/context_metadata_and_xml.txt": (
        "Concept: MimicAI uses an XML partitioning protocol to keep background technical context isolated from conversational chat. Models separate tagged system data from user speech far more reliably than they separate prose from prose.\n"
        "Common tags: `<archive_context>` recalled long-term memories; `<external_context>` web search summaries; `<document_context>` text fetched from URLs; `<time_context>` the profile's local time; `<whisper_context>` and `<private_response>` hidden exchanges; `<internal_note>` system prompts that steer a round; `<scene_prompt>` the session master prompt; `<training_data>` matched style examples; `<content_policy>` the general-audience note.\n"
        "Identity Headers: Every message in history is prefixed `<Name> [ID: PID] [Timestamp]:`. The PID guarantees two characters with the same name are never conflated, and the timestamp gives the model real chronological awareness.\n"
        "Scrubbing: These tags are stripped from model output before delivery, so they never appear in chat.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'XML tags are appearing in the bot's messages.' Fix: The model is imitating the tags it sees rather than the tags leaking. Add an instruction telling it never to output XML, or switch to a model better suited to roleplay."
    ),

    # --- FEATURES ---
    "features/image_generation.txt": (
        "Commands: `!image` or `!imagine` followed by a prompt.\n"
        "Capabilities: Profiles can generate `.png` visuals. The generated image is then shown to the profile's own text model, which writes an in-character comment to send with it -- so the character presents the picture as something it made or found.\n"
        "Self-portraits: The system automatically injects the profile's 'Appearance' text into the image prompt if the request appears to be for a picture of the character itself, keeping it consistent between generations.\n"
        "Editing: Reply to an existing image with `!image` and a new instruction to iterate on that image.\n"
        "Enabling: `/profile manage` -> Tools -> Toggle Image Generation.\n"
        "Output settings: `/profile manage` -> Tools -> Set Image Output chooses aspect ratio, resolution and thinking level. The options offered are only the ones the profile's chosen image model accepts, so the list changes with the model. Leaving a setting on 'Model default' sends no preference at all.\n"
        "Resolution: 512, 1K and 2K. 4K is not offered -- a 4K render usually exceeds Discord's attachment limit and would fail to upload after being paid for.\n"
        "Thinking level: Only the Gemini 3 image models take one. HIGH refines the composition before drawing, which is slower and billed for the extra thinking; MINIMAL draws straight away.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Paid Key Required' or image generation fails. Fix: Image models are completely blocked by Google on free-tier API keys. A valid billing account must be active in Google AI Studio.\n"
        "- Symptom: '!image does nothing.' Fix: Image generation is disabled for that profile, or the profile runs on OpenRouter or Ollama, which cannot generate images. Only Google models can.\n"
        "- Symptom: 'The aspect ratio I picked was ignored.' Fix: The profile's image model does not carry that ratio. The setting is kept on the profile and left out of the request rather than failing it, so switching back to a model that supports it restores the setting."
    ),
    "features/speech_tts.txt": (
        "Concept: Text-To-Speech uses a generative audio model, so it is directed with description rather than tuned with sliders.\n"
        "Configuration: The Director's Desk (`/profile manage` -> Persona -> TTS Instructions) takes plain English for Vocal Archetype, Accent, Dynamics, and Pacing and Style.\n"
        "Voice: `/profile manage` -> Params -> Choose TTS Voice offers all thirty prebuilt Gemini voices, grouped by gender (14 female, 16 male) and described by Google's own one-word character (Kore is firm, Enceladus breathy, Sulafat warm). It is a dropdown rather than a text box because an unrecognised voice name is rejected by the API and reaches you as silence, not an error.\n"
        "Other speech settings: TTS on/off and temperature are at `/profile manage` -> Params -> Set Speech Settings; the speech model is under Set Models. A lower temperature produces stable audio; values far from default cause audible artefacts.\n"
        "Audio tags: Inline tags in a response steer delivery for that stretch of text -- `[whispers]`, `[shouting]`, `[laughs]`, `[sighs]`, `[excitedly]`, `[sarcastic]`. There is no fixed list; the model interprets what it is given.\n"
        "Session audio: `/session config` -> Config -> Toggle TTS chooses text-only, audio plus text, audio only, or multi-audio, which holds each participant's audio until the round ends and stitches it into one file.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Audio sounds garbled or erratic.' Fix: Lower the speech temperature back towards its default.\n"
        "- Symptom: 'No audio is generated.' Fix: TTS requires a Google API key. OpenRouter and Ollama profiles cannot produce speech.\n"
        "- Symptom: 'The channel goes quiet for a long time before the voice arrives.' Fix: Synthesis takes ten to thirty seconds. The placeholder reads 'Synthesising speech...' with a running timer for that whole stretch, and a child bot that had no placeholder gets one created for it once the wait passes ten seconds.\n"
        "- Symptom: 'The voice read my Director's Desk notes out loud.' Fix: This is a documented failure of the speech classifier on vague prompts. Each request already carries a preamble telling the model the notes are direction rather than lines; making the notes more concrete and less like dialogue makes it rarer."
    ),
    "features/neuro_engine.txt": (
        "Concept: The Neuro-Endocrine Engine simulates hormonal states to produce dynamic emotional characterisation that carries between turns.\n"
        "Variables: Four values on a 0-100 scale -- Dopamine (joy, motivation, reward), Cortisol (stress, anxiety, frustration), Oxytocin (bonding, trust, empathy), and Adrenaline (energy, urgency).\n"
        "Execution: The current state is described in the system prompt. At the end of each response the model emits a hidden update tag reflecting how the exchange affected it, and the next turn reads that state back. None of this is visible in chat.\n"
        "Enabling: `/profile manage` -> Tools -> Toggle Neuro-Endocrine Engine.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'The character is stuck in a bad mood.' Fix: Cortisol has climbed and is carrying forward. Reopen the Neuro settings to reset the values, or `/refresh` the channel."
    ),
    "features/grounding_and_rag.txt": (
        "Concept: Web Grounding enables real-time Google Search. URL Context lets the bot read links posted in chat. Both cycle OFF -> NATIVE -> RAG at `/profile manage` -> Tools.\n"
        "Native Mode: Uses Google's own tooling. Accurate, supports inline citations, and for URLs can read large files and PDFs server-side. Works ONLY with official Google Gemini models.\n"
        "RAG Mode: A background model searches or scrapes and summarises before your profile sees the result. Entirely model-agnostic, so it MUST be used if the profile runs on OpenRouter or Ollama. Costs one extra API call per message.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Grounding does nothing on my OpenRouter profile.' Fix: Switch it from NATIVE to RAG. Native tools are Google-only.\n"
        "- Symptom: 'Responses got much slower after I enabled these.' Fix: RAG mode adds a model call per message for each of Grounding and URL Context. Set them to OFF, or to NATIVE on a Google model."
    ),
    "features/typing_simulation.txt": (
        "Concept: Realistic Typing splits a reply on sentence boundaries and streams it with human-like delays instead of posting one block.\n"
        "Configuration: `/profile manage` -> Tools -> Toggle Realistic Typing. Set the characters-per-second rate, a maximum delay cap, and the chunking mode.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Messages take far too long to arrive.' Fix: Raise the characters-per-second value or lower the maximum delay cap. Long replies multiply the effect."
    ),
    "features/repetition_critic.txt": (
        "Concept: The Anti-Repetition Critic is a secondary evaluation layer that detects a profile falling into linguistic loops.\n"
        "Mechanism: Before a message is sent, the Critic reviews recent output for structural repetition -- the same opening phrase every turn, the same sentence rhythm, verbatim loops. If it finds one, it injects an explicit negative constraint banning that pattern for the coming turn.\n"
        "Tradeoff: It reduces repetition noticeably and costs one extra API call per message. Worth enabling on long-running sessions; not worth it on a fast casual profile.\n"
        "Enabling: `/profile manage` -> Tools -> Toggle Anti-Repetition Critic.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Replies became noticeably slower.' Fix: The Critic is an extra model call per message. Disable it if latency matters more than variety."
    ),
    "features/multimodal_media_handling.txt": (
        "Concept: Multimodal processing allows profiles to analyse media files (images, audio, video) attached to your Discord messages.\n"
        "Vision Processing: Models with native vision support can analyse image attachments. Replying to a message containing an image pulls that image into the profile's context.\n"
        "Audio & Video: Capable models can process direct audio and video files.\n"
        "Limitations & Errors: If you attach media to a profile powered by a text-only model -- which many OpenRouter and Ollama models are -- the API call fails and the bot reports 'Unsupported File Format (Model lacks Vision/Audio support)'.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Unsupported File Format.' Fix: The profile's model cannot read that media type. Switch to a vision-capable model, or omit the attachment.\n"
        "- Symptom: 'File Too Large' or 413. Fix: The attachment exceeds what the provider accepts. Send a smaller file."
    ),
    "features/help_mode.txt": (
        "Commands: `/help`, `/help ask:<question>`, `/guide`\n"
        "Concept: `/help ask` answers a technical question about the bot directly, retrieving from the documentation. The answer is ephemeral. `/help` with no argument toggles MimicGuide, a built-in system profile, in and out of the current session so you can ask follow-ups conversationally. `/guide` opens a browsable documentation menu.\n"
        "Help Mode on your own profiles: `/profile manage` -> Tools -> Toggle Help Mode lets one of your own characters answer technical questions in character. Relevant documentation is retrieved and injected for that turn only; if the question is not technical, nothing is injected and the profile talks normally.\n"
        "Operator control: The documentation corpus is editable at `/mod` -> Docs, and re-embeds when saved.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: '/help ask says documentation vectors are not loaded.' Fix: The bot operator has no Google API key configured. Embedding the documentation requires one.\n"
        "- Symptom: 'A Help Mode profile answers technical questions when I wanted roleplay.' Fix: Turn Help Mode off for that profile. It only triggers on questions that match the documentation, but a roleplay scenario about the bot itself can match."
    ),
    "features/data_portability.txt": (
        "Commands: `/export`, `/import`, `/privacy` (DM only)\n"
        "Export: Writes selected profiles and their memories to a plaintext `.mimic` file. The data is decrypted on the way out so it can move between instances -- which also means anyone holding the file can read the personas and memories in it.\n"
        "Import: Validates the schema, resolves name collisions by renaming, re-encrypts the content and files it into your account.\n"
        "Privacy: `/privacy` covers reviewing and deleting your stored data, including full account data deletion.\n"
        "Troubleshooting / Symptoms:\n"
        "- Symptom: 'Export or import is not available.' Fix: Both are DM-only. Run them in a direct message with the bot.\n"
        "- Symptom: 'Imported profiles came in with different names.' Fix: A name already existed in your account. Imports rename around collisions rather than overwriting."
    ),
}


OLLAMA_GUIDE_TEXT = (
    "**Ollama Localhost Setup Guide**\n\n"
    "To allow MimicAI to connect to your local Ollama instance remotely, you need to expose your local port securely. "
    "You can do this easily without port forwarding using a free service like [localhost.run](https://localhost.run/).\n\n"
    "**Steps:**\n"
    "1. Ensure Ollama is running locally on port `11434`.\n"
    "2. Open your terminal or command prompt.\n"
    "3. Run the following command: `ssh -R 80:localhost:11434 nokey@localhost.run`\n"
    "4. The output will provide you with a secure `https://` URL (e.g., `https://your-tunnel.localhost.run`).\n"
    "5. Click **Set Host URL** and paste that `https://` URL.\n\n"
    "*Note: If the **Set Host URL** appears green, the URL is working. Otherwise, it will appear red. Every time you restart the SSH tunnel, the URL may change.*"
)
