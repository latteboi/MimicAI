# MimicAI

A self-hosted Discord bot that runs AI characters — as webhooks inside your server, or as
their own bot applications in the member list — with persistent personas, long-term memory,
and multi-character roleplay sessions.

**Beta.** Prosperity Public License 3.0.0 (free for non-commercial use).
[mimic-ai.org](https://mimic-ai.org/)

---

## What it does

You write a character: backstory, personality, speech patterns, appearance. MimicAI gives it
a name and an avatar in Discord, a memory that survives restarts, and a model of your
choosing behind it. Then you put several of them in a channel and let them talk — to your
users, and to each other.

Everything runs on your own hardware, under your own API keys. Profile data is encrypted at
rest with a key only your instance holds.

- **Three providers.** Google Gemini and OpenRouter over their HTTP APIs, plus **Ollama**
  for models running on your own machine.
- **Two ways to appear.** Automatic webhook management gives every profile a distinct name
  and avatar with no extra bot applications. Or provision a *child bot* — a real Discord
  application with its own presence, status and member-list entry — managed from the same
  process.
- **Multi-profile sessions.** Put up to 200 profiles in a channel's cast and let them take
  turns, sequentially or randomly, up to a per-round response limit.
- **Three memory layers.** Short-term conversation history, long-term memories written by
  the model as it goes, and training examples you author by hand to pin down voice.

---

## Gallery

<img width="609" height="422" alt="Screenshot 2026-04-06 at 9 36 53 am" src="https://github.com/user-attachments/assets/412d00b3-3760-41c7-992b-c903a39d1fcf" />
<img width="604" height="377" alt="Screenshot 2026-04-06 at 9 38 34 am" src="https://github.com/user-attachments/assets/ec0a7b6e-291c-4dc5-9f6a-607c31b962d7" />
<img width="586" height="872" alt="Screenshot 2026-04-06 at 9 35 25 am" src="https://github.com/user-attachments/assets/496362be-e1f7-4f58-9dd1-ccab4fa8833c" />

---

## Features

The feature surface is large and moves quickly, so the authoritative map lives inside the
bot: run **`/guide`** for a full dashboard-by-dashboard tree, or **`/help`** to ask a
question and get an answer retrieved from the bundled documentation. What follows is the
shape of it.

### Characters

Personas are split into backstory, traits, likes, dislikes and appearance, with four
separate instruction segments for behaviour and formatting. `/profile generate` will draft
a whole profile from a one-line concept if you would rather start from something than from
nothing. Profiles can be shared directly by time-limited share code, or published to the
**Public Library** for anyone to borrow.

### Sessions

`/session config` builds a channel's cast and its rules: turn order, a scene prompt shared
by every participant, a response limit per round, and TTS stitching for the whole round.
Reactivity settings let profiles interject on a chance roll or on wakewords. Proactivity
hands a timer to an **AI Director** model that decides when the cast should start talking
on its own.

Alongside the main session loop: `/whisper` for a private, ephemeral exchange with one
participant, `/speak` to post anonymously as one of your profiles, `/session trigger` to
force a round, and `/session audit` for token telemetry and diagnostics.

### Memory

Long-term memories are summarised by a model at an interval you set, embedded, and
retrieved semantically — with scopes for global, server-only or user-only recall. Training
examples work the same way, matching your current message against a library of
`input → response` pairs to inject the most relevant examples for the moment. Both are
editable by hand from the profile dashboard.

### Tools and multimodality

Image generation (`!image` / `!imagine`) renders through a Gemini image model, then shows
the result to the character's own model so it can comment in voice; appearance text is
injected automatically when the request is for a picture of the profile itself. Grounding
and URL context each cycle through **off → native → RAG**, so they work on models with no
native tool support. Text-to-speech has a director's desk for archetype, accent, dynamics
and pacing.

### Tuning

Temperature, Top P and Top K, plus Min P, Top A, and frequency, presence and repetition
penalties where the provider supports them. Reasoning models get a thinking budget,
reasoning level, and optional thought summaries. Beyond the sampler: realistic typing
delays, response-mode gating (mention and/or reply only), per-profile timezone awareness,
an anti-repetition critic that detects loops and writes negative constraints, and a
neuro-endocrine engine that carries four emotional variables between turns.

### Ownership and safety

`/export` and `/import` move profiles and memories between instances as plaintext.
`/privacy` covers data deletion. Server administrators get channel suspension and purge;
the bot owner gets a `/mod` dashboard with a blacklist, an automatic moderator, and a
content classifier that confines profiles declared 18+ to age-restricted channels.

---

## Requirements

- **Python 3.10 or newer.**
- A Discord bot application (free).
- At least one API key: **Google AI Studio (Gemini)** or **OpenRouter**. Paid-tier Gemini
  keys are strongly recommended — memory summarisation, training retrieval, grounding and
  image generation all make their own calls, and free-tier rate limits will throttle them.
  Alternatively, point a profile at a local **Ollama** server and pay nothing.
- Linux, macOS or Windows. Production runs on a **GCP e2-micro** (1 GB RAM, 0.25 vCPU)
  24/7; see [ARCHITECTURE.md](ARCHITECTURE.md) for what that constraint did to the design.

---

## Quick start

### 1. Create your Discord bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** and name it.
3. Under **Bot**:
   - **Reset Token** to get your bot token.
   - Under *Privileged Gateway Intents*, enable **Presence**, **Server Members** and
     **Message Content**. All three are required.
4. Under **OAuth2 → URL Generator**:
   - Scopes: `bot`, `applications.commands`.
   - Permissions: `Administrator`, or individually — View Channels, Send Messages, Send
     Messages in Threads, Manage Messages, Manage Webhooks, Embed Links, Attach Files,
     Read Message History, Use External Emojis, Bypass Slowmode.
   - Use the generated link to invite the bot.

### 2. Install

```bash
git clone https://github.com/latteboi/MimicAI.git
cd MimicAI
python3 setup.py      # Windows: python setup.py
```

The setup utility creates a virtual environment, installs dependencies from
`requirements.txt`, builds the data directories, and prompts for your bot token and Discord
user ID. On Linux it will offer to install a systemd service for 24/7 uptime.

### 3. Run

```bash
# Linux / macOS
source .venv/bin/activate && python3 BotManager.py

# Windows
.venv\Scripts\python.exe BotManager.py
```

### 4. Add an API key

Open a **direct message** with your bot and run `/settings`, then the **API Keys** tab.

There are four slots — two for Google Gemini, two for OpenRouter. Pick an empty one,
**Submit Key**, then use the assignment dropdown to point it at your **Personal** account
(for DM chats) and at any server you administrate (to power that server's sessions). Save
the assignments.

To use Ollama instead, set the host URL per profile under
`/profile manage → Params → Set Models`.

### 5. Create a character

```
/profile create profile_name:detective
/profile manage profile_name:detective
```

From the dashboard: **Edit Persona** for backstory and traits, **Edit Instructions** for
speech and formatting, **Edit Appearance** for a display name and avatar URL. (Tip: upload
an image to any Discord channel and copy its URL to use as an avatar.)

Then start a session:

```
/session config
```

Add "detective" to the cast, click **Start / Update Session**, and send a message in the
channel.

---

## For developers

[**ARCHITECTURE.md**](ARCHITECTURE.md) covers the layout of the codebase, the storage
format, the turn-rotation engine, how child bots run without subprocesses, and the
performance constraints that shaped all of it. Start there before changing anything on a
per-message path.

---

## Built by AI

The entire codebase was generated by AI, directed by a human with a product vision rather
than a patch to apply. It began in Google AI Studio with Gemini 2.5 Pro; the architecture
notes, the optimisation work and the ongoing maintenance have continued in the same mode.

This is worth stating plainly rather than burying: it explains the codebase's
characteristic shapes — the very long comments that argue for a decision, the god-cog that
grew before it was split, the mixin layering. It also means bugs and inconsistencies exist
that a hand-written codebase of this size would have shaken out differently. They are being
worked through.

---

## Beta status

- Some features add latency by design — the anti-repetition critic and RAG-mode grounding
  each cost an extra model call per turn.
- Interfaces still move between releases. The in-bot `/guide` is regenerated with the code
  and is always more current than any external documentation.
- Bug reports are welcome.

### Roadmap

- **Parallel responses** — an alternative to sequential turn rotation, letting multiple
  profiles speak at once for genuinely chaotic group conversation.

---

## Licence

Released under the **Prosperity Public License 3.0.0**.

- **Non-commercial use** — free for individuals and non-profits.
- **Commercial use** — requires a separate license agreement.

See the `LICENSE` file for the full terms.

---

Developed by **latteboi** · [mimic-ai.org](https://mimic-ai.org/)
