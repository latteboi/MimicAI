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

**Web search, in character.** Each character searches when a reply needs something current,
and cites its sources under the message.

<img width="1252" height="385" alt="01-web-search" src="https://github.com/user-attachments/assets/e64ed8e0-d60b-4a06-9e3a-0a213687764d" />

**A cast in one channel.** `/session config` seats a child bot and two webhook profiles —
then they argue.

<img width="1270" height="848" alt="02-cast" src="https://github.com/user-attachments/assets/dea317d0-8c6a-4f71-b1bd-6da23ed5f944" />

**Memory past a reset.** `/refresh` clears the channel's conversation; what the character
remembers long-term survives it.

<img width="1035" height="331" alt="03-memory" src="https://github.com/user-attachments/assets/6f639fb1-8d10-423b-a4fb-07239b14d852" />

**Images and whispers.** `!imagine` draws, and the character comments on its own picture. A
`/whisper` to another participant stays between the two of you.

<img width="1262" height="781" alt="04-images-and-whispers" src="https://github.com/user-attachments/assets/d63a0ddb-2cb3-40ab-af9c-922c7c389712" />

**A voice.** With text-to-speech on for the session, replies arrive with an audio clip.

<img width="1281" height="806" alt="05-voice" src="https://github.com/user-attachments/assets/41bfef9f-07f3-415e-be74-152d9cbd61e4" />

**Getting started.** The greeting on joining a server, and `/start`'s live checklist.

<img width="972" height="609" alt="06-getting-started" src="https://github.com/user-attachments/assets/4fe9b6fb-f4c9-4baa-8db5-6cd74ee4486c" />

**Thinking, shown.** With Thought Summary on, each reply carries the model's reasoning as an
attached file.

<img width="969" height="784" alt="07-thinking-and-reasoning" src="https://github.com/user-attachments/assets/235ae4d6-7ac2-4f91-b537-30ac1e4a5a39" />

**A follow-up, and its trace.** Left hanging, a character with Proactivity on keeps talking.
**View Generation Trace** on any reply shows its model, cost, recalled memories and the
neuro engine's four levels at that moment.

<img width="1272" height="884" alt="08-follow_up-generation_trace-and-neuro_engine" src="https://github.com/user-attachments/assets/ab0a86da-e9c7-40eb-909a-27fb40d39317" />

**A table game.** `/play eights` deals the cast into Mimic Eights; each character plays its
own hand, and the table reacts when someone wins.

<img width="1256" height="563" alt="09-mimic_eights" src="https://github.com/user-attachments/assets/0bf810ea-1571-4f18-a720-12940c9f0504" />

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
participant, `/speak` to post as one of your profiles — either verbatim, or re-voiced
in character with a private preview before it goes out — `/trigger` to force a round,
and `/session audit` for token telemetry and diagnostics. The **View Generation Trace**
context menu opens the same telemetry for a single reply.

`/play eights` seats the cast, and any people who join, at a table of **Mimic Eights**, a
Crazy Eights variant for two to six players. Moves cost no model call: each character's
temperament and neuro state weight what it plays, and it speaks only when something happens
to it and once more at the end.

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

`/export` and `/import` move profiles and memories as encrypted `.mimic` files. A standard
export imports only on the instance that made it; a passphrase export moves to a
self-hosted one. The official instance imports nothing from other instances.
`/privacy` covers data deletion. Server administrators get channel suspension and purge;
the bot owner gets a `/mod` dashboard with a blacklist, an automatic moderator, and a
content classifier that confines profiles declared 18+ to age-restricted channels.

---

## Requirements

- **Python 3.10 or newer.**
- A Discord bot application (free).
- At least one API key: **Google AI Studio (Gemini)** or **OpenRouter**. A Gemini key has
  to be paid-tier for conversations: Google may train on what a free-tier key is sent, so
  the bot keeps server messages and Global Chat off one (see `/privacy`). Paid tier also avoids the
  rate limits that throttle memory summarisation, training retrieval, grounding and image
  generation. Alternatively, point a profile at a local **Ollama** server and pay nothing.
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
