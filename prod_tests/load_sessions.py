"""Load test: many chat sessions at once through the real turn path, on one event loop.

Not part of the test suite (it lives outside tests/): it takes minutes and answers a
question rather than asserting one. The question is what a reply costs the machine. The bot is
a relay -- the model runs at OpenRouter -- so capacity is CPU per reply against the
e2-micro's quarter of a core, and whether the loop thread stays responsive while
replies land together. Neither shows up in a unit test, and neither can be read off
the code.

What is real: MimicCog and every manager, built as BotManager builds them, over a
temporary data directory (MIMIC_DATA_DIR), with the same allocator tuning, gc.freeze
and two-worker executor. Profiles are stored encrypted and read back through
ProfileManager; user messages are real `discord.Message` objects handed to the real
`on_message`; the session worker, prompt building, history windowing, the generation
gate, the OpenRouter adapter's payload and parse, delivery and session flushes all run
as they do in production.

What is faked, and so not counted: the network. OpenRouter and Google answer from an
in-process transport after a delay, with canned replies. Discord's REST calls go to a
fake webhook and channel; its gateway is skipped (messages are built from dicts, not
from websocket JSON). So the CPU figure is the bot's own work per reply, not discord.py's
HTTP or websocket cost, and it is measured on whatever machine runs this. Run it on the
e2-micro itself for the number that matters there -- with the bot stopped, or at a
quiet hour, since both would share the same quarter of a core. Nothing touches the live
data directory or the live bot's instance lock.

    python3 prod_tests/load_sessions.py                     # 100 sessions, 2 characters each
    python3 prod_tests/load_sessions.py --sessions 20 --measure 60
    python3 prod_tests/load_sessions.py --profile           # adds where the CPU went

Each simulated person posts, waits for the whole cast to answer, thinks for `--think`
seconds (0.5-1.5x, at random) and posts again, so the reply rate follows from the
session count, the cast size, the model latency and the think time, as it would.
"""
import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sessions", type=int, default=100, help="channels with a live session (100)")
    p.add_argument("--cast", type=int, default=2, help="characters answering each post (2)")
    p.add_argument("--latency", type=float, default=8.0,
                   help="seconds a model takes to answer, 0.5-1.5x at random (8)")
    p.add_argument("--think", type=float, default=30.0,
                   help="seconds a person waits after the replies before posting again (30)")
    p.add_argument("--history", type=int, default=80, help="turns already in each session's log (80)")
    p.add_argument("--warmup", type=float, default=45.0, help="seconds before measuring starts (45)")
    p.add_argument("--measure", type=float, default=120.0, help="seconds measured (120)")
    p.add_argument("--slots", type=int, default=0, help="override MIMIC_GENERATION_SLOTS")
    p.add_argument("--profile", action="store_true",
                   help="cProfile the measured window (inflates the CPU figures)")
    p.add_argument("--keep", action="store_true", help="keep the temporary data directory")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


ARGS = parse_args()

# Everything below reads these at import, so they are set before the first cogs import.
DATA = tempfile.mkdtemp(prefix="mimic-load-")
os.environ["MIMIC_DATA_DIR"] = DATA
os.environ.pop("GCP_PROJECT_ID", None)  # never reach for real secrets
from cryptography.fernet import Fernet  # noqa: E402

os.environ["ENCRYPTION_KEY"] = Fernet.generate_key().decode()
os.environ["DISCORD_OWNER_ID"] = "100000000000000001"
os.environ["DISCORD_SDK"] = "load-test"
if ARGS.slots:
    os.environ["MIMIC_GENERATION_SLOTS"] = str(ARGS.slots)

# As BotManager does, before anything allocates.
from cogs.utils.memory_tuning import tune_allocator  # noqa: E402

tune_allocator()

import asyncio  # noqa: E402
import cProfile  # noqa: E402
import collections  # noqa: E402
import datetime  # noqa: E402
import gc  # noqa: E402
import io  # noqa: E402
import platform  # noqa: E402
import pstats  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import discord  # noqa: E402
import httpx  # noqa: E402
import orjson  # noqa: E402
from discord.ext import commands  # noqa: E402

from cogs.utils import event_loop, http_client  # noqa: E402
from cogs.utils.constants import PLACEHOLDER_EMOJI, defaultConfig  # noqa: E402
from cogs.utils.helpers import _format_history_entry, _get_user_hash  # noqa: E402
from cogs.utils.loop_probe import rss_bytes  # noqa: E402
from cogs.services.api import google_rest  # noqa: E402
from cogs.services.generation.gate import generation_gate  # noqa: E402
from cogs.managers.session_manager import intern_turn  # noqa: E402

BOT_ID = 200000000000000001
GUILD_ID = 300000000000000001
CHANNEL_BASE = 400000000000000000
USER_BASE = 500000000000000000
OWNER_BASE = 600000000000000000
MODEL = "OPENROUTER/google/gemini-2.5-flash"
rng = random.Random(ARGS.seed)

REPLIES = [
    "The rain had not let up since noon, and she watched it streak the window while the "
    "kettle worked itself up to a whistle. \"You're asking the wrong question,\" she said at "
    "last. \"It isn't whether he lied. Everyone lies about that night. It's what he was "
    "trying to keep you from looking at.\" She set two cups down, pushed one across the "
    "table, and waited for him to notice the photograph tucked beneath the saucer.",
    "He laughed, short and surprised, and leaned back until the chair creaked. \"All right. "
    "Fair. I walked into that one.\" For a moment he turned the ring on his finger, the way "
    "he did when he was deciding how much of the truth to spend. \"The ledger was never in "
    "the study. My father kept it in the boathouse, under the loose board by the third "
    "mooring. If it's gone, then someone knew where to look, and that's a short list.\"",
    "Somewhere below them a door banged in the wind. Neither of them moved. \"We should go "
    "now,\" she said, \"before the tide turns and the path floods. If we wait for morning "
    "we'll be asking the harbourmaster for a boat, and he talks to everyone.\" She was "
    "already reaching for her coat, one eye on the lamp, the other on the dark beyond the "
    "glass where the lane bent down towards the water.",
]
REPLY_MARKERS = tuple(r[:40] for r in REPLIES)
USER_LINES = [
    "Wait, back up. What did the harbourmaster actually say to you?",
    "I pick up the photograph and turn it over. Is there anything written on the back?",
    "Okay but if the ledger's gone, who else even knew about the boathouse?",
    "Let's go. I grab the lamp and follow her out.",
    "Hang on, I want to ask him one more thing before we leave.",
]


# --- the fake network ---------------------------------------------------------------

class Wire:
    """In-process OpenRouter and Google, answering after a delay. Kept cheap on purpose:
    anything it spends is counted as the bot's CPU."""

    def __init__(self, latency):
        self.latency = latency
        self.routes = collections.Counter()
        self.chat_bodies = [orjson.dumps({
            "id": "gen-load", "model": "google/gemini-2.5-flash", "provider": "Google",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 2400, "completion_tokens": 180, "total_tokens": 2580,
                      "cost": 0.0006},
        }) for text in REPLIES]
        self.google_bodies = [orjson.dumps({
            "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2400, "candidatesTokenCount": 180,
                              "totalTokenCount": 2580},
        }) for text in REPLIES]
        self._vectors = {}

    def vector(self, dims):
        if dims not in self._vectors:
            self._vectors[dims] = [round(rng.uniform(-0.1, 0.1), 5) for _ in range(dims)]
        return self._vectors[dims]

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        await request.aread()
        if path.endswith("/chat/completions"):
            self.routes["openrouter chat"] += 1
            await asyncio.sleep(self.latency * rng.uniform(0.5, 1.5))
            return httpx.Response(200, content=rng.choice(self.chat_bodies),
                                  headers={"content-type": "application/json"})
        if path.endswith("/embeddings"):
            self.routes["openrouter embeddings"] += 1
            dims = orjson.loads(request.content).get("dimensions") or 256
            await asyncio.sleep(0.2)
            return httpx.Response(200, content=orjson.dumps(
                {"data": [{"embedding": self.vector(dims)}], "usage": {"prompt_tokens": 20}}))
        if ":generateContent" in path or ":streamGenerateContent" in path:
            self.routes["google generate"] += 1
            await asyncio.sleep(self.latency * rng.uniform(0.5, 1.5))
            body = rng.choice(self.google_bodies)
            if ":streamGenerateContent" in path:
                return httpx.Response(200, content=b"data: " + body + b"\r\n\r\n",
                                      headers={"content-type": "text/event-stream"})
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})
        if ":embedContent" in path or ":batchEmbedContents" in path:
            self.routes["google embeddings"] += 1
            await asyncio.sleep(0.2)
            dims = 256
            try:
                dims = orjson.loads(request.content).get("outputDimensionality") or 256
            except Exception:
                pass
            if ":batch" in path:
                count = len(orjson.loads(request.content).get("requests") or [1])
                return httpx.Response(200, content=orjson.dumps(
                    {"embeddings": [{"values": self.vector(dims)}] * count}))
            return httpx.Response(200, content=orjson.dumps({"embedding": {"values": self.vector(dims)}}))
        self.routes[f"unanswered {request.method} {request.url.host}{path}"] += 1
        return httpx.Response(404, content=b'{"error":"load test"}')


# --- the fake Discord ---------------------------------------------------------------

_ids = iter(range(700000000000000000, 800000000000000000))


class Sent:
    """What a webhook or channel send hands back: enough of a Message for delivery."""

    def __init__(self, channel, content):
        self.id = next(_ids)
        self.channel = channel
        self.content = content
        self.embeds = []
        self.attachments = []
        self.author = SimpleNamespace(id=BOT_ID, bot=True, display_name="mimic", mention=f"<@{BOT_ID}>")
        self.jump_url = f"https://discord.com/channels/{GUILD_ID}/{channel.id}/{self.id}"

    async def edit(self, **_):
        return self

    async def delete(self, **_):
        return None

    async def add_reaction(self, *_):
        return None

    async def clear_reaction(self, *_):
        return None


class Recorder:
    def __init__(self):
        self.replies = 0
        self.other_sends = 0
        self.edits = 0
        self.deletes = 0
        self.measuring = False
        self.measured_replies = 0


REC = Recorder()


def is_placeholder(content):
    return not content or content.startswith(PLACEHOLDER_EMOJI) or content.strip() == PLACEHOLDER_EMOJI


def record_send(channel, content):
    if content and any(m in content for m in REPLY_MARKERS):
        REC.replies += 1
        if REC.measuring:
            REC.measured_replies += 1
        channel.replies_this_round += 1
        if channel.replies_this_round >= ARGS.cast:
            channel.round_done.set()
    elif not is_placeholder(content or ""):
        REC.other_sends += 1
        if REC.other_sends <= 5:
            print(f"  [load] non-reply message in #{channel.name}: {content[:160]!r}", flush=True)


class FakeWebhook:
    def __init__(self, channel):
        self.channel = channel
        self.id = next(_ids)
        self.url = f"https://discord.com/api/webhooks/{self.id}/load"

    async def send(self, content=None, **_):
        record_send(self.channel, content)
        return Sent(self.channel, content)

    async def edit_message(self, message_id, **_):
        REC.edits += 1
        return SimpleNamespace(id=message_id)

    async def delete_message(self, message_id, **_):
        REC.deletes += 1

    async def fetch(self, **_):
        return self


class _Typing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class LoadChannel(discord.TextChannel):
    """A real TextChannel, so every isinstance test passes, with its REST calls local."""

    async def send(self, content=None, **_):
        record_send(self, content)
        return Sent(self, content)

    async def fetch_message(self, message_id, /):
        raise discord.NotFound(SimpleNamespace(status=404, reason="load test"), "Unknown Message")

    def typing(self):
        return _Typing()

    async def webhooks(self):
        return [self.webhook]

    async def create_webhook(self, **_):
        return self.webhook


# --- building the world ---------------------------------------------------------------

PERSONA = {
    "backstory": ["Raised above a chandler's shop in a harbour town that smells of tar and "
                  "rain, the second of four children and the only one who stayed.",
                  "Worked six years on the ferries before an accident she does not discuss.",
                  "Keeps her late father's ledgers and reads them the way other people read "
                  "letters, looking for what was left out."] * 2,
    "personality_traits": ["Dry, patient, slow to anger and slower to forgive.",
                           "Answers a question with a better question when she can.",
                           "Notices hands, shoes and what people do not look at."] * 2,
    "likes": ["Strong tea, weather, maps with mistakes in them, people who admit they were wrong."],
    "dislikes": ["Being hurried, the harbourmaster, anyone who says 'trust me'."],
    "appearance": ["Tall, weathered, grey at the temples early; an oilskin coat older than she is."],
}
INSTRUCTIONS = [
    "Write in third person, past tense, with dialogue in double quotes. Two or three short "
    "paragraphs. Stay in the scene; never summarise what other characters feel.",
    "Keep the mystery moving: every reply should reveal one small concrete detail or raise "
    "one question. Do not resolve the central mystery.",
    "", "",
]


def make_profiles(cog, count):
    """`count` profiles across owners of five each, stored as the bot stores them."""
    enc = cog.storage_manager._encrypt_data
    persona = {k: [enc(line) for line in v] for k, v in PERSONA.items()}
    instructions = [enc(x) for x in INSTRUCTIONS]
    seats = []
    for n in range(count):
        owner = OWNER_BASE + n // 5
        name = f"character{n}"
        profile = cog.profile_manager._get_or_create_user_profile(owner, name)
        config = profile["config"]
        config.update(primary_model=MODEL, fallback_model="NONE",
                      custom_display_name=f"Character {n}")
        cog.profile_manager._save_profile_config(owner, name, config)
        prompts = dict(profile.get("prompts") or {})
        prompts.update(persona=persona, ai_instructions=instructions)
        cog.profile_manager._save_profile_prompts(owner, name, prompts)
        pid = cog.profile_manager._get_pid_from_name_any(owner, name)
        seats.append({"owner_id": owner, "profile_name": name, "pid": pid})
    return seats


def seeded_log(cast, turns):
    """A conversation already under way, in the shape the worker writes it."""
    log = []
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=2)
    user_id = USER_BASE
    for t in range(turns):
        when = start + datetime.timedelta(seconds=40 * t)
        if t % (len(cast) + 1) == 0:
            text = rng.choice(USER_LINES)
            log.append({"turn_id": f"seed-{t}", "is_user": True, "speaker_pid": str(user_id),
                        "message_ids": [next(_ids)], "timestamp": when.isoformat(),
                        "content": _format_history_entry("Reader", when, text, "UTC",
                                                         entity_id=_get_user_hash(user_id))})
        else:
            seat = cast[t % (len(cast) + 1) - 1]
            text = rng.choice(REPLIES)
            log.append({"turn_id": f"seed-{t}", "is_user": False, "speaker_pid": seat["pid"],
                        "owner_id": seat["owner_id"], "profile_name": seat["profile_name"],
                        "message_ids": [next(_ids)], "timestamp": when.isoformat(),
                        "content": _format_history_entry(seat["profile_name"], when, text, "UTC",
                                                         entity_id=seat["pid"]),
                        "meta": {}})
    return [intern_turn(t) for t in log]


def build_session(cast, turns):
    return {
        "type": "multi",
        "profiles": [{"owner_id": s["owner_id"], "profile_name": s["profile_name"],
                      "method": "webhook", "chance": 100} for s in cast],
        "unified_log": seeded_log(cast, turns), "is_hydrated": True,
        "owner_id": cast[0]["owner_id"], "is_running": False,
        "task_queue": asyncio.Queue(), "worker_task": None,
        "session_prompt": "A harbour town in autumn. The ledger is missing.",
        "session_mode": "sequential",
        "proactivity": {"enabled": False, "chance": 10, "cooldown": 300,
                        "director_model": "off", "director_instructions": ""},
        "compaction": {"enabled": True},
        "started": True,
    }


def user_message(state, channel, user_id, text):
    return discord.Message(state=state, channel=channel, data={
        "id": next(_ids), "channel_id": channel.id, "guild_id": GUILD_ID, "type": 0,
        "content": text, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "edited_timestamp": None, "tts": False, "mention_everyone": False, "mentions": [],
        "mention_roles": [], "attachments": [], "embeds": [], "pinned": False, "flags": 0,
        "author": {"id": str(user_id), "username": f"reader{user_id % 1000}",
                   "discriminator": "0", "avatar": None, "global_name": "Reader"},
    })


# --- measuring ------------------------------------------------------------------------

async def lag_probe(samples, stop):
    interval = 0.05
    while not stop.is_set():
        t0 = time.perf_counter()
        await asyncio.sleep(interval)
        if REC.measuring:
            samples.append(max(0.0, time.perf_counter() - t0 - interval))


async def gate_probe(peaks, stop):
    gate = generation_gate()
    while not stop.is_set():
        if REC.measuring:
            peaks["active"] = max(peaks["active"], gate.active)
            peaks["waiting"] = max(peaks["waiting"], len(gate._queue))
        await asyncio.sleep(0.5)


def pct(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


async def person(cog, state, channel, user_id, round_times, timeouts, stop):
    await asyncio.sleep(rng.uniform(0, ARGS.think))  # arrive spread out, not all at once
    while not stop.is_set():
        channel.replies_this_round = 0
        channel.round_done.clear()
        started = time.monotonic()
        await cog.on_message(user_message(state, channel, user_id, rng.choice(USER_LINES)))
        try:
            await asyncio.wait_for(channel.round_done.wait(), timeout=300)
            if REC.measuring:
                round_times.append(time.monotonic() - started)
        except asyncio.TimeoutError:
            if REC.measuring:
                timeouts.append(channel.id)
        await asyncio.sleep(ARGS.think * rng.uniform(0.5, 1.5))


async def main():
    from cogs.MimicCog import MimicCog

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=2, thread_name_prefix="mimic-io"))

    wire = Wire(ARGS.latency)
    transport = httpx.MockTransport(wire.handle)
    http_client._shared_client = httpx.AsyncClient(transport=transport)
    http_client._openrouter_client = httpx.AsyncClient(transport=transport)
    google_rest._google_rest_client = httpx.AsyncClient(transport=transport)

    bot = commands.Bot(command_prefix="!", intents=discord.Intents.none(),
                       help_command=None, max_messages=None)
    bot.manager_queue = asyncio.Queue()
    await bot._async_setup_hook()
    state = bot._connection
    state.user = discord.ClientUser(state=state, data={
        "id": str(BOT_ID), "username": "mimic", "discriminator": "0", "avatar": None, "bot": True})
    discord_calls = collections.Counter()

    async def no_discord(route, **_):
        discord_calls[f"{route.method} {route.path}"] += 1
        raise discord.HTTPException(SimpleNamespace(status=503, reason="load test"), "no Discord here")

    bot.http.request = no_discord

    guild = discord.Guild(data={
        "id": str(GUILD_ID), "name": "load", "owner_id": str(OWNER_BASE), "features": [],
        "roles": [{"id": str(GUILD_ID), "name": "@everyone", "permissions": "0", "position": 0,
                   "color": 0, "hoist": False, "managed": False, "mentionable": False}],
        "emojis": [], "stickers": [], "member_count": ARGS.sessions + 1,
    }, state=state)
    state._add_guild(guild)

    print(f"Building {ARGS.sessions} sessions in {DATA} ...", flush=True)
    cog = MimicCog(bot)
    if not cog.has_lock:
        raise SystemExit("load test could not take its own instance lock")
    cog.storage_manager._get_api_key_for_guild = (
        lambda _guild_id, provider="gemini": "sk-or-load" if provider == "openrouter" else None)
    cog.all_bot_ids = {BOT_ID}

    seats = make_profiles(cog, max(ARGS.cast, min(100, ARGS.sessions * ARGS.cast)))
    channels = []
    for i in range(ARGS.sessions):
        channel = LoadChannel(state=state, guild=guild, data={
            "id": str(CHANNEL_BASE + i), "name": f"scene-{i}", "type": 0, "position": i,
            "nsfw": False, "parent_id": None, "permission_overwrites": [],
            "rate_limit_per_user": 0, "topic": None, "last_message_id": None})
        channel.webhook = FakeWebhook(channel)
        channel.round_done = asyncio.Event()
        channel.replies_this_round = 0
        guild._add_channel(channel)
        cog.server_manager._webhook_from_cache[channel.id] = channel.webhook
        cast = rng.sample(seats, ARGS.cast)
        cog.multi_profile_channels[channel.id] = build_session(cast, ARGS.history)
        channels.append(channel)

    # As BotManager does once the cog is loaded: what exists now lives for the process.
    gc.collect()
    gc.freeze()

    stop = asyncio.Event()
    lag, round_times, timeouts = [], [], []
    gate_peaks = {"active": 0, "waiting": 0}
    probes = [asyncio.create_task(lag_probe(lag, stop)), asyncio.create_task(gate_probe(gate_peaks, stop))]
    people = [asyncio.create_task(person(cog, state, ch, USER_BASE + i, round_times, timeouts, stop))
              for i, ch in enumerate(channels)]

    print(f"Warming up for {ARGS.warmup:.0f}s, then measuring for {ARGS.measure:.0f}s ...", flush=True)
    await asyncio.sleep(ARGS.warmup)
    profiler = cProfile.Profile() if ARGS.profile else None
    cpu0, wall0 = time.process_time(), time.monotonic()
    REC.measuring = True
    if profiler:
        profiler.enable()
    await asyncio.sleep(ARGS.measure)
    if profiler:
        profiler.disable()
    REC.measuring = False
    cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
    rss_now = rss_bytes()

    stop.set()
    for task in people + probes:
        task.cancel()
    await asyncio.gather(*people, *probes, return_exceptions=True)
    for session in cog.multi_profile_channels.values():
        worker = session.get("worker_task")
        if worker and not worker.done():
            worker.cancel()
    await asyncio.sleep(0.5)
    for task_loop in (cog.session_manager.flush_dirty_sessions_task,
                      cog.session_manager.evict_inactive_sessions_task, cog.refresh_lock_task):
        task_loop.cancel()

    replies = REC.measured_replies
    rate = replies / wall if wall else 0.0
    per_reply = cpu / replies * 1000 if replies else float("nan")
    core_share = cpu / wall if wall else 0.0
    print()
    print(f"{ARGS.sessions} sessions x {ARGS.cast} characters, model latency {ARGS.latency:.0f}s "
          f"(0.5-1.5x), think {ARGS.think:.0f}s, {ARGS.history} turns of history, "
          f"{generation_gate().slots} generation slots")
    print(f"Python {platform.python_version()}, {event_loop.loop_name()}, "
          f"{platform.machine()} {platform.system()}; measured {wall:.0f}s after {ARGS.warmup:.0f}s warm-up"
          + (" UNDER cProfile, so CPU is inflated" if profiler else ""))
    print(f"  replies delivered    {replies}  ({rate:.2f}/s)   other messages {REC.other_sends}   "
          f"rounds not finished in 300s {len(timeouts)}")
    print(f"  CPU                  {cpu:.1f}s = {core_share * 100:.0f}% of one core "
          f"-> {per_reply:.1f} ms per reply")
    print(f"  loop lag (50 ms tick) p50 {pct(lag, 50) * 1000:.0f} ms, p99 {pct(lag, 99) * 1000:.0f} ms, "
          f"max {max(lag or [0]) * 1000:.0f} ms; stalls >=100 ms: {sum(1 for x in lag if x >= 0.1)}")
    mem = f"RSS {rss_now / 1048576:.0f} MB now, " if rss_now else ""
    print(f"  memory               {mem}peak {peak_rss_bytes() / 1048576:.0f} MB")
    print(f"  generation gate      peak {gate_peaks['active']} in flight, {gate_peaks['waiting']} waiting")
    if round_times:
        print(f"  post -> last reply   p50 {pct(round_times, 50):.1f}s, p95 {pct(round_times, 95):.1f}s")
    needed = rate * per_reply / 1000 if replies else 0.0
    print(f"  at this reply rate the bot needs {needed:.2f} of a core; the e2-micro sustains 0.25")
    if replies:
        # 0.2 rather than 0.25: the gateway, discord.py's HTTP and child bots take a share
        # this run does not count. Only meaningful when run on the machine it describes.
        sustainable = 0.2 / (per_reply / 1000)
        print(f"  on an e2-micro, {per_reply:.0f} ms a reply carries about {sustainable:.1f} replies/s; at "
              f"{ARGS.latency:.0f}s a model call that is MIMIC_GENERATION_SLOTS={max(1, int(sustainable * ARGS.latency))} "
              f"(true only if this ran on the e2-micro)")
    print(f"  whole run: {REC.replies} replies, {dict(wire.routes)}; "
          f"webhook edits {REC.edits}, deletes {REC.deletes}")
    if discord_calls:
        print(f"  Discord REST calls that reached the stub (each failed): {dict(discord_calls)}")

    if profiler:
        out = io.StringIO()
        stats = pstats.Stats(profiler, stream=out)
        stats.sort_stats("tottime").print_stats(30)
        out.write("\n--- cumulative, the bot's own code ---\n")
        stats.sort_stats("cumulative").print_stats(r"/cogs/", 40)
        print(out.getvalue())

    await http_client.close_shared_client()
    await http_client.close_openrouter_client()


if __name__ == "__main__":
    try:
        event_loop.run(main())
    finally:
        if ARGS.keep:
            print(f"Data kept in {DATA}")
        else:
            shutil.rmtree(DATA, ignore_errors=True)
