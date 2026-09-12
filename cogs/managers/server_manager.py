import datetime
import math
import os
import time
import traceback
from typing import Dict, Any, List, Optional, Set, Union
import discord

from .storage_manager import IOManager
from ..utils.constants import (BLACKLIST_FILE_PATH, BLACKLIST_FORMAT_VERSION,
                               BLACKLIST_SCOPE_FULL, BLACKLIST_SCOPE_GENERATION,
                               BLACKLIST_SCOPES, GLOBAL_PROMPTS_FILE_PATH,
                               GUILD_BLOCK_ACTIONS, GUILD_BLOCK_LEAVE,
                               GUILD_BLOCK_QUARANTINE)

#: Discord's "Unknown Webhook". A webhook URL is cached *and persisted*, and
#: `Webhook.from_url` contacts nobody, so a webhook deleted in the server leaves a URL
#: that 404s on every send -- forever, and across restarts, because the dead URL is on
#: disk. Every webhook path must tell this apart from Unknown Message (10008), which is
#: an ordinary outcome when editing or deleting and belongs to the caller.
UNKNOWN_WEBHOOK = 10015

try:
    import orjson as json
except ImportError:
    import json


class ServerManager:
    """Owns server/guild index state, channel webhooks, the global blacklist, global system prompts, and parent bot presence.

    Holds a back-reference to the parent cog for shared instance caches and generic storage helpers,
    per the transitional Dependency Injection pattern in CLAUDE.md.
    """

    def __init__(self, cog):
        self.cog = cog
        self._webhook_from_cache = {}

    def _load_global_prompts(self):
        self.cog.global_prompts = {}
        if os.path.exists(GLOBAL_PROMPTS_FILE_PATH):
            data = IOManager.read_json(GLOBAL_PROMPTS_FILE_PATH)
            if data:
                self.cog.global_prompts = data

    def _save_global_prompts(self):
        IOManager.write_json(self.cog.global_prompts, GLOBAL_PROMPTS_FILE_PATH)

    def _get_server_index(self, server_id_str: str) -> Dict[str, Any]:
        if server_id_str == "dm":
            return {
                "user_active_profiles": {},
                "active_sessions": {"regular": {}, "freewill": {}},
                "freewill_config": {},
                "freewill_participation": {}
            }

        if hasattr(self.cog, 'server_indices') and server_id_str in self.cog.server_indices:
            return self.cog.server_indices[server_id_str]

        path = os.path.join(self.cog.SERVERS_DIR, server_id_str, "index.json")
        index = IOManager.read_json(path)

        if not index:
            index = {
                "user_active_profiles": {},
                "active_sessions": {"regular": {}, "freewill": {}},
                "freewill_config": {},
                "freewill_participation": {}
            }
        else:
            if "active_sessions" not in index or isinstance(index.get("active_sessions"), list):
                index["active_sessions"] = {"regular": {}, "freewill": {}}
            if "freewill_config" not in index:
                index["freewill_config"] = {}
            if "freewill_participation" not in index:
                index["freewill_participation"] = {}

        if hasattr(self.cog, 'server_indices'):
            self.cog.server_indices[server_id_str] = index
        return index

    def _save_server_index(self, server_id_str: str, data: Dict[str, Any]):
        if server_id_str == "dm":
            return

        path = os.path.join(self.cog.SERVERS_DIR, server_id_str, "index.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        IOManager.write_json(data, path)
        if hasattr(self.cog, 'server_indices'):
            self.cog.server_indices[server_id_str] = data

    def _load_channel_webhooks(self):
        self.cog.channel_webhooks = {}
        # Webhook objects rebuilt from cached URLs, so from_url runs once per channel
        # rather than once per message. Reset whenever the URL map is reloaded.
        self._webhook_from_cache = {}
        servers_dir = self.cog.SERVERS_DIR
        if not os.path.isdir(servers_dir):
            return

        for server_id_str in os.listdir(servers_dir):
            server_path = os.path.join(servers_dir, server_id_str)
            if os.path.isdir(server_path):
                webhooks_file = os.path.join(server_path, "webhooks.json.gz")
                if os.path.exists(webhooks_file):
                    server_webhooks_data = self.cog.storage_manager._load_json_gzip(webhooks_file)
                    if server_webhooks_data:
                        # The keys in the file are channel_ids as strings, need to convert to int
                        for ch_id_str, wh_data in server_webhooks_data.items():
                            try:
                                self.cog.channel_webhooks[int(ch_id_str)] = wh_data
                            except ValueError:
                                print(f"Warning: Found non-integer channel ID '{ch_id_str}' in webhook file for server {server_id_str}")

    def _save_channel_webhooks(self, only_servers: Optional[Set[int]] = None):
        """Persist the webhook URL map, one file per server.

        `only_servers` scopes the write to the servers that actually changed.
        Acquiring a webhook touches exactly one channel in one server, but this used
        to rewrite every server's file -- each one a separate zstd compress and
        encrypt -- so the first message in a fresh channel cost one of those
        per guild the bot is in. Pass None only when the change really is global
        (the daily cleanup, which drops orphans across every server at once).

        A server with no webhooks left is still not written, scoped or not: that is
        what leaves its stale file to be re-validated and overwritten on the next
        acquisition, rather than deleting an entry that a transient failure produced.
        """
        try:
            # Group webhooks by server_id
            server_grouped_webhooks = {}
            for channel_id, webhook_data in self.cog.channel_webhooks.items():
                channel = self.cog.bot.get_channel(channel_id)
                if channel and hasattr(channel, 'guild'):
                    server_id = channel.guild.id
                    if only_servers is not None and server_id not in only_servers:
                        continue
                    if server_id not in server_grouped_webhooks:
                        server_grouped_webhooks[server_id] = {}
                    # Store channel_id as string for JSON compatibility
                    server_grouped_webhooks[server_id][str(channel_id)] = webhook_data

            # Save each server's webhooks to its own file
            servers_dir = self.cog.SERVERS_DIR
            for server_id, webhooks_for_server in server_grouped_webhooks.items():
                server_path = os.path.join(servers_dir, str(server_id))
                os.makedirs(server_path, exist_ok=True)
                file_path = os.path.join(server_path, "webhooks.json.gz")
                self.cog.storage_manager._atomic_json_save_gzip(webhooks_for_server, file_path)
        except Exception as e:
            print(f"Error saving sharded channel webhook configurations: {e}"); traceback.print_exc()

    # --- Blacklist -------------------------------------------------------------
    #
    # The records are the operator's; the *sets* are what every gateway path reads.
    # `_rebuild_blacklist_sets` is the only thing that writes them, and it is the reason
    # `on_message` still costs one `int in frozenset` however much detail a record grows.
    # Nothing on a hot path may reach into `blacklist_records`.

    def _load_blacklist(self):
        """Reads blacklist.json in either format and derives the enforcement sets.

        A v1 file is a bare list of user ids, which upgrades to a full block with no
        recorded reason -- the only reading of it that cannot be wrong, since v1 stored
        no scope to narrow and no date to expire.
        """
        records = {"users": {}, "guilds": {}}
        try:
            if os.path.exists(BLACKLIST_FILE_PATH):
                with open(BLACKLIST_FILE_PATH, 'rb') as f:
                    data = json.loads(f.read())

                if isinstance(data, list):
                    for uid in data:
                        records["users"][str(uid)] = {
                            "scope": BLACKLIST_SCOPE_FULL,
                            "reason": "", "at": None, "by": None, "until": None,
                        }
                elif isinstance(data, dict):
                    for key in ("users", "guilds"):
                        section = data.get(key)
                        if isinstance(section, dict):
                            records[key] = {str(k): v for k, v in section.items()
                                            if isinstance(v, dict)}
        except (IOError, json.JSONDecodeError, ValueError) as e:
            # Refusing to enforce is the safe direction here: an unreadable file must not
            # leave a half-parsed list blocking people it never named.
            print(f"Error loading global blacklist: {e}")
            records = {"users": {}, "guilds": {}}

        self.cog.blacklist_records = records
        self._rebuild_blacklist_sets()

    def _save_blacklist(self):
        """Writes v2 and rebuilds the sets. The only writer."""
        try:
            IOManager.write_json({
                "version": BLACKLIST_FORMAT_VERSION,
                "users": self.cog.blacklist_records.get("users", {}),
                "guilds": self.cog.blacklist_records.get("guilds", {}),
            }, BLACKLIST_FILE_PATH)
        except Exception as e:
            print(f"Error saving global blacklist: {e}")
        self._rebuild_blacklist_sets()

    def _rebuild_blacklist_sets(self):
        """Derives the enforcement sets, and the timestamp the expiry sweep waits on.

        Expired entries are skipped rather than deleted -- pruning them is
        `expire_blacklist`'s job, so that this stays free of I/O and can run on any path
        that touches a record.
        """
        now = time.time()
        full, generation = set(), set()
        next_expiry = math.inf

        for uid_str, entry in self.cog.blacklist_records.get("users", {}).items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            until = entry.get("until")
            if until is not None:
                if until <= now:
                    continue
                next_expiry = min(next_expiry, until)
            # An unrecognised scope reads as a full block: a record written by a newer
            # version, or hand-edited, must never fail open.
            if entry.get("scope") != BLACKLIST_SCOPE_GENERATION:
                full.add(uid)
            generation.add(uid)

        leave, quarantine = set(), set()
        for gid_str, entry in self.cog.blacklist_records.get("guilds", {}).items():
            try:
                gid = int(gid_str)
            except (TypeError, ValueError):
                continue
            if entry.get("action") == GUILD_BLOCK_QUARANTINE:
                quarantine.add(gid)
            else:
                leave.add(gid)

        self.cog.global_blacklist = frozenset(full)
        self.cog.generation_blocked = frozenset(generation)
        self.cog.blocked_guilds_leave = frozenset(leave)
        self.cog.quarantined_guilds = frozenset(quarantine)
        self.cog._next_blacklist_expiry = next_expiry

    def expire_blacklist(self) -> bool:
        """Drops entries whose `until` has passed. True if anything was removed.

        Called off the lock heartbeat rather than the daily cleanup: a temporary block
        that lifts up to 24 hours late is not a temporary block. The caller checks
        `_next_blacklist_expiry` first, so the common case never reaches here.
        """
        now = time.time()
        users = self.cog.blacklist_records.get("users", {})
        expired = [uid for uid, entry in users.items()
                   if entry.get("until") is not None and entry["until"] <= now]
        if not expired:
            return False
        for uid in expired:
            users.pop(uid, None)
        self._save_blacklist()
        return True

    def blacklist_entry(self, user_id: int) -> Optional[Dict[str, Any]]:
        """The stored record for a user, or None. For the dashboard, and for the user's
        own standing page -- which must never render `reason` or `by`."""
        return self.cog.blacklist_records.get("users", {}).get(str(int(user_id)))

    def guild_block_entry(self, guild_id: int) -> Optional[Dict[str, Any]]:
        return self.cog.blacklist_records.get("guilds", {}).get(str(int(guild_id)))

    def block_user(self, user_id: int, *, scope: str = BLACKLIST_SCOPE_FULL,
                   reason: str = "", by: Optional[int] = None,
                   until: Optional[float] = None):
        """Adds or replaces a user record. Re-blocking overwrites, so an operator
        correcting a scope or a reason does not have to unblock first."""
        if scope not in BLACKLIST_SCOPES:
            scope = BLACKLIST_SCOPE_FULL
        self.cog.blacklist_records.setdefault("users", {})[str(int(user_id))] = {
            "scope": scope,
            "reason": reason[:300],
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "by": int(by) if by else None,
            "until": until,
        }
        self._save_blacklist()

    def unblock_user(self, user_id: int) -> bool:
        removed = self.cog.blacklist_records.get("users", {}).pop(str(int(user_id)), None)
        if removed is None:
            return False
        self._save_blacklist()
        return True

    def block_guild(self, guild_id: int, *, action: str = GUILD_BLOCK_LEAVE,
                    reason: str = "", by: Optional[int] = None):
        if action not in GUILD_BLOCK_ACTIONS:
            action = GUILD_BLOCK_LEAVE
        self.cog.blacklist_records.setdefault("guilds", {})[str(int(guild_id))] = {
            "action": action,
            "reason": reason[:300],
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "by": int(by) if by else None,
        }
        self._save_blacklist()

    def unblock_guild(self, guild_id: int) -> bool:
        removed = self.cog.blacklist_records.get("guilds", {}).pop(str(int(guild_id)), None)
        if removed is None:
            return False
        self._save_blacklist()
        return True

    async def leave_blocked_guild(self, guild: discord.Guild):
        """Departs one blocked guild, child bots included, and drops what it left behind.

        Every child bot is its own gateway connection in this process, so a parent that
        leaves while the children stay is a block that visibly did not work -- the
        profiles go on speaking through their own bots. The webhook cache and the
        sessions go too: a cached webhook for a server we have left is a URL that will
        be recreated by the next button press, and a session left hydrated goes on
        rotating against a channel nobody can reach.
        """
        suspended = 0
        for channel_id in [c.id for c in guild.channels]:
            if channel_id in self.cog.multi_profile_channels:
                try:
                    if await self.cog.session_manager.suspend_channel_session(channel_id):
                        suspended += 1
                except Exception as e:
                    print(f"[GuildBlock] Session teardown failed for {channel_id}: {e}")
            self._webhook_from_cache.pop(channel_id, None)
        if suspended:
            # Once for the guild, not once per channel: this rewrites every server index.
            self.cog.session_manager._save_multi_profile_sessions()

        for bot_id, client in list(self.cog.child_bot_manager.clients.items()):
            child_guild = client.get_guild(guild.id)
            if child_guild is None:
                continue
            try:
                await child_guild.leave()
            except Exception as e:
                print(f"[GuildBlock] Child bot {bot_id} could not leave {guild.id}: {e}")

        try:
            await guild.leave()
        except Exception as e:
            print(f"[GuildBlock] Could not leave guild {guild.id}: {e}")

    async def enforce_guild_blocks(self):
        """Leaves every blocked guild this instance is still in. Boot sweep.

        `on_guild_join` covers a live invite; this covers a guild joined while the
        process was down, and a guild blocked while it was running.
        """
        for guild in list(self.cog.bot.guilds):
            if guild.id in self.cog.blocked_guilds_leave:
                print(f"[GuildBlock] Leaving blocked guild {guild.id} ({guild.name}).")
                await self.leave_blocked_guild(guild)

    async def apply_ban_effects(self, user_id: int, *, stop_bots: bool = False,
                                unpublish: bool = False,
                                suspend_sessions: bool = False) -> List[str]:
        """The opt-in half of a ban: stops the artefacts that outlive the person.

        Blocking someone leaves their child bots speaking, their published profiles
        borrowable and their sessions rotating, because none of those paths asks who
        owns them. Each of these is deliberately a separate decision, defaulted off:
        unpublishing cannot be undone from the dashboard, and a ban is not always a
        reason to tear down other people's borrows.

        Deliberately *not* a deletion. `_execute_account_deletion` is still the only
        thing that removes a user's data, and it is the user's own to invoke.

        Returns a line per effect applied, for the confirmation to report.
        """
        user_id_str = str(int(user_id))
        applied = []

        if stop_bots:
            bot_ids = [bot_id for bot_id, data in self.cog.child_bots.items()
                       if str(data.get("owner_id")) == user_id_str]
            for bot_id in bot_ids:
                await self.cog.manager_queue.put({"action": "shutdown_bot", "bot_id": bot_id})
                self.cog.child_bots.pop(bot_id, None)
            applied.append(f"Stopped {len(bot_ids)} child bot(s).")

        if unpublish:
            pids = []
            for pub_id, info in self.cog.public_profiles.items():
                if isinstance(info, str) and ":" in info:
                    if info.startswith(user_id_str + ":"):
                        pids.append(pub_id)
                elif isinstance(info, dict) and str(info.get("owner_id")) == user_id_str:
                    pids.append(pub_id)
            for pid in pids:
                self.cog.public_profiles.pop(pid, None)
            if pids:
                self.cog.profile_manager._save_public_index()
            applied.append(f"Unpublished {len(pids)} profile(s) from the Public Library.")

        if suspend_sessions:
            # A channel session has no host -- `_participant_key` is (owner_id,
            # profile_name) and that is the only claim anyone has on one. So this is
            # "every session one of their profiles is seated in", which is also what
            # keeps generating in their name.
            seated = [cid for cid, session in self.cog.multi_profile_channels.items()
                      if any(int(p.get("owner_id") or 0) == int(user_id)
                             for p in session.get("profiles", []))]
            for channel_id in seated:
                await self.cog.session_manager.suspend_channel_session(channel_id)
            if seated:
                self.cog.session_manager._save_multi_profile_sessions()
            applied.append(f"Suspended {len(seated)} session(s).")

        return applied

    # --- Global prompts --------------------------------------------------------

    def _load_parent_presence(self) -> Dict[str, Any]:
        path = os.path.join(self.cog.MOD_DATA_DIR, "parent_presence.json")
        data = IOManager.read_json(path)
        return data if data else {}

    def _save_parent_presence(self, data: Dict[str, Any]):
        path = os.path.join(self.cog.MOD_DATA_DIR, "parent_presence.json")
        IOManager.write_json(data, path)

    def _build_activity_from_dict(self, data: Dict[str, Any]) -> Optional[discord.Activity]:
        atype = data.get("activity_type")
        text = data.get("activity_text")
        url = data.get("activity_url")
        if atype and text:
            act_classes = {
                "playing": discord.ActivityType.playing,
                "watching": discord.ActivityType.watching,
                "listening": discord.ActivityType.listening,
                "competing": discord.ActivityType.competing
            }
            if atype == "streaming": return discord.Streaming(name=text, url=url)
            elif atype in act_classes: return discord.Activity(type=act_classes[atype], name=text)
        return None

    @staticmethod
    def _parent_of(channel: Union[discord.TextChannel, discord.Thread]):
        """The channel a webhook actually belongs to. A thread posts through its parent's."""
        return channel.parent if isinstance(channel, discord.Thread) else channel

    def invalidate_webhook(self, channel: Union[discord.TextChannel, discord.Thread]) -> None:
        """Forget a channel's webhook after Discord said it no longer exists.

        Only the in-memory object is dropped, never the persisted URL: `_save_channel_webhooks`
        rebuilds the per-server files from the entries that remain, so a server left with no
        entries is simply not rewritten and its stale file survives to be re-read at boot.
        Leaving the dead URL in place is safe because the next acquisition re-validates it,
        fails, creates a replacement and overwrites the entry -- which does rewrite the file.
        """
        parent_channel = self._parent_of(channel)
        if parent_channel is not None:
            self._webhook_from_cache.pop(parent_channel.id, None)

    async def _get_or_create_webhook(self, channel: Union[discord.TextChannel, discord.Thread],
                                     *, force_refresh: bool = False) -> Optional[discord.Webhook]:
        parent_channel = self._parent_of(channel)

        # Soft-fail for DMs or environments without guilds
        if not getattr(parent_channel, 'guild', None):
            return None

        try:
            # `_webhook_from_cache` holds only webhooks this process has confirmed exist,
            # so a hit is free. Both branches below used to call parent_channel.webhooks()
            # — a REST round trip — so the cache never actually saved anything; it stored a
            # URL nothing read. Webhook.from_url sends no request, which takes that round
            # trip off the front of every placeholder and every webhook message.
            if not force_refresh:
                cached_wh = self._webhook_from_cache.get(parent_channel.id)
                if cached_wh is not None:
                    return cached_wh

                cached = self.cog.channel_webhooks.get(parent_channel.id)
                if cached and cached.get('url'):
                    try:
                        cached_wh = discord.Webhook.from_url(cached['url'], client=self.cog.bot)
                    except Exception:
                        cached_wh = None

                    if cached_wh is not None:
                        try:
                            # One round trip, once per channel per process -- not per send.
                            # `from_url` validates nothing, so without this a webhook deleted
                            # in the server poisoned the channel permanently: every profile
                            # fell back to a plain bot message under the bot's own name, and
                            # the dead URL was reloaded from disk on the next boot.
                            # prefer_auth=False authenticates with the webhook's own token,
                            # so this works without Manage Webhooks.
                            await cached_wh.fetch(prefer_auth=False)
                            self._webhook_from_cache[parent_channel.id] = cached_wh
                            return cached_wh
                        except discord.NotFound:
                            print(f"Webhook for #{parent_channel.name} no longer exists; recreating.")
                        except Exception:
                            # A rate limit or a 5xx says nothing about whether the webhook
                            # exists. Hand it over unvalidated -- the old behaviour -- rather
                            # than creating a second webhook against a transient failure.
                            return cached_wh

            # No usable cache entry: fetch or create, which does cost a round trip.
            webhooks = await parent_channel.webhooks()
            bot_webhook = next((wh for wh in webhooks if wh.user and wh.user.id == self.cog.bot.user.id), None)
            if not bot_webhook:
                bot_webhook = await parent_channel.create_webhook(name=f"{self.cog.bot.user.name} Webhook", reason="For custom appearances")
            
            self.cog.channel_webhooks[parent_channel.id] = {'url': bot_webhook.url}
            self._webhook_from_cache[parent_channel.id] = bot_webhook
            # One channel changed, so only its server's file is rewritten.
            self._save_channel_webhooks(only_servers={parent_channel.guild.id})
            return bot_webhook
        except discord.Forbidden:
            # The usual cause of a whole server delivering under the bot's own name.
            print(f"Missing Manage Webhooks in #{getattr(parent_channel, 'name', '?')}; "
                  f"profiles will speak as the bot there.")
        except Exception as e:
            print(f"Failed to get/create webhook for {parent_channel.name}: {e}")
        return None

    async def run_webhook(self, channel: Union[discord.TextChannel, discord.Thread],
                          op: str, *args, **kwargs):
        """Perform one webhook operation, healing a webhook Discord no longer has.

        Returns None when the channel has no webhook to be had at all (a DM, or missing
        Manage Webhooks), which is the caller's cue to fall back to a plain message.
        Unknown Webhook (10015) is the only error retried, and only once, against a
        freshly created one; Unknown Message (10008) is a normal result of editing or
        deleting something a child bot owns and must reach the caller unchanged.
        """
        for attempt in (0, 1):
            webhook = await self._get_or_create_webhook(channel, force_refresh=bool(attempt))
            if webhook is None:
                return None
            try:
                return await getattr(webhook, op)(*args, **kwargs)
            except discord.HTTPException as e:
                if attempt or e.code != UNKNOWN_WEBHOOK:
                    raise
                self.invalidate_webhook(channel)
                self._rewind_files(args, kwargs)
        return None

    @staticmethod
    def _rewind_files(args, kwargs) -> None:
        """Seek any file about to be sent a second time back to the start.

        The failed attempt read the whole multipart body, and discord.py only rewinds on
        its *own* internal retries (`reset(seek=tries)`, and a fresh request starts at
        tries 0) -- so a retried image would otherwise upload as zero bytes.
        """
        for value in list(args) + list(kwargs.values()):
            items = value if isinstance(value, (list, tuple)) else (value,)
            for item in items:
                if isinstance(item, discord.File):
                    try:
                        item.reset(seek=True)
                    except Exception:
                        pass

