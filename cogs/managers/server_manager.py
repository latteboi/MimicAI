import os
import traceback
from typing import Dict, Any, Optional, Union
import discord

from .storage_manager import IOManager
from ..utils.constants import GLOBAL_PROMPTS_FILE_PATH, BLACKLIST_FILE_PATH

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

    def _save_channel_webhooks(self):
        try:
            # Group webhooks by server_id
            server_grouped_webhooks = {}
            for channel_id, webhook_data in self.cog.channel_webhooks.items():
                channel = self.cog.bot.get_channel(channel_id)
                if channel and hasattr(channel, 'guild'):
                    server_id = channel.guild.id
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

    def _load_blacklist(self):
        try:
            if os.path.exists(BLACKLIST_FILE_PATH):
                with open(BLACKLIST_FILE_PATH, 'rb') as f:
                    user_ids = json.loads(f.read())
                    self.cog.global_blacklist = set(user_ids)
            else:
                self.cog.global_blacklist = set()
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading global blacklist: {e}")
            self.cog.global_blacklist = set()

    def _save_blacklist(self):
        try:
            IOManager.write_json(list(self.cog.global_blacklist), BLACKLIST_FILE_PATH)
        except Exception as e:
            print(f"Error saving global blacklist: {e}")

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
            self._save_channel_webhooks()
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

