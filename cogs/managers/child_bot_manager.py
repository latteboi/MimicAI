import os
import io
import time
import uuid
import base64
import asyncio
import traceback
from typing import Dict, Any, Optional, Tuple, List, get_args
import aiohttp
from PIL import Image

import discord
from discord import app_commands
from discord.ext import commands

from ..utils.constants import PLACEHOLDER_EMOJI, HarmBlockThreshold, HarmCategory, IMAGE_QUEUE_PRIORITY
from ..utils.helpers import _split_into_sentences_with_abbreviations
from .storage_manager import IOManager

MAX_AVATAR_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class ChildBotManager:
    """Owns in-process child bot client instances, command trees, presence synchronization,
    and direct Discord API delivery without subprocesses or WebSocket IPC.
    """

    def __init__(self, cog):
        self.cog = cog
        self.clients: Dict[str, commands.Bot] = {}
        self.bot_tasks: Dict[str, asyncio.Task] = {}
        self.typing_tasks: Dict[Tuple[str, int], asyncio.Task] = {}
        self.pending_toggles: Dict[str, Dict[str, Any]] = {}
        self.queue_worker_task: Optional[asyncio.Task] = None

    def _find_borrowed_name_for_owner(self, author_id: int, original_owner_id: int, original_profile_name: str) -> Optional[str]:
        """Finds the name under which author_id has borrowed original_owner_id's original_profile_name profile, if any."""
        author_index = self.cog.profile_manager._get_user_index(author_id)
        for b_name in author_index.get("borrowed", []):
            b_data = self.cog.profile_manager._get_profile_config(author_id, b_name, True) or {}
            if int(b_data.get("original_owner_id", 0)) == original_owner_id and b_data.get("original_profile_name") == original_profile_name:
                return b_name
        return None

    def _load_child_bots(self):
        self.cog.child_bots = {}
        self.cog.child_bots_by_owner_profile = {}
        if not os.path.isdir(self.cog.USERS_DIR):
            return

        for user_id_str in os.listdir(self.cog.USERS_DIR):
            if not user_id_str.isdigit():
                continue
            profiles_dir = os.path.join(self.cog.USERS_DIR, user_id_str, "profiles")
            if not os.path.isdir(profiles_dir):
                continue

            for pid_folder in os.listdir(profiles_dir):
                profile_file = os.path.join(profiles_dir, pid_folder, "profile.json.gz")
                if os.path.exists(profile_file):
                    profile_data = IOManager.read_json_gzip(profile_file, self.cog.fernet)
                    if profile_data and profile_data.get("child_bot"):
                        bot_data = profile_data["child_bot"]
                        if isinstance(bot_data, dict) and "bot_id" in bot_data:
                            bot_data["owner_id"] = int(user_id_str)
                            bot_data["profile_name"] = profile_data.get("name", pid_folder)
                            bot_data["pid"] = pid_folder
                            self.cog.child_bots[bot_data["bot_id"]] = bot_data
                            self.cog.child_bots_by_owner_profile[(bot_data["owner_id"], bot_data["profile_name"])] = bot_data["bot_id"]

    async def start_all_child_bots(self):
        """Launches all configured child bots directly into the shared asyncio event loop."""
        self._load_child_bots()
        if not self.queue_worker_task or self.queue_worker_task.done():
            self.queue_worker_task = asyncio.create_task(self._manager_queue_listener())

        for bot_id, config in list(self.cog.child_bots.items()):
            if bot_id in self.clients:
                continue
            try:
                token_encrypted = config.get("token_encrypted")
                if not token_encrypted:
                    continue
                try:
                    token = self.cog.fernet.decrypt(token_encrypted.encode()).decode()
                except Exception:
                    token = token_encrypted

                await self.launch_bot(
                    bot_id=bot_id,
                    token=token,
                    owner_id=config.get("owner_id"),
                    profile_name=config.get("profile_name"),
                    profile_id=config.get("pid"),
                    presence=config.get("presence")
                )
            except Exception as e:
                print(f"[ChildBotManager] Failed to launch child bot {bot_id}: {e}")

    async def launch_bot(self, bot_id: str, token: str, owner_id: Optional[int] = None, profile_name: Optional[str] = None, profile_id: Optional[str] = None, presence: Optional[Dict] = None):
        """Instantiates a dedicated discord client running concurrently in the main event loop."""
        if bot_id in self.clients:
            return

        intents = discord.Intents.none()
        intents.guilds = True

        child = commands.Bot(command_prefix="!", intents=intents)
        self.clients[bot_id] = child
        parent_name = self.cog.bot.user.name if self.cog.bot.user else "MimicAI"
        parent_id = self.cog.bot.user.id if self.cog.bot.user else 0

        # --- Register Commands ---
        @child.tree.command(name="whoami", description="Displays information about this bot's identity.")
        async def whoami(interaction: discord.Interaction):
            embed = discord.Embed(
                title=f"Bot Identity: {child.user.name}",
                description=f"Managed by {parent_name}.",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=child.user.display_avatar.url)
            owner_mention = f"<@{owner_id}>" if owner_id else "Unknown"
            embed.add_field(name="Profile Owner", value=owner_mention, inline=True)
            embed.add_field(name="Profile ID", value=str(profile_id) if profile_id else "Unknown", inline=True)
            embed.add_field(name="Profile Name", value=str(profile_name) if profile_name else "Unknown", inline=True)

            if owner_id and interaction.user.id == int(owner_id):
                class InProcessPresenceView(discord.ui.View):
                    def __init__(view_self, b_id):
                        super().__init__(timeout=120)
                        view_self.b_id = b_id

                        status_options = [
                            discord.SelectOption(label="Online", value="online", emoji="🟢"),
                            discord.SelectOption(label="Idle", value="idle", emoji="🌙"),
                            discord.SelectOption(label="Do Not Disturb", value="dnd", emoji="⛔"),
                            discord.SelectOption(label="Invisible", value="invisible", emoji="🔘")
                        ]
                        view_self.status_select = discord.ui.Select(placeholder="Change Online Status...", options=status_options, row=0)
                        view_self.status_select.callback = view_self.status_callback
                        view_self.add_item(view_self.status_select)

                        activity_options = [
                            discord.SelectOption(label="Playing...", value="playing", emoji="🎮"),
                            discord.SelectOption(label="Watching...", value="watching", emoji="📺"),
                            discord.SelectOption(label="Listening to...", value="listening", emoji="🎧"),
                            discord.SelectOption(label="Competing in...", value="competing", emoji="🏆"),
                            discord.SelectOption(label="Streaming...", value="streaming", emoji="🟪")
                        ]
                        view_self.act_select = discord.ui.Select(placeholder="Set Activity Type...", options=activity_options, row=1)
                        view_self.act_select.callback = view_self.act_callback
                        view_self.add_item(view_self.act_select)

                        clear_btn = discord.ui.Button(label="Clear Activity", style=discord.ButtonStyle.danger, row=2)
                        clear_btn.callback = view_self.clear_callback
                        view_self.add_item(clear_btn)

                    async def status_callback(view_self, i: discord.Interaction):
                        status_map = {"online": discord.Status.online, "idle": discord.Status.idle, "dnd": discord.Status.dnd, "invisible": discord.Status.invisible}
                        val = view_self.status_select.values[0]
                        act = child.activity
                        await child.change_presence(status=status_map[val], activity=act)
                        await self.handle_child_bot_presence({"bot_id": view_self.b_id, "presence": {"status": val}})
                        await i.response.send_message(f"Status changed to **{val.title()}**.", ephemeral=True)

                    async def act_callback(view_self, i: discord.Interaction):
                        val = view_self.act_select.values[0]

                        class InProcessActModal(discord.ui.Modal, title="Set Activity Details"):
                            text_in = discord.ui.TextInput(label="Activity Text", placeholder="e.g. the conversation", required=True, max_length=128)

                            def __init__(modal_self, b_id, atype):
                                super().__init__()
                                modal_self.b_id = b_id
                                modal_self.atype = atype
                                if atype == "streaming":
                                    modal_self.url_in = discord.ui.TextInput(label="Twitch/YouTube URL", placeholder="https://twitch.tv/example", required=True)
                                    modal_self.add_item(modal_self.url_in)

                            async def on_submit(modal_self, mi: discord.Interaction):
                                text = modal_self.text_in.value.strip()
                                url = getattr(modal_self, "url_in", None)
                                url_val = url.value.strip() if url else None

                                act_classes = {"playing": discord.ActivityType.playing, "watching": discord.ActivityType.watching, "listening": discord.ActivityType.listening, "competing": discord.ActivityType.competing}
                                if modal_self.atype == "streaming":
                                    act = discord.Streaming(name=text, url=url_val)
                                else:
                                    act = discord.Activity(type=act_classes[modal_self.atype], name=text)

                                stat = child.status
                                await child.change_presence(status=stat, activity=act)
                                await self.handle_child_bot_presence({"bot_id": modal_self.b_id, "presence": {"activity_type": modal_self.atype, "activity_text": text, "activity_url": url_val}})
                                await mi.response.send_message(f"Activity set to **{modal_self.atype.title()} {text}**.", ephemeral=True)

                        await i.response.send_modal(InProcessActModal(view_self.b_id, val))

                    async def clear_callback(view_self, i: discord.Interaction):
                        stat = child.status
                        await child.change_presence(status=stat, activity=None)
                        await self.handle_child_bot_presence({"bot_id": view_self.b_id, "presence": {"activity_type": None, "activity_text": None, "activity_url": None}})
                        await i.response.send_message("Activity cleared.", ephemeral=True)

                await interaction.response.send_message(embed=embed, view=InProcessPresenceView(bot_id), ephemeral=True)
            else:
                await interaction.response.send_message(embed=embed, ephemeral=True)

        @child.tree.command(name="toggle", description="Toggles this bot's participation in this channel (Admin Only).")
        @app_commands.checks.has_permissions(administrator=True)
        async def toggle(interaction: discord.Interaction):
            if parent_id and interaction.guild:
                try:
                    await interaction.guild.fetch_member(parent_id)
                except discord.NotFound:
                    await interaction.response.send_message("You need to invite MimicAI (parent bot) to this server to use child bots.", ephemeral=True)
                    return
                except Exception:
                    await interaction.response.send_message("An error occurred while verifying parent bot presence.", ephemeral=True)
                    return

            await interaction.response.defer(ephemeral=True)
            correlation_id = str(uuid.uuid4())

            toggle_data = {
                "action": "toggle_session_participation",
                "bot_id": bot_id,
                "channel_id": interaction.channel_id,
                "guild_id": interaction.guild_id,
                "user_id": interaction.user.id,
                "correlation_id": correlation_id
            }

            res = await self.handle_child_bot_toggle(toggle_data)
            await interaction.followup.send(res or "Updated channel participation.", ephemeral=True)

        # Background runner task
        async def runner():
            try:
                async def enforce_bio_loop():
                    await child.wait_until_ready()
                    while not child.is_closed():
                        try:
                            app_info = await child.application_info()
                            sig = f"Managed by {parent_name}."
                            if not app_info.description or sig not in app_info.description:
                                new_desc = (app_info.description or "")
                                if len(new_desc) + len(sig) > 400:
                                    new_desc = new_desc[:400 - len(sig)]
                                new_desc += sig
                                await child.http.request(discord.http.Route('PATCH', '/applications/@me'), json={'description': new_desc})
                        except Exception as e:
                            print(f"[ChildBotManager] Could not enforce bio for {bot_id}: {e}")
                        await asyncio.sleep(3600)

                async def on_ready_sync():
                    await child.wait_until_ready()
                    try:
                        if presence:
                            status_val = presence.get("status", "online")
                            status_map = {"online": discord.Status.online, "idle": discord.Status.idle, "dnd": discord.Status.dnd, "invisible": discord.Status.invisible}
                            atype = presence.get("activity_type")
                            text = presence.get("activity_text")
                            url = presence.get("activity_url")
                            act = None
                            if atype and text:
                                act_classes = {"playing": discord.ActivityType.playing, "watching": discord.ActivityType.watching, "listening": discord.ActivityType.listening, "competing": discord.ActivityType.competing}
                                if atype == "streaming":
                                    act = discord.Streaming(name=text, url=url)
                                elif atype in act_classes:
                                    act = discord.Activity(type=act_classes[atype], name=text)

                            await child.change_presence(status=status_map.get(status_val, discord.Status.online), activity=act)

                        await child.tree.sync()
                    except Exception as e:
                        print(f"[ChildBotManager] {bot_id} sync failed: {e}")

                # Use standard asyncio.create_task on the active event loop instead of child.loop
                asyncio.create_task(enforce_bio_loop())
                asyncio.create_task(on_ready_sync())
                await child.start(token)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[ChildBotManager] Child bot {bot_id} failed: {e}")
            finally:
                self.clients.pop(bot_id, None)

        self.bot_tasks[bot_id] = asyncio.create_task(runner())

    async def shutdown_bot(self, bot_id: str):
        """Gracefully disconnects and terminates an in-process child bot client."""
        bot_id_str = str(bot_id)
        child = self.clients.get(bot_id_str)
        if child:
            try:
                await child.change_presence(status=discord.Status.offline)
                await child.close()
            except Exception:
                pass
            self.clients.pop(bot_id_str, None)

        task = self.bot_tasks.get(bot_id_str)
        if task and not task.done():
            task.cancel()
            self.bot_tasks.pop(bot_id_str, None)

    async def shutdown_all(self):
        """Terminates all running child bot instances."""
        if self.queue_worker_task and not self.queue_worker_task.done():
            self.queue_worker_task.cancel()
        for bot_id in list(self.clients.keys()):
            await self.shutdown_bot(bot_id)

    async def _manager_queue_listener(self):
        """Asynchronously consumes manager queue instructions for cog interoperability."""
        while True:
            try:
                command = await self.cog.manager_queue.get()
                action = command.get("action")

                if action == "launch_bot":
                    bot_id = str(command.get("bot_id"))
                    token = command.get("token")
                    cfg = command.get("config") or {}
                    self.cog.child_bots[bot_id] = cfg
                    await self.launch_bot(
                        bot_id=bot_id,
                        token=token,
                        owner_id=cfg.get("owner_id"),
                        profile_name=cfg.get("profile_name"),
                        profile_id=cfg.get("pid"),
                        presence=cfg.get("presence")
                    )

                elif action == "shutdown_bot":
                    bot_id = str(command.get("bot_id"))
                    self.cog.child_bots.pop(bot_id, None)
                    await self.shutdown_bot(bot_id)

                elif action == "send_to_child":
                    bot_id = str(command.get("bot_id"))
                    payload = command.get("payload", {})
                    child_action = payload.get("action")

                    if child_action == "send_message":
                        asyncio.create_task(self.execute_send(bot_id, payload))
                    elif child_action == "regenerate_message":
                        asyncio.create_task(self.execute_regenerate(bot_id, payload))
                    elif child_action == "delete_message":
                        asyncio.create_task(self.execute_delete(bot_id, payload))
                    elif child_action == "start_typing":
                        asyncio.create_task(self.execute_typing(bot_id, payload))
                    elif child_action == "stop_typing":
                        ch_id = payload.get("channel_id")
                        t_key = (bot_id, ch_id)
                        if t_key in self.typing_tasks:
                            self.typing_tasks[t_key].cancel()
                            self.typing_tasks.pop(t_key, None)
                    elif child_action in ["update_username", "update_avatar"]:
                        asyncio.create_task(self.update_appearance(bot_id, payload))

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ChildBotManager] Queue listener error: {e}")

    async def stop_typing(self, bot_id: str, channel_id: int):
        """Immediately stops the typing loop for a given child bot and channel."""
        task_key = (str(bot_id), channel_id)
        if task_key in self.typing_tasks:
            self.typing_tasks[task_key].cancel()
            self.typing_tasks.pop(task_key, None)

    async def execute_send(self, bot_id: str, payload: Dict[str, Any]) -> List[discord.Message]:
        channel_id = payload.get("channel_id")
        task_key = (str(bot_id), channel_id)
        if task_key in self.typing_tasks:
            self.typing_tasks[task_key].cancel()
            self.typing_tasks.pop(task_key, None)

        bot = self.clients.get(str(bot_id))
        if not bot or not bot.is_ready():
            return []

        content = payload.get("content", "")
        attachment_data = payload.get("attachment")
        realistic_typing = payload.get("realistic_typing", False)
        typing_cps = payload.get("typing_cps", 30.0)
        typing_max_delay = payload.get("typing_max_delay", 2.5)
        typing_mode = payload.get("typing_mode", "sentence")
        reply_to_id = payload.get("reply_to_id")
        ping = payload.get("ping", False)

        try:
            channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
            if not channel:
                return []
        except Exception:
            return []

        file_to_send = None
        if attachment_data:
            try:
                image_bytes = base64.b64decode(attachment_data['data_base64'])
                file_to_send = discord.File(io.BytesIO(image_bytes), filename=attachment_data.get('filename', 'attachment.png'))
            except Exception as e:
                print(f"[ChildBotManager] Bot {bot_id} attachment decode error: {e}")

        sent_messages: List[discord.Message] = []
        try:
            # Handle attachment-only delivery with no text body
            if not content.strip() and file_to_send:
                kwargs = {"file": file_to_send}
                if reply_to_id:
                    try:
                        ref_msg = await channel.fetch_message(reply_to_id)
                        m = await ref_msg.reply(mention_author=ping, **kwargs)
                    except Exception:
                        m = await channel.send(**kwargs)
                else:
                    m = await channel.send(**kwargs)
                sent_messages.append(m)
                return sent_messages

            if realistic_typing and content.strip():
                chunks = _split_into_sentences_with_abbreviations(content)
                displayed_text = ""
                last_edit_time = 0
                sent_message = None

                try:
                    typing_cps_float = float(typing_cps)
                    if typing_cps_float <= 0:
                        typing_cps_float = 30.0
                except Exception:
                    typing_cps_float = 30.0

                try:
                    typing_max_delay_float = float(typing_max_delay)
                except Exception:
                    typing_max_delay_float = 2.5

                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    delay = max(0.5, min(len(chunk) / typing_cps_float, typing_max_delay_float))
                    await asyncio.sleep(delay)

                    separator = "\n" if typing_mode == "line" and displayed_text else (" " if displayed_text else "")
                    if len(displayed_text) + len(separator) + len(chunk) > 2000:
                        displayed_text = chunk
                        sent_message = None
                    else:
                        displayed_text += separator + chunk

                    kwargs = {"content": displayed_text}
                    if not sent_message:
                        if i == 0:
                            if file_to_send:
                                kwargs["file"] = file_to_send
                            if reply_to_id:
                                try:
                                    ref_msg = await channel.fetch_message(reply_to_id)
                                    sent_message = await ref_msg.reply(mention_author=ping, **kwargs)
                                except Exception:
                                    sent_message = await channel.send(**kwargs)
                            else:
                                sent_message = await channel.send(**kwargs)
                        else:
                            sent_message = await channel.send(**kwargs)
                        sent_messages.append(sent_message)
                        last_edit_time = asyncio.get_running_loop().time()
                    else:
                        now = asyncio.get_running_loop().time()
                        if now - last_edit_time < 1.5:
                            await asyncio.sleep(1.5 - (now - last_edit_time))
                        try:
                            await sent_message.edit(content=displayed_text)
                        except Exception:
                            pass
                        last_edit_time = asyncio.get_running_loop().time()
            else:
                remaining_content = content
                first_chunk = True
                while remaining_content:
                    if len(remaining_content) <= 2000:
                        chunk = remaining_content
                        remaining_content = ""
                    else:
                        split_pos = remaining_content.rfind('\n', 0, 2000)
                        if split_pos == -1:
                            split_pos = 2000
                        chunk = remaining_content[:split_pos]
                        remaining_content = remaining_content[split_pos:]

                    kwargs = {"content": chunk}
                    if first_chunk:
                        if file_to_send:
                            kwargs["file"] = file_to_send
                        if reply_to_id:
                            try:
                                ref_msg = await channel.fetch_message(reply_to_id)
                                m = await ref_msg.reply(mention_author=ping, **kwargs)
                            except Exception:
                                m = await channel.send(**kwargs)
                        else:
                            m = await channel.send(**kwargs)
                        first_chunk = False
                    else:
                        m = await channel.send(**kwargs)
                    sent_messages.append(m)

            # Resolve pending confirmations so message IDs are registered in session history
            correlation_id = payload.get("correlation_id")
            if correlation_id:
                await self.handle_child_bot_confirmation({
                    "correlation_id": correlation_id,
                    "message_ids": [m.id for m in sent_messages]
                })

            return sent_messages
        except Exception as e:
            print(f"[ChildBotManager] Delivery error for {bot_id}: {e}")
            return sent_messages
        finally:
            if file_to_send:
                file_to_send.close()

    async def execute_typing(self, bot_id: str, payload: Dict[str, Any]):
        bot = self.clients.get(str(bot_id))
        if not bot or not bot.is_ready():
            return

        channel_id = payload.get("channel_id")
        task_key = (str(bot_id), channel_id)
        if task_key in self.typing_tasks:
            return

        async def typing_loop():
            try:
                channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
                if not channel:
                    return
                start_time = asyncio.get_running_loop().time()
                while True:
                    if asyncio.get_running_loop().time() - start_time > 60:
                        break
                    await channel.typing()
                    await asyncio.sleep(7)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[ChildBotManager] Typing loop error for {bot_id}: {e}")
            finally:
                if self.typing_tasks.get(task_key) == asyncio.current_task():
                    self.typing_tasks.pop(task_key, None)

        self.typing_tasks[task_key] = asyncio.create_task(typing_loop())

    async def update_appearance(self, bot_id: str, payload: Dict[str, Any]):
        bot = self.clients.get(str(bot_id))
        if not bot or not bot.is_ready():
            return

        action = payload.get("action")
        try:
            if action == "update_username":
                await bot.user.edit(username=payload.get("username"))
            elif action == "update_avatar":
                url = payload.get("avatar_url")
                if url:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                if len(data) < MAX_AVATAR_SIZE_BYTES:
                                    try:
                                        with Image.open(io.BytesIO(data)) as img:
                                            if img.mode != 'RGBA':
                                                img = img.convert('RGBA')
                                            with io.BytesIO() as out_buffer:
                                                img.save(out_buffer, format='PNG')
                                                png_data = out_buffer.getvalue()
                                        await bot.user.edit(avatar=png_data)
                                    except Exception:
                                        await bot.user.edit(avatar=data)
                else:
                    await bot.user.edit(avatar=None)
        except Exception as e:
            print(f"[ChildBotManager] Appearance update error for {bot_id}: {e}")

    async def execute_delete(self, bot_id: str, payload: Dict[str, Any]):
        bot = self.clients.get(str(bot_id))
        if not bot or not bot.is_ready():
            return
        channel_id = payload.get("channel_id")
        message_id = payload.get("message_id")
        try:
            channel = await bot.fetch_channel(channel_id)
            if channel:
                message = await channel.fetch_message(message_id)
                if message:
                    await message.delete()
        except discord.NotFound:
            pass
        except Exception as e:
            print(f"[ChildBotManager] Delete error for {bot_id}: {e}")

    async def execute_regenerate(self, bot_id: str, payload: Dict[str, Any]):
        bot = self.clients.get(str(bot_id))
        if not bot or not bot.is_ready():
            return
        channel_id = payload.get("channel_id")
        message_id = payload.get("message_id")
        content = payload.get("content")
        try:
            channel = await bot.fetch_channel(channel_id)
            if not channel:
                return
            message = await channel.fetch_message(message_id)
            if not message:
                return
            if message.attachments:
                kept = [a for a in message.attachments if a.content_type and a.content_type.startswith("image/")]
                await message.edit(content=content, attachments=kept)
            else:
                await message.edit(content=content)
        except discord.NotFound:
            pass
        except Exception as e:
            print(f"[ChildBotManager] Regeneration edit error for {bot_id}: {e}")

    async def handle_child_bot_event(self, event_data: Dict):
        if event_data.get("message", {}).get("author_id") in self.cog.global_blacklist:
            return

        event_type = event_data.get("event_type")
        bot_id = str(event_data.get("bot_id"))

        if event_type == "message_received":
            message_payload = event_data.get("message", {})
            channel_id = message_payload.get("channel_id")
            guild_id = message_payload.get("guild_id")
            message_id = message_payload.get("id")

            if message_id:
                self.cog.processed_child_messages[message_id] = True

            bot_config = self.cog.child_bots.get(bot_id)
            if not bot_config:
                return

            original_owner_id = bot_config['owner_id']
            original_profile_name = bot_config['profile_name']
            effective_owner_id = original_owner_id
            effective_profile_name = original_profile_name

            guild = self.cog.bot.get_guild(guild_id) if guild_id else None
            if guild and not guild.get_member(original_owner_id):
                author_id = message_payload.get("author_id")
                borrowed_name = self._find_borrowed_name_for_owner(author_id, original_owner_id, original_profile_name)

                if borrowed_name:
                    effective_owner_id = author_id
                    effective_profile_name = borrowed_name
                else:
                    session = self.cog.multi_profile_channels.get(channel_id)
                    found_in_session = False
                    if session:
                        for p in session.get("profiles", []):
                            p_index = self.cog.profile_manager._get_user_index(p['owner_id'])
                            if p['profile_name'] in p_index.get("borrowed", []):
                                b_data = self.cog.profile_manager._get_profile_config(p['owner_id'], p['profile_name'], True) or {}
                                if int(b_data.get("original_owner_id", 0)) == original_owner_id and b_data.get("original_profile_name") == original_profile_name:
                                    effective_owner_id = p['owner_id']
                                    effective_profile_name = p['profile_name']
                                    found_in_session = True
                                    break

                    if not found_in_session:
                        await self.execute_send(bot_id, {
                            "channel_id": channel_id,
                            "content": "My original owner is not in this server, and you have not borrowed my profile. Use `/profile hub` to find and borrow me first!"
                        })
                        return

            session = self.cog.multi_profile_channels.get(channel_id)
            ephemeral_participant = {
                "owner_id": effective_owner_id,
                "profile_name": effective_profile_name,
                "method": "child_bot",
                "bot_id": bot_id,
                "ephemeral": True
            }

            if not session:
                session = {
                    "type": "multi", "unified_log": [], "is_hydrated": False,
                    "last_bot_message_id": None, "owner_id": message_payload.get("author_id"), "is_running": False,
                    "task_queue": asyncio.Queue(), "worker_task": None, "turns_since_last_ltm": 0,
                    "session_prompt": None, "session_mode": "sequential", "profiles": []
                }
                self.cog.multi_profile_channels[channel_id] = session

            trigger = ('child_mention', message_payload, ephemeral_participant)
            await session['task_queue'].put(trigger)

            if not session.get('worker_task') or session['worker_task'].done():
                task = self.cog.bot.loop.create_task(self.cog.generation_service._multi_profile_worker(channel_id))
                session['worker_task'] = task
                self.cog.background_tasks.add(task)

    async def handle_child_bot_presence(self, event_data: Dict):
        bot_id = str(event_data.get("bot_id"))
        presence_update = event_data.get("presence")
        if not bot_id or not presence_update:
            return

        bot_config = self.cog.child_bots.get(bot_id)
        if bot_config:
            owner_id = bot_config['owner_id']
            profile_name = bot_config['profile_name']

            def _sync_update_presence():
                return self.cog.profile_manager._update_child_bot_presence(owner_id, profile_name, presence_update)

            bot_config["presence"] = await asyncio.to_thread(_sync_update_presence)

    async def handle_child_bot_image_request(self, event_data: Dict):
        if event_data.get("message", {}).get("author_id") in self.cog.global_blacklist:
            return

        bot_id = str(event_data.get("bot_id"))
        message_data = event_data.get("message", {})
        channel_id = message_data.get("channel_id")

        if channel_id in self.cog.multi_profile_channels:
            return

        async def send_notification_to_child(content: str):
            await self.execute_send(bot_id, {"channel_id": channel_id, "content": f"(Notice for {message_data['author_name']}): {content}"})

        try:
            if self.cog.image_request_queue.full():
                await send_notification_to_child("The image generation backlog is currently full. Please try again in a moment.")
                return

            if self.cog.image_gen_semaphore.locked():
                qsize = self.cog.image_request_queue.qsize()
                await send_notification_to_child(f"Your image generation request has been queued. You are #{qsize + 1} in line.")
            else:
                await self.execute_typing(bot_id, {"channel_id": channel_id})

            bot_config = self.cog.child_bots.get(bot_id)
            if not bot_config:
                return

            guild_id = message_data.get("guild_id")
            original_owner_id = bot_config['owner_id']
            original_profile_name = bot_config['profile_name']
            owner_id = original_owner_id
            profile_name = original_profile_name

            guild = self.cog.bot.get_guild(guild_id) if guild_id else None
            if guild and not guild.get_member(original_owner_id):
                author_id = message_data.get("author_id")
                borrowed_name = self._find_borrowed_name_for_owner(author_id, original_owner_id, original_profile_name)
                if borrowed_name:
                    owner_id = author_id
                    profile_name = borrowed_name
                else:
                    await send_notification_to_child("My original owner is not in this server, and you have not borrowed my profile. Use `/profile hub` to find and borrow me first!")
                    return

            index = self.cog.profile_manager._get_user_index(owner_id)
            is_borrowed = profile_name in index.get("borrowed", [])
            profile_data = self.cog.profile_manager._get_profile_config(owner_id, profile_name, is_borrowed) or {}

            placeholder_message_obj = None
            if profile_data.get("child_bot_placeholder", False):
                custom_emoji = profile_data.get("placeholder_emoji") or PLACEHOLDER_EMOJI
                msg_id = await self.cog.generation_service._send_child_bot_placeholder(bot_id, channel_id, custom_emoji)
                if msg_id:
                    try:
                        ch = self.cog.bot.get_channel(channel_id)
                        placeholder_message_obj = await ch.fetch_message(msg_id)
                    except Exception:
                        pass
            else:
                await self.execute_typing(bot_id, {"channel_id": channel_id})

            image_prefixes = ("!image", "!imagine")
            used_prefix = next((p for p in image_prefixes if message_data.get("content", "").lower().startswith(p)), "!image")
            prompt_text = message_data.get("content", "")[len(used_prefix):].strip()
            if not prompt_text:
                return

            if not profile_data.get("image_generation_enabled", False):
                return

            safety_level_str = profile_data.get("safety_level", "low")
            safety_map = {"unrestricted": HarmBlockThreshold.BLOCK_NONE, "low": HarmBlockThreshold.BLOCK_ONLY_HIGH, "medium": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, "high": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE}
            threshold = safety_map.get(safety_level_str, HarmBlockThreshold.BLOCK_ONLY_HIGH)
            dynamic_safety_settings = {cat: threshold for cat in get_args(HarmCategory)}

            source_owner_id = owner_id
            source_profile_name = profile_name
            if is_borrowed:
                borrowed_data = self.cog.profile_manager._get_profile_config(owner_id, profile_name, True) or {}
                source_owner_id = int(borrowed_data.get("original_owner_id", owner_id))
                source_profile_name = borrowed_data.get("original_profile_name", profile_name)

            source_prompts = self.cog.profile_manager._get_profile_prompts(source_owner_id, source_profile_name) or {}
            persona = source_prompts.get("persona", {})
            appearance_lines_encrypted = persona.get("appearance", [])
            appearance_text = "\n".join([self.cog.storage_manager._decrypt_data(line) for line in appearance_lines_encrypted])

            bot_user = self.cog.bot.get_user(int(bot_id))
            bot_display_name = bot_user.name if bot_user else profile_name

            final_prompt_text = prompt_text
            if appearance_text.strip():
                prompt_lower = prompt_text.lower()
                second_person_pronouns = ["you", "your", "yourself", "u", "ur"]
                if any(pronoun in prompt_lower.split() for pronoun in second_person_pronouns) or \
                   bot_display_name.lower() in prompt_lower or \
                   profile_name.lower() in prompt_lower:
                    final_prompt_text = f"Your appearance:\n{appearance_text.strip()}\n\nUser's prompt:\n{prompt_text}"

            system_instruction = self.cog.media_service._get_image_gen_system_instruction(owner_id, profile_name)

            reference_image_urls = []
            replied_to_data = message_data.get("replied_to")
            if replied_to_data and replied_to_data.get("attachment_url"):
                reference_image_urls.append({"url": replied_to_data["attachment_url"], "mime_type": "image/png"})

            attachments_data = message_data.get("attachments", [])
            if len(reference_image_urls) < 10 and attachments_data:
                for attachment in attachments_data:
                    if attachment.get("url"):
                        reference_image_urls.append({"url": attachment.get("url"), "mime_type": attachment.get("content_type", "image/png")})
                        if len(reference_image_urls) >= 10:
                            break

            grounding_sources = []
            grounding_mode = profile_data.get("grounding_mode", "off")
            if isinstance(grounding_mode, bool):
                grounding_mode = "on" if grounding_mode else "off"

            if grounding_mode in ["on", "on+"]:
                session_key = (channel_id, owner_id, profile_name)
                img_session = self.cog.multi_profile_channels.get(channel_id) or {}
                g_bot_pid = self.cog.profile_manager._get_pid_from_name_any(owner_id, profile_name)
                history_for_grounding = self.cog.session_manager._build_history_for_participant(
                    img_session.get("unified_log", []), g_bot_pid, profile_data
                )

                mapping_key = self.cog.session_manager._get_mapping_key_for_session(session_key, 'multi')
                ch_obj = self.cog.bot.get_channel(channel_id)
                grounding_result = await self.cog.tools_service._get_hybrid_grounding_context(prompt_text, guild_id, history_for_grounding, mapping_key, is_for_image=True, warning_channel=ch_obj)
                if grounding_result:
                    grounding_context, sources, *_ = grounding_result
                    if grounding_context:
                        final_prompt_text = f"{prompt_text}\n\nUse this information to help generate the image:\n{grounding_context}"
                        grounding_sources = sources

            request_data = {
                "is_child_bot": True, "bot_id": bot_id, "author_id": message_data['author_id'],
                "channel_id": channel_id, "guild_id": guild_id, "original_message_id": message_data['id'],
                "original_content": message_data['content'], "prompt_text": final_prompt_text,
                "effective_profile_owner_id": owner_id, "effective_profile_name": profile_name,
                "bot_display_name": bot_display_name, "safety_settings": dynamic_safety_settings,
                "system_instruction": system_instruction, "reference_image_urls": reference_image_urls,
                "placeholder_message": placeholder_message_obj,
                "grounding_sources": grounding_sources, "grounding_mode": grounding_mode,
                "image_generation_model": profile_data.get("image_generation_model", "gemini-2.5-flash-image")
            }

            await self.cog.image_request_queue.put((IMAGE_QUEUE_PRIORITY, time.time(), request_data))
        except Exception as e:
            print(f"[ChildBotManager] Image dispatch error for {bot_id}: {e}")
            traceback.print_exc()

    async def handle_child_bot_toggle(self, event_data: Dict) -> str:
        bot_id = str(event_data.get("bot_id"))
        channel_id = event_data.get("channel_id")
        bot_config = self.cog.child_bots.get(bot_id)
        if not bot_config:
            return "Bot configuration not found."

        session = self.cog.multi_profile_channels.get(channel_id)
        result_msg = ""

        if not session:
            participant = {
                "owner_id": bot_config['owner_id'], "profile_name": bot_config['profile_name'],
                "method": "child_bot", "bot_id": bot_id, "ephemeral": False
            }
            session = {
                "type": "multi", "profiles": [participant],
                "unified_log": [], "is_hydrated": False, "last_bot_message_id": None,
                "owner_id": event_data.get("user_id"), "is_running": False,
                "task_queue": asyncio.Queue(),
                "worker_task": None, "turns_since_last_ltm": 0, "session_prompt": None,
                "session_mode": "sequential", "audio_mode": "off"
            }
            self.cog.multi_profile_channels[channel_id] = session
            result_msg = "Created a new Chat Session with this bot."
        else:
            participant_index = -1
            for i, p in enumerate(session['profiles']):
                if str(p.get('bot_id')) == bot_id:
                    participant_index = i
                    break

            if participant_index != -1:
                session['profiles'].pop(participant_index)
                result_msg = "Removed from the current Chat Session."

                if not session['profiles']:
                    self.cog.multi_profile_channels.pop(channel_id, None)
            else:
                if len(session['profiles']) >= 200:
                    return "The current Chat Session contains the maximum of 200 participating profiles. Please remove a profile and try again."

                participant = {
                    "owner_id": bot_config['owner_id'], "profile_name": bot_config['profile_name'],
                    "method": "child_bot", "bot_id": bot_id, "ephemeral": False
                }
                session['profiles'].append(participant)
                result_msg = "Added to the current Chat Session."

        self.cog.session_manager._save_multi_profile_sessions()
        return result_msg

    async def handle_child_bot_refresh(self, command_data: Dict):
        bot_id = command_data.get("bot_id")
        channel_id = command_data.get("channel_id")
        if not bot_id or not channel_id:
            return

        bot_config = self.cog.child_bots.get(bot_id)
        if not bot_config:
            return

        owner_id = bot_config['owner_id']
        profile_name = bot_config['profile_name']

        worker_key = (channel_id, bot_id)
        if worker_key in self.cog.child_bot_single_sessions:
            worker_data = self.cog.child_bot_single_sessions.pop(worker_key)
            if worker_data and worker_data.get('task'):
                self.cog.session_manager._safe_cancel_task(worker_data['task'])

        session_key = (channel_id, owner_id, profile_name)
        self.cog.channel_models.pop(session_key, None)
        self.cog.channel_model_last_profile_key.pop(session_key, None)
        self.cog.session_last_accessed.pop(session_key, None)
        self.cog.ltm_recall_history.pop(session_key, None)

        ltm_counter_key = (owner_id, profile_name, "guild")
        self.cog.message_counters_for_ltm.pop(ltm_counter_key, None)

    async def handle_child_bot_confirmation(self, event_data: Dict):
        correlation_id = event_data.get("correlation_id")
        if not correlation_id or correlation_id not in self.cog.pending_child_confirmations:
            return

        confirmation_data = self.cog.pending_child_confirmations.pop(correlation_id)
        message_ids = event_data.get("message_ids", [])
        if not message_ids:
            return

        try:
            if confirmation_data.get("type") == "heartbeat_placeholder":
                confirmation_data["message_ids"] = message_ids

            elif confirmation_data.get("type") == "multi_profile":
                channel_id = confirmation_data.get("channel_id")
                turn_id = confirmation_data.get("turn_id")

                if not all([channel_id, turn_id]):
                    return

                session = self.cog.multi_profile_channels.get(channel_id)
                if session:
                    session['last_bot_message_id'] = message_ids[-1]
                    for turn in session.get("unified_log", []):
                        if turn.get("turn_id") == turn_id:
                            turn.setdefault("message_ids", []).extend(message_ids)
                            break

                    session_type = session.get("type", "multi")
                    await self.cog.session_manager._save_session_to_disk((channel_id, None, None), session_type, session.get("unified_log", []))

        except Exception as e:
            print(f"[ChildBotManager] Confirmation processing error ({correlation_id}): {e}")
            traceback.print_exc()

        finally:
            if "event" in confirmation_data and confirmation_data["event"]:
                confirmation_data["event"].set()