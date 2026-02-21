import discord
from discord import app_commands
from discord.ext import commands
import asyncio
import websockets
import orjson as json
import sys
import base64
import io
import aiohttp
from PIL import Image

# Global State
IPC_URI = ""
MAX_AVATAR_SIZE_BYTES = 10 * 1024 * 1024 # 10 MB

def _split_into_sentences_with_abbreviations(text):
    import re
    abbreviations = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'rev.', 'hon.', 'st.', 'sr.', 'jr.', 'capt.', 'sgt.', 'col.', 'gen.',
        'etc.', 'vs.', 'i.e.', 'e.g.', 'cf.', 'et al.', 'viz.',
        'ave.', 'blvd.', 'rd.',
        'a.m.', 'p.m.', 'in.', 'ft.', 'yd.', 'mi.',
        'approx.', 'apt.', 'assn.', 'asst.', 'bldg.', 'co.', 'corp.', 'dept.', 'est.', 'inc.', 'ltd.', 'mfg.', 'vol.'
    }
    potential_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not potential_sentences: return []
    merged_sentences = []
    for s in potential_sentences:
        if not merged_sentences:
            merged_sentences.append(s); continue
        last_sentence = merged_sentences[-1]
        words = last_sentence.split()
        if words and words[-1].lower() in abbreviations:
            merged_sentences[-1] += " " + s
        else:
            merged_sentences.append(s)
    return merged_sentences

class HiveMind:
    def __init__(self):
        self.clients = {} # {bot_id: commands.Bot}
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.ws = None
        self.typing_tasks = {} # { (bot_id, channel_id): asyncio.Task }

    async def execute_typing(self, bot_id, payload):
        bot = self.clients.get(bot_id)
        if not bot or not bot.is_ready(): return
        
        channel_id = payload.get("channel_id")
        task_key = (bot_id, channel_id)
        
        if task_key in self.typing_tasks: return

        async def typing_loop():
            try:
                channel = await bot.fetch_channel(channel_id)
                if not channel: return
                while True:
                    # Triggering typing() context manager or raw call
                    # raw call lasts ~10 seconds. We refresh every 8.
                    await channel.typing()
                    await asyncio.sleep(8)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[Hive] Typing loop error for {bot_id}: {e}")

        self.typing_tasks[task_key] = asyncio.create_task(typing_loop())

    async def launch_bot(self, bot_id, token, parent_id):
        if bot_id in self.clients: return

        # Minimal Intents: Blind & Deaf (No Message Content)
        # We need default intents to handle interactions properly in some cases,
        # but Intents.none() is efficient. Interactions are sent via Gateway but
        # don't require the Message Content intent.
        intents = discord.Intents.none()
        intents.guilds = True # Needed to register commands to guilds
        
        bot = commands.Bot(command_prefix="!", intents=intents)
        self.clients[bot_id] = bot

        # --- Define Commands ---
        @bot.tree.command(name="whoami", description="Displays information about this bot's identity.")
        async def whoami(interaction: discord.Interaction):
            embed = discord.Embed(
                title=f"Bot Identity: {bot.user.name}",
                description="I am a Child Bot managed by MimicAI.",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=bot.user.display_avatar.url)
            embed.add_field(name="Bot ID", value=str(bot.user.id), inline=True)
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @bot.tree.command(name="toggle", description="Toggles this bot's participation in this channel (Admin Only).")
        @app_commands.checks.has_permissions(administrator=True)
        async def toggle(interaction: discord.Interaction):
            if not self.ws:
                await interaction.response.send_message("IPC disconnected.", ephemeral=True); return
            
            # [NEW] Parent Detection Check
            try:
                await interaction.guild.fetch_member(parent_id)
            except discord.NotFound:
                await interaction.response.send_message("You need to invite MimicAI (parent bot) to this server to use child bots.", ephemeral=True)
                return
            except Exception:
                await interaction.response.send_message("An error occurred while verifying parent bot presence.", ephemeral=True)
                return

            toggle_data = {
                "action": "toggle_session_participation",
                "bot_id": bot_id,
                "channel_id": interaction.channel_id,
                "guild_id": interaction.guild_id,
                "user_id": interaction.user.id
            }
            await self.ws.send(json.dumps(toggle_data))
            await interaction.response.send_message("Added to this session.", ephemeral=True)

        async def runner():
            try:
                await bot.login(token)
                # We must use connect() directly or start() to run.
                # Since we have multiple bots, we need to schedule them.
                # bot.start() blocks. We need to create a task.
                # However, bot.login + bot.connect is the breakdown of start.
                # To sync commands, we need to be logged in.
                
                # Background task to sync once ready
                async def on_ready_sync():
                    await bot.wait_until_ready()
                    try:
                        await bot.tree.sync()
                        print(f"[Hive] {bot.user.name} ({bot_id}) synced commands.")
                    except Exception as e:
                        print(f"[Hive] {bot_id} sync failed: {e}")
                
                bot.loop.create_task(on_ready_sync())
                
                await bot.connect()
            except Exception as e:
                print(f"[Hive] Bot {bot_id} failed to start: {e}")
                if bot_id in self.clients: del self.clients[bot_id]

        self.loop.create_task(runner())
        print(f"[Hive] Scheduled launch for bot ID {bot_id}")

    async def shutdown_bot(self, bot_id):
        if bot_id in self.clients:
            print(f"[Hive] Shutting down bot {bot_id}")
            await self.clients[bot_id].close()
            del self.clients[bot_id]

    async def execute_send(self, bot_id, payload):
        # [NEW] Cleanup typing task immediately when sending begins
        channel_id = payload.get("channel_id")
        task_key = (bot_id, channel_id)
        if task_key in self.typing_tasks:
            self.typing_tasks[task_key].cancel()
            del self.typing_tasks[task_key]

        bot = self.clients.get(bot_id)
        if not bot or not bot.is_ready(): return

        channel_id = payload.get("channel_id")
        content = payload.get("content")
        attachment_data = payload.get("attachment")
        correlation_id = payload.get("correlation_id")
        realistic_typing = payload.get("realistic_typing", False)
        reply_to_id = payload.get("reply_to_id")
        ping = payload.get("ping", False)

        try:
            channel = await bot.fetch_channel(channel_id)
            if not channel: return
        except: return

        file_to_send = None
        if attachment_data:
            try:
                image_bytes = base64.b64decode(attachment_data['data_base64'])
                file_to_send = discord.File(io.BytesIO(image_bytes), filename=attachment_data.get('filename', 'image.png'))
            except Exception as e:
                print(f"[Hive] Bot {bot_id} failed to decode attachment: {e}")

        if file_to_send and not realistic_typing:
            if not content.endswith("\n\u200b"): content += "\n\u200b"

        sent_messages = []
        try:
            if realistic_typing:
                sentences = _split_into_sentences_with_abbreviations(content)
                async with channel.typing():
                    for i, sentence in enumerate(sentences):
                        if not sentence.strip(): continue
                        delay = max(1.0, min(len(sentence) / 20, 4.0))
                        await asyncio.sleep(delay)
                        
                        kwargs = {"content": sentence}
                        if i == 0:
                            if file_to_send: kwargs["file"] = file_to_send
                            if reply_to_id:
                                try:
                                    ref_msg = await channel.fetch_message(reply_to_id)
                                    m = await ref_msg.reply(mention_author=ping, **kwargs)
                                except: m = await channel.send(**kwargs)
                            else: m = await channel.send(**kwargs)
                        else: m = await channel.send(**kwargs)
                        sent_messages.append(m)
            else:
                try: await channel.typing()
                except: pass
                
                # [NEW] Chunking logic for non-realistic typing
                remaining_content = content
                first_chunk = True
                while remaining_content:
                    chunk = ""
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
                    # Only apply file/reply to the very first chunk
                    if first_chunk:
                        if file_to_send: kwargs["file"] = file_to_send
                        if reply_to_id:
                            try:
                                ref_msg = await channel.fetch_message(reply_to_id)
                                m = await ref_msg.reply(mention_author=ping, **kwargs)
                            except: m = await channel.send(**kwargs)
                        else: m = await channel.send(**kwargs)
                        first_chunk = False
                    else:
                        m = await channel.send(**kwargs)
                    sent_messages.append(m)

            if correlation_id:
                confirmation = {
                    "event_type": "message_sent_confirmation",
                    "bot_id": bot_id,
                    "correlation_id": correlation_id,
                    "message_ids": [m.id for m in sent_messages],
                    "channel_id": channel.id
                }
                if self.ws: await self.ws.send(json.dumps(confirmation))

        except Exception as e:
            print(f"[Hive] Error sending message for {bot_id}: {e}")
        
        finally:
            # Cleanup resources
            if file_to_send:
                file_to_send.close()
                del file_to_send
            if 'image_bytes' in locals():
                del image_bytes
            # Payload can be large due to base64
            del payload

    async def execute_typing(self, bot_id, payload):
        bot = self.clients.get(bot_id)
        if not bot or not bot.is_ready(): return
        
        channel_id = payload.get("channel_id")
        task_key = (bot_id, channel_id)
        
        if task_key in self.typing_tasks: return

        async def typing_loop():
            try:
                channel = await bot.fetch_channel(channel_id)
                if not channel: return
                
                start_time = asyncio.get_running_loop().time()
                while True:
                    # Check for 60s timeout
                    if asyncio.get_running_loop().time() - start_time > 60:
                        break
                        
                    await channel.typing()
                    await asyncio.sleep(7)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[Hive] Typing loop error for {bot_id}: {e}")
            finally:
                # Ensure key is removed so typing can be restarted if needed
                if self.typing_tasks.get(task_key) == asyncio.current_task():
                    self.typing_tasks.pop(task_key, None)

        self.typing_tasks[task_key] = asyncio.create_task(typing_loop())

    async def update_appearance(self, bot_id, payload):
        bot = self.clients.get(bot_id)
        if not bot or not bot.is_ready(): return
        
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
                                        # [FIXED] Force image to RGBA and strict PNG format to preserve transparency
                                        with Image.open(io.BytesIO(data)) as img:
                                            if img.mode != 'RGBA':
                                                img = img.convert('RGBA')
                                            with io.BytesIO() as out_buffer:
                                                img.save(out_buffer, format='PNG')
                                                png_data = out_buffer.getvalue()
                                        await bot.user.edit(avatar=png_data)
                                    except Exception as img_e:
                                        print(f"[Hive] Failed to process avatar image transparency: {img_e}")
                                        # Fallback to raw data if processing fails
                                        await bot.user.edit(avatar=data)
                else:
                    await bot.user.edit(avatar=None)
        except Exception as e:
            print(f"[Hive] Appearance update error for {bot_id}: {e}")

    async def execute_regenerate(self, bot_id, payload):
        bot = self.clients.get(bot_id)
        if not bot or not bot.is_ready(): return
        
        channel_id = payload.get("channel_id")
        message_id = payload.get("message_id")
        content = payload.get("content")
        
        try:
            channel = await bot.fetch_channel(channel_id)
            if not channel: return
            message = await channel.fetch_message(message_id)
            if not message: return
            
            # Filter attachments: Keep images only
            kept_attachments = [a for a in message.attachments if a.content_type and a.content_type.startswith("image/")]
            
            # Note: Discord requires the original Attachment objects to be passed back to keep them
            await message.edit(content=content, attachments=kept_attachments)
        except Exception as e:
            print(f"[Hive] Regeneration edit error for {bot_id}: {e}")

    async def connect(self):
        while True:
            try:
                async with websockets.connect(IPC_URI, max_size=2**24) as ws:
                    self.ws = ws
                    print("[Hive] Connected to Manager.")
                    await ws.send(json.dumps({"action": "identify_hive"}))

                    async for message in ws:
                        data = json.loads(message)
                        action = data.get("action")
                        
                        # Manager Commands
                        if action == "launch":
                            await self.launch_bot(str(data.get("bot_id")), data.get("token"), data.get("parent_id"))
                        elif action == "shutdown":
                            await self.shutdown_bot(str(data.get("bot_id")))
                        
                        # Child Commands (Routed)
                        elif action == "send_to_child":
                            target_id = str(data.get("bot_id"))
                            child_payload = data.get("payload", {})
                            child_action = child_payload.get("action")
                            
                            if child_action == "send_message":
                                asyncio.create_task(self.execute_send(target_id, child_payload))
                            elif child_action == "regenerate_message":
                                asyncio.create_task(self.execute_regenerate(target_id, child_payload))
                            elif child_action == "start_typing":
                                asyncio.create_task(self.execute_typing(target_id, child_payload))
                            elif child_action == "stop_typing":
                                ch_id = child_payload.get("channel_id")
                                t_key = (target_id, ch_id)
                                if t_key in self.typing_tasks:
                                    self.typing_tasks[t_key].cancel()
                                    del self.typing_tasks[t_key]
                            elif child_action in ["update_username", "update_avatar"]:
                                asyncio.create_task(self.update_appearance(target_id, child_payload))

            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
                print("[Hive] Connection lost. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"[Hive] Error: {e}")
                await asyncio.sleep(5)

    def run(self):
        self.loop.run_until_complete(self.connect())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: child_bot.py <IPC_URI>")
        sys.exit(1)
    
    IPC_URI = sys.argv[1]
    hive = HiveMind()
    hive.run()