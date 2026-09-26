import uuid
import discord
import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from ...utils.helpers import (
    _format_history_entry, _get_user_hash, attachment_mime, attachment_tag,
    image_command_prefix, image_command_prompt, is_media_attachment,
)
from ...utils.attachment_limits import over_attachment_limit, skipped_attachment_note
from ...utils.http_client import get_shared_client
from ...managers.session_manager import intern_turn, log_user_turn, reply_quote
from ...utils.constants import (
    defaultConfig, ROUND_MEDIA_MAX, ROUND_MEDIA_SKIPPED_NOTE,
)
from ...utils.discord_cdn import unsigned_attachment_url


def reply_snapshot(message: Any) -> Dict[str, Any]:
    """The message a reply points at, as plain data: what a child bot's payload carries,
    and what `reply_record` reads, whichever way the reply arrived."""
    return {
        "id": message.id,
        "author_id": message.author.id,
        "author_name": message.author.display_name,
        "author_is_bot": bool(getattr(message.author, "bot", False)),
        "content": message.clean_content,
        "created_at": message.created_at.isoformat(),
        "attachments": [{"url": a.url, "filename": a.filename, "content_type": a.content_type,
                         "size": a.size} for a in message.attachments],
    }


async def referenced_message(message: Any) -> Optional[Dict[str, Any]]:
    """What `message` replies to, None when it is not a reply, and `{"missing": True}`
    when what it replied to cannot be loaded.

    The gateway delivers the replied-to message with the reply, and the client caches
    recent ones: a REST fetch is the last resort, not the first. It used to be the only
    one -- a round trip on the round's critical path for every reply, one after another.
    """
    reference = getattr(message, "reference", None)
    # A forward carries a reference too, to a message usually in another channel: it is
    # not a reply, and fetching it here would report it missing.
    if (not reference or not reference.message_id
            or getattr(reference, "type", None) == discord.MessageReferenceType.forward):
        return None
    target = reference.resolved or reference.cached_message
    if target is None:
        try:
            target = await message.channel.fetch_message(reference.message_id)
        except Exception:
            target = None
    if target is None or isinstance(target, discord.DeletedReferencedMessage):
        return {"id": reference.message_id, "missing": True}
    return reply_snapshot(target)


def reply_record(unified_log: List[Dict[str, Any]], ref: Optional[Dict[str, Any]]
                 ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """What a turn stores about the message it replies to, and that message's media to
    send with this round.

    A reply to a turn in the log stores its `turn_id` and nothing of its text: the
    history builder names it for a reader who has it in view and quotes it, from the
    log, for one who has not (`render_reply`). Anything else -- chat from before the
    session, another bot -- is a snapshot: who, when, and the start of what it said.

    The media goes out with this round because the log keeps none: a reply to a picture
    asks about the picture. Every file, not the first, and tagged in the reply so the
    character knows which message it came from.
    """
    if not ref:
        return None, []
    if ref.get("missing"):
        return {"missing": True}, []
    reply_media = []
    files = []
    for attachment in ref.get("attachments") or ():
        if not is_media_attachment(attachment):
            continue
        if over_attachment_limit(attachment):
            files.append(skipped_attachment_note(attachment))
            continue
        reply_media.append({"url": attachment["url"], "mime_type": attachment_mime(attachment)})
        files.append(attachment_tag(attachment))
    record: Dict[str, Any] = {"files": files} if files else {}
    target = next((t for t in reversed(unified_log) if ref["id"] in (t.get("message_ids") or ())), None)
    if target is not None and not target.get("type") and target.get("turn_id"):
        record["turn_id"] = target["turn_id"]
    else:
        # A person as their turns name them; a bot has no id a header would carry.
        record["speaker"] = (ref["author_name"] if ref.get("author_is_bot")
                             else f"{ref['author_name']} [ID: {_get_user_hash(ref['author_id'])}]")
        record["at"] = ref["created_at"]
        record["quote"] = reply_quote(ref.get("content") or "")
    return record, reply_media


class UserTurn(NamedTuple):
    """What one user message contributes to a round.

    `text_attachments` is already inside `content` and is handed back separately for the
    one caller that needs it on its own: an `!image` prompt, which is built from the
    message before the turn is, and which reads a .txt because Discord caps a message at
    2,000 characters. Returned rather than re-read, so the file is downloaded once.
    """
    content: str
    media_parts: List[Dict[str, str]]
    text_attachments: str


class TriggerIntakeMixin:
    """Turns the raw triggers batched for one multi-profile round -- messages,
    reactions, replies, proactive kicks -- into the round's user-side history.
    """

    async def _compose_user_turn(self, typed: str, attachments: Sequence[Any], *,
                                 edited: bool = False) -> UserTurn:
        """A user message's turn content, and the media parts to send with it.

        The one builder for what a user's message says in `unified_log`: a new message, one
        batched mid-round and an edited one all come through here. Each used to build it by
        hand, and they drifted -- an edit rebuilt the turn from the message text alone, which
        dropped any text file the message carried from the log, and so from every later
        turn and regeneration; a batched message never read its text files at all.

        `attachments` are discord.Attachment objects, or the dicts a child bot's payload
        carries. `typed` is only what the person wrote: URL Context reads links from that
        and nothing folded in here. What it replies to is not folded in either: it is the
        turn's `reply_to`, rendered per reader (`reply_record`).
        """
        content = f"{typed}\n(edited)" if edited else typed

        # Shared client: _process_text_attachments sets its own
        # per-request timeout, so nothing is lost by not owning one.
        text_att_content = await self.cog.media_service._process_text_attachments(
            attachments, get_shared_client())
        if text_att_content:
            content = f"{content}\n\n{text_att_content}"

        media_parts = []
        att_tags = []
        for attachment in attachments:
            if not is_media_attachment(attachment):
                continue
            if over_attachment_limit(attachment):
                att_tags.append(skipped_attachment_note(attachment))
                continue
            url = attachment.get('url') if isinstance(attachment, dict) else attachment.url
            # The bare type, without Discord's `; charset=` parameters: it goes on the wire
            # as a part's mimeType, and every provider matches it against a fixed table.
            media_parts.append({"url": url, "mime_type": attachment_mime(attachment)})
            att_tags.append(attachment_tag(attachment))

        if att_tags:
            content = f"{' '.join(att_tags)}\n{content}".strip()
        return UserTurn(content, media_parts, text_att_content)

    @staticmethod
    def _extend_image_prompt(typed: str, text_attachments: str) -> str:
        """An `!image` prompt with the text file attached to it folded in.

        Discord caps a message at 2,000 characters for anyone without Nitro, which is not
        much of a scene description, so a .txt stands in for the rest. It *appends* rather
        than replaces: `!image in watercolour` plus a file of scene notes has to keep the
        steer as well as the detail, and silently dropping what someone typed is the
        worse failure of the two.

        Capped at `LIMIT_IMAGE_PROMPT_CHARS`. The file itself is read to 40,000
        characters, which is sized for a character to *read*; an image model attends to a
        fraction of that and bills for all of it.
        """
        if not text_attachments:
            return typed
        joined = "\n\n".join(part for part in (typed, text_attachments) if part)
        return joined[:defaultConfig.LIMIT_IMAGE_PROMPT_CHARS]

    @staticmethod
    def _round_media(new_round_turn_data, links: bool = True) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """The round's attachments to send with a character's turn, and a note for any left out.

        Every attachment in the round used to go to every character: up to ten a message,
        across however many messages the round took in. The newest ROUND_MEDIA_MAX go.
        The rest are counted in the note, because their messages still say something was
        attached.

        `links` False leaves out images fetched off a posted link -- a profile with URL
        Context off sees nothing a link brought in, as grounding off sees no search.
        """
        # Once each: a reply to a picture posted in the same round carries it again.
        media, seen = [], set()
        for _text, _url, turn_media in new_round_turn_data:
            for part in turn_media:
                if not links and part.get("from_link"):
                    continue
                key = unsigned_attachment_url(part.get("url"))
                if key and key in seen:
                    continue
                seen.add(key)
                media.append(part)
        if len(media) <= ROUND_MEDIA_MAX:
            return media, None
        note = ROUND_MEDIA_SKIPPED_NOTE.format(limit=ROUND_MEDIA_MAX, count=len(media) - ROUND_MEDIA_MAX)
        return media[-ROUND_MEDIA_MAX:], note

    async def _collect_round_triggers(
        self, session, session_type, channel_id, all_triggers_for_round,
        new_round_turn_data, pending_url_fetches, recent_processed_ids,
        is_image_gen_round, image_gen_prompt, starting_profile_override,
        round_author_name, triggering_user_id, batch_start_index,
    ):
        """Normalises the round's batched triggers into history turns.

        Appends to new_round_turn_data, pending_url_fetches and recent_processed_ids in
        place, and returns the six round-scoped values a trigger can change:
        (is_image_gen_round, image_gen_prompt, starting_profile_override,
        round_author_name, triggering_user_id, batch_start_index). They are passed in as
        well as returned so a round whose triggers set none of them keeps the caller's
        values.

        The long parameter list is the real coupling this step has to the round, not a
        shape worth hiding: it was previously all ambient locals in the worker frame.
        """
        for i, trigger in enumerate(all_triggers_for_round):
            if not trigger:
                continue

            message_trigger, reaction_trigger, message_payload = None, None, None

            # A game beat is neither a message nor a reaction: nobody said it, the table
            # did. It contributes the round's system note and, at index 0, the seat that
            # has to answer it -- and it is handled ahead of the index-0 branch below
            # because the batch branch would otherwise read its payload dict as a child
            # bot's message and go looking for an author.
            #
            # Deliberately not appended to `unified_log`. The mechanical record already
            # lives in `<game_context>`; a log carrying bracketed stage directions would
            # have every later round reading them back as things that were said.
            if isinstance(trigger, tuple) and trigger[0] == 'game_beat':
                payload = trigger[1]
                if i == 0:
                    starting_profile_override = trigger[2]
                content = payload.get('content')
                if content:
                    new_round_turn_data.append((content, None, []))
                continue

            if i == 0:
                if isinstance(trigger, tuple):
                    if trigger[0] == 'reply':
                        _, message_trigger, starting_profile_override = trigger
                    elif trigger[0] == 'reaction': 
                        _, reaction_trigger, starting_profile_override = trigger
                        try:
                            ch = self.cog.bot.get_channel(reaction_trigger.channel_id)
                            msg_obj = await ch.fetch_message(reaction_trigger.message_id)
                            await msg_obj.clear_reaction(reaction_trigger.emoji)
                        except: pass
                    elif trigger[0] == 'reaction_single': 
                        _, reaction_trigger, starting_profile_override = trigger
                        try:
                            ch = self.cog.bot.get_channel(reaction_trigger.channel_id)
                            msg_obj = await ch.fetch_message(reaction_trigger.message_id)
                            await msg_obj.clear_reaction(reaction_trigger.emoji)
                        except: pass
                    elif trigger[0] == 'child_mention': _, message_payload, starting_profile_override = trigger
                elif isinstance(trigger, discord.RawReactionActionEvent): reaction_trigger = trigger
                elif isinstance(trigger, str):
                    # Handle string prompts
                    content = trigger
                    turn_id = str(uuid.uuid4())

                    # XML-standardised system turn without Name/Timestamp header
                    system_content = f"<system_note>\n{content}\n</system_note>"

                    turn_object = {
                        "turn_id": turn_id,
                        "is_user": False,
                        "speaker_pid": "SYSTEM",
                        "message_ids": [],
                        "content": system_content,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    session.setdefault("unified_log", []).append(intern_turn(turn_object))

                    new_round_turn_data.append((system_content, None, []))

                    message_trigger = None
                else: message_trigger = trigger
            else:
                # [UPDATED] Unpack structured tuples in batches to prevent lost replies/mentions/child bots
                if isinstance(trigger, discord.Message):
                    message_trigger = trigger
                elif isinstance(trigger, tuple) and len(trigger) > 1:
                    if isinstance(trigger[1], discord.Message):
                        message_trigger = trigger[1]
                    elif isinstance(trigger[1], dict):
                        message_payload = trigger[1]

            # --- Deduplication Check ---
            check_id = None
            if message_trigger: check_id = message_trigger.id
            elif message_payload: check_id = message_payload.get('id')

            if check_id:
                if check_id in recent_processed_ids:
                    continue # Skip duplicate trigger
                recent_processed_ids.append(check_id)
            # ---------------------------

            # Whether this trigger is the one that opened the image round, so the text file
            # it carried can be folded into the prompt once `_compose_user_turn` below has
            # read it. Reset per trigger: only the first `!image` in a batch starts a round.
            image_prompt_from_this_trigger = False

            if (message_trigger or message_payload) and not is_image_gen_round:
                trigger_content = message_payload['content'] if message_payload else message_trigger.clean_content

                # Detection is prefix-based only. The round opens on the prefix rather
                # than on the prompt because a prompt this message does not carry may
                # still arrive in a text file attached to it, folded in below -- which is
                # also where a round that turns out to have nothing to draw is closed.
                if image_command_prefix(trigger_content):
                    is_image_gen_round = True
                    image_gen_prompt = image_command_prompt(trigger_content) or ""
                    image_prompt_from_this_trigger = True

            if message_trigger or message_payload:
                is_child_mention = message_payload is not None
                trigger_obj = message_payload if is_child_mention else message_trigger

                triggering_user_id = trigger_obj['author_id'] if is_child_mention else trigger_obj.author.id
                author_name = trigger_obj['author_name'] if is_child_mention else trigger_obj.author.display_name
                if round_author_name == "A user": round_author_name = author_name

                # A child bot's payload carries the snapshot its listener took.
                reply_ref = (trigger_obj.get('replied_to') if is_child_mention
                             else await referenced_message(message_trigger) if message_trigger else None)
                reply_to, reply_media = reply_record(session.get("unified_log", []), reply_ref)

                # What the person typed, before text files and the quoted reply are folded
                # in. URL Context reads links from this alone: a link inside an attached file
                # or someone else's quoted message is not one they asked the profile to
                # open, and the quote is cut at 150 characters, so its link can be half a URL.
                typed_text = trigger_obj['content'] if is_child_mention else trigger_obj.clean_content

                raw_att_list = trigger_obj['attachments'] if is_child_mention else trigger_obj.attachments
                user_turn = await self._compose_user_turn(typed_text, raw_att_list)
                content, own_media_parts = user_turn.content, user_turn.media_parts

                if image_prompt_from_this_trigger:
                    image_gen_prompt = self._extend_image_prompt(
                        image_gen_prompt, user_turn.text_attachments)
                    if not image_gen_prompt:
                        # `!image` with nothing after it and no text file to fold in. An
                        # empty prompt is a schema error on OpenRouter's Image API and
                        # then the same empty prompt again on the fallback model, so no
                        # image round opens and the message is an ordinary turn.
                        is_image_gen_round = False
                        image_prompt_from_this_trigger = False

                # [NEW] URL Context Logic: Enforce Profile Setting & Separation
                any_url_enabled = False
                any_url_rag = False
                for p in session['profiles']:
                    p_index = self.cog.profile_manager._get_user_index(p['owner_id'])
                    p_is_b = p['profile_name'] in p_index.get("borrowed", [])
                    p_settings = self.cog.profile_manager._get_profile_config(p['owner_id'], p['profile_name'], p_is_b) or {}

                    u_mode = p_settings.get("url_mode", "off")
                    if "url_mode" not in p_settings:
                        u_mode = "rag" if p_settings.get("url_fetching_enabled", False) else "off"

                    if u_mode != "off":
                        any_url_enabled = True
                    if u_mode == "rag":
                        any_url_rag = True

                url_text_content = None
                trigger_media_parts = []

                if any_url_enabled and any_url_rag:
                    # Defer URL fetching until after placeholder is sent
                    pending_url_fetches.append({
                        "content": typed_text,
                        "guild_id": trigger_obj['guild_id'] if is_child_mention else trigger_obj.guild.id,
                        "turn_data_index": len(new_round_turn_data)
                    })

                # The user's own clock, from `/settings` -> About Me. This used to
                # resolve the user's active profile in the channel and read the
                # timezone off that character's config -- so your timestamps followed
                # whoever you last activated here, and a user with no active profile
                # silently got UTC. _get_profile_config caches nothing, so it also
                # cost a decrypt per message to read one string.
                author_tz = self.cog.profile_manager.user_timezone(triggering_user_id)
                user_hash = _get_user_hash(triggering_user_id)

                created_at = datetime.datetime.now(datetime.timezone.utc) if is_child_mention else trigger_obj.created_at
                user_line = _format_history_entry(author_name, created_at, content, author_tz, entity_id=user_hash)

                turn_id = str(uuid.uuid4())
                trigger_id = trigger_obj['id'] if is_child_mention else trigger_obj.id

                turn_object = {
                    "turn_id": turn_id, 
                    "is_user": True,
                    "speaker_pid": str(triggering_user_id),
                    "message_ids": [trigger_id],
                    "content": user_line
                }
                if reply_to:
                    turn_object["reply_to"] = reply_to
                # Where the channel shows it, which for a message sent while the previous
                # round's last character was still generating is above that reply, not
                # below it. The reserve has to follow it back, or the round's own user
                # message is the one turn the STM window may drop.
                insert_index = log_user_turn(session, turn_object, created_at)
                batch_start_index = min(batch_start_index, insert_index)

                if pending_url_fetches and pending_url_fetches[-1]["turn_data_index"] == len(new_round_turn_data):
                    pending_url_fetches[-1]["turn_object"] = turn_object

                # [NEW] Immediate persistence for user turns
                # Appends only, and the round-end flush is moments away -- see
                # SessionManager.mark_session_dirty.
                self.cog.session_manager.mark_session_dirty((channel_id, None, None), session_type)

                # The replied-to message's media first, then the message's own: the order
                # they had when the reply scanner fetched the first of them here.
                trigger_media_parts.extend(reply_media + own_media_parts)

                # Store raw components for gating logic
                new_round_turn_data.append((user_line, url_text_content, trigger_media_parts))

            elif reaction_trigger and i == 0:
                triggering_user_id = reaction_trigger.user_id
                user_obj = self.cog.bot.get_user(triggering_user_id)
                if user_obj:
                    round_author_name = user_obj.display_name

        return (is_image_gen_round, image_gen_prompt, starting_profile_override,
                round_author_name, triggering_user_id, batch_start_index)
