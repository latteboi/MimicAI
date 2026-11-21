"""The profile dashboard embed."""

import datetime
import asyncio
import discord
from typing import Dict, Set, Any, Optional, Tuple
from ..utils.user_defaults import (
    MODEL_SLOT_PAIRS, chain_lines, model_chain, model_provider, short_model_name, subscription_chain)
from ..utils.constants import (
    defaultConfig, PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME, TTS_VOICE_GENDER, TTS_VOICE_CHARACTER, CONTENT_RATING_LABELS, CHANNEL_ACCESS_LABELS, CONTENT_RATING_REASON_LABELS, CONTENT_RATING_REASON_FALLBACK, CONTENT_RATING_PENDING, CONTENT_RATING_ADULT, CONTENT_RATING_EMOJI, DEFAULT_IMAGE_MODEL, DEFAULT_SPEECH_VOICE, SPEECH_LANGUAGE_NAMES, IMAGE_GROUNDING_LABELS)
from ..utils.helpers import (
    auto_tier, image_rag_enabled, ltm_auto_recall_enabled, always_reads_synopsis, grounding_mode_display, resolve_image_output_params, resolve_auto_models, neuro_levels, resolve_image_tools, resolve_unreadable_media_mode, resolve_url_mode, _get_user_hash)
from ..utils.bonds import describe as describe_bond

def feature_fields(config: Dict[str, Any], speech_providers: Set[str]) -> Tuple[str, str, str]:
    """The dashboard's Tools, Behaviour and Media fields: one setting to a line.

    They were three inline columns under names that did not fit them -- Timezone and
    Placeholder under Tools -- so they are regrouped: what the character can reach for,
    how it conducts itself, and what it draws and says aloud. A feature that is on gets
    subtext lines of its detail.
    """
    on, off = "**`ON`**", "`OFF`"
    flag = lambda value: on if value else off
    url = {"native": "**`NATIVE`**", "rag": "**`RAG`**"}.get(resolve_url_mode(config), off)
    # Resolved, not read: the stored value can be anything an import wrote, and this has
    # to say what the profile will actually do with an attachment.
    unreadable = "**`SIMULATED`**" if resolve_unreadable_media_mode(config) == "simulated" else off
    response = str(config.get("response_mode", "regular")).replace("_", " ").title()

    image = speech = neuro = ()
    if config.get("image_generation_enabled"):
        # Resolved rather than read, so this reports what the request will carry: a ratio
        # the current image model does not accept is stored but not sent.
        model = config.get("image_generation_model") or DEFAULT_IMAGE_MODEL
        output = resolve_image_output_params(config, model)
        grounded = resolve_image_tools(config, model) or image_rag_enabled(config)
        image = ((output.get("aspect_ratio"), output.get("image_size"), output.get("quality"),
                  output.get("thinking_level"),
                  grounded and IMAGE_GROUNDING_LABELS.get(config.get("image_grounding_mode"))),)
    if config.get("speech_tts_enabled"):
        # Gender and character tell thirty star names apart; an unknown voice stays bare.
        # Temperature and language reach Gemini voices only, speed OpenRouter's only, so
        # each shows only when the speech chain can reach a model that takes it.
        voice = config.get("speech_voice", DEFAULT_SPEECH_VOICE)
        described = " \u00b7 ".join(d for d in (TTS_VOICE_GENDER.get(voice),
                                                TTS_VOICE_CHARACTER.get(voice)) if d)
        gemini, openrouter = "gemini" in speech_providers, "openrouter" in speech_providers
        language, speed = config.get("speech_language") or "", config.get("speech_speed")
        speech = ((f"{voice} ({described})" if described else voice,
                   gemini and f"Temperature {config.get('speech_temperature', 1.0)}",
                   gemini and SPEECH_LANGUAGE_NAMES.get(language, language),
                   openrouter and speed is not None and f"Speed {speed}\u00d7"),)
    if config.get("neuro_engine_enabled"):
        # Two to a line: a half-width column wraps all four.
        state = neuro_levels(config)
        neuro = tuple(tuple(f"{axis.title()} {state[axis]}" for axis in pair)
                      for pair in (("dopamine", "cortisol"), ("oxytocin", "adrenaline")))

    def feature(name, key, rows):
        details = (" \u00b7 ".join(str(p) for p in row if p) for row in rows)
        return "\n".join([f"{name}: {flag(config.get(key))}", *(f"-# {d}" for d in details if d)])

    tools = [f"Grounding: {grounding_mode_display(config)}",
             f"URL Context: {url}",
             f"Unreadable Media: {unreadable}",
             f"Help Mode: {flag(config.get('help_mode_enabled'))}"]
    behaviour = [f"Response Mode: `{response}`",
                 f"Realistic Typing: {flag(config.get('realistic_typing_enabled'))}",
                 f"Placeholder: {config.get('placeholder_emoji') or 'Default'}",
                 f"Timezone: `{config.get('timezone', 'UTC')}`",
                 feature("Neuro Engine", "neuro_engine_enabled", neuro),
                 f"Bonds: {flag(config.get('bond_engine_enabled'))}"]
    media = [feature("Image Gen", "image_generation_enabled", image),
             feature("Speech TTS", "speech_tts_enabled", speech)]
    return "\n".join(tools), "\n".join(behaviour), "\n".join(media)


class ProfileDashboard:
    """The profile dashboard embed. Reached as `cog.profile_dashboard`."""

    def __init__(self, cog):
        self.cog = cog

    async def _build_profile_embed(self, user_id: int, profile_name: str, channel_id: int,
                                   viewer_id: Optional[int] = None,
                                   all_models: bool = False) -> discord.Embed:
        """`viewer_id`, for a card shown to someone other than its owner, adds how the
        character feels about them, where Bonds is on. `all_models` lists every model
        category, where the Response chain alone is the default.

        Every profile and memory read in one worker thread: a cache miss is a decrypt, and
        both dashboards repaint on every edit. Discord's own caches are read on the loop."""
        # This server's memories, the count its limit applies to; every server's in a DM.
        guild = getattr(self.cog.bot.get_channel(channel_id), "guild", None)
        guild_id = guild.id if guild else None

        def _read():
            is_borrowed = profile_name in self.cog.profile_manager._get_user_index(user_id).get("borrowed", {})
            eff_owner, eff_name = self.cog.profile_manager._resolve_effective_profile(user_id, profile_name)
            config = self.cog.profile_manager._get_profile_config(user_id, profile_name, is_borrowed) or {}
            mm = self.cog.memory_manager
            training = mm._load_training_shard(str(eff_owner), eff_name)
            held = (self.cog.persona_content.bonds(user_id, profile_name)
                    if viewer_id is not None and config.get("bond_engine_enabled") else None)
            if is_borrowed:
                ids = (config.get("original_pid", "Unknown"), self.cog.profile_manager._get_pid_from_name_any(user_id, profile_name))
            else:
                ids = (self.cog.profile_manager._get_profile_id(eff_owner, eff_name), None)
            return (is_borrowed, eff_owner, eff_name, config,
                    mm._ltm_count(user_id, profile_name, guild_id), len(training) if training else 0,
                    held, self.cog.profile_manager._get_user_appearance(eff_owner, eff_name),
                    self.cog.content_rating._resolve_enforced_safety_level(user_id, profile_name),
                    self.cog.content_rating._content_rating_state(user_id, profile_name),
                    self.cog.content_rating.is_held(user_id, profile_name),
                    self.cog.library.distribution_of(user_id, profile_name), ids,
                    self.cog.user_settings.provider_preference(user_id),
                    self.cog.storage_manager.subscription_payer(guild_id, user_id))
        (is_borrowed, effective_owner_id, effective_profile_name, config, ltm_count, train_count,
         held, appearance_data, enforced, (verdict, verdict_reason), awaiting_rating,
         (is_public, borrowers), (pid_a, pid_b), preferred, paid) = await asyncio.to_thread(_read)

        embed = discord.Embed(title=f"Profile Dashboard: '{profile_name}'", color=discord.Color.blue())
        profile_type = "Personal"

        if is_borrowed:
            owner_user = self.cog.bot.get_user(effective_owner_id)
            profile_type = f"Borrowed (from {owner_user.name if owner_user else 'Unknown User'})"

        # Off this config, as `_get_user_profile_for_model` reads them -- which would read
        # the config and the prompts again, on a worker thread, for these alone.
        temp = float(config.get("temperature", defaultConfig.GEMINI_TEMPERATURE))
        top_p = float(config.get("top_p", defaultConfig.GEMINI_TOP_P))
        top_k = int(config.get("top_k", defaultConfig.GEMINI_TOP_K))
        train_ctx = int(config.get("training_context_size", defaultConfig.TRAINING_CONTEXT_SIZE))
        train_rel = float(config.get("training_relevance_threshold", defaultConfig.TRAINING_RELEVANCE_THRESHOLD))
        prim_model = config.get("primary_model", PRIMARY_MODEL_NAME)
        fall_model = config.get("fallback_model", FALLBACK_MODEL_NAME)

        created_str = config.get('created_at')
        created_display = "Unknown"
        if created_str:
            try:
                dt = datetime.datetime.fromisoformat(created_str)
                ts = int(dt.timestamp())
                created_display = f"<t:{ts}:D>"
            except: pass

        display_name = appearance_data.get("custom_display_name") or effective_profile_name

        embed.add_field(name="Profile Type", value=f"`{profile_type}`", inline=True)
        embed.add_field(name="Created", value=created_display, inline=True)
        embed.add_field(name="Display Name", value=f"`{display_name}`", inline=True)
        if held is not None:
            embed.add_field(name=f"How {display_name} feels about you"[:256],
                            value=describe_bond(held.get(_get_user_hash(viewer_id))), inline=False)

        rating_display = ("⏳ `Held for rating`" if awaiting_rating else
                          f"{CONTENT_RATING_EMOJI[verdict]} `{CONTENT_RATING_LABELS[verdict]}`")

        # Discord packs inline fields three to a row, so these are emitted as two
        # deliberate rows rather than in the order they were computed: identity in
        # the left column, content state in the middle, consequence on the right.
        # Content Rating and Public Library sharing the middle column is the point --
        # read downwards, they are the two fields that say where the profile may go.
        # The zero-width fields are the padding that holds the grid square.
        if is_borrowed:
            left_column = [("Profile ID (Source)", f"`{pid_a}`"),
                           ("Profile ID (Local)", f"`{pid_b}`")]
        else:
            profile_id = pid_a
            left_column = [("Profile ID (PID)", f"`{profile_id}`"),
                           ("\u200b", "\u200b")]

            if profile_id.startswith("X"):
                embed.description = f"\u26a0\ufe0f **System Profile.** Global settings managed by Bot Admin.\n\n" + (embed.description or "")
                profile_type = "System"

        embed.add_field(name=left_column[0][0], value=left_column[0][1], inline=True)
        embed.add_field(name="Content Rating", value=rating_display, inline=True)
        embed.add_field(name="Channel Access",
                        value=f"`{CHANNEL_ACCESS_LABELS[enforced]}`", inline=True)

        embed.add_field(name=left_column[1][0], value=left_column[1][1], inline=True)
        library_value = "\U0001f310 `Published`" if is_public else "`Not published`"
        # The owner's reach; on a borrow the count would be the source's.
        if borrowers and not is_borrowed:
            library_value += f"\n{borrowers} borrowing"
        embed.add_field(name="Public Library", value=library_value, inline=True)
        embed.add_field(name="\u200b", value="\u200b", inline=True)

        if verdict == CONTENT_RATING_PENDING and verdict_reason and not awaiting_rating:
            # Surface why a profile is stuck on Pending. A silent Pending was
            # indistinguishable from one still in flight. This reason describes an
            # operational failure rather than the persona, so it is shown as stored.
            embed.description = ((embed.description or "") +
                                 f"\n\u23f3 **Classification unavailable:** {verdict_reason}").strip()
        if awaiting_rating:
            embed.description = ((embed.description or "") +
                                 "\n\u23f3 **Held for its content rating.** It runs nowhere until the "
                                 "rating is in; Content Safety says where that stands.").strip()
        elif verdict == CONTENT_RATING_ADULT:
            # A category, never the classifier's own words about the persona.
            embed.description = ((embed.description or "") +
                                 "\n\U0001f6ab **Not Allowed: "
                                 + CONTENT_RATING_REASON_LABELS.get(
                                     verdict_reason, CONTENT_RATING_REASON_FALLBACK) +
                                 ".** MimicAI does not host this, so the profile runs nowhere. "
                                 "To have it rated again, edit the persona and submit it "
                                 "from Content Safety.").strip()

        # No stale-rating check here: this renders on every dashboard repaint, including
        # the one straight after an edit, where it raced the post-edit prompt and
        # decided before the owner could. `_open_profile_manage` runs it on open.

        embed.add_field(name="\u200b", value="\u200b", inline=False)

        # Each category's whole chain, read from `model_chain`, the resolver its generation
        # path calls, so this cannot name a model it skips or leave out one it tries. The
        # Primary in bold on the label line, then each fallback on a subtext line of its own,
        # as `GO/`/`OR/`/`OL/` and the bare model: the full ids ran three to a line and
        # wrapped into a wall.
        stored = {**config, "primary_model": prim_model, "fallback_model": fall_model}
        resolved = resolve_auto_models(self.cog, stored)
        # Where a MimicAI subscription pays -- the server's key here, or the Personal one in a
        # DM -- its models are what run, so they are what is shown.
        model_lines = (["-# A MimicAI subscription pays here, so these are its models "
                        "(Set Models → API: Subscription)."] if paid else [])
        providers = {}
        for label, key in (("Response", "primary_model"), ("Image", "image_generation_model"),
                           ("Audio", "speech_model"), ("Grounding", "grounding_rag_model"),
                           ("LTM", "ltm_model"),
                           ("Training", "training_analysis_model")):
            if paid:
                chain = subscription_chain(self.cog, stored, key)
                providers[key] = {"openrouter"} if chain else set()
                if key == "primary_model" or all_models:
                    model_lines.extend(chain_lines(label, chain[0], chain[1]) if chain else
                                       [f"**{label}** → not in the subscription"])
                continue
            primary, fallbacks = model_chain(resolved, key, preferred)
            providers[key] = {model_provider(m) for m in (primary, *fallbacks)}
            if key != "primary_model" and not all_models:
                continue
            # A MimicAI Auto slot reads as its tier, with the model it runs on today beside it.
            tiers = {resolved[k]: short_model_name(stored[k])
                     for k in (key, MODEL_SLOT_PAIRS[key]) if auto_tier(stored.get(k))}
            shown = lambda m: (f"{tiers[m]} ({short_model_name(m)})" if m in tiers
                               else short_model_name(m))
            model_lines.extend(chain_lines(label, primary, fallbacks, shown))

        embed.add_field(name="Models", value="\n".join(model_lines)[:1024], inline=False)

        gen_val = (
            f"Temp: `{temp}`\n"
            f"Top P: `{top_p}`\n"
            f"Top K: `{top_k}`"
        )
        embed.add_field(name="\u200bGeneration Parameters", value=gen_val, inline=True)

        # OpenRouter and Ollama are sent these; Google ignores them, so a chain that never
        # leaves Google has nothing to show here -- the same test as the Params row's gate.
        # An unset key is not sent, which is the parameter off: the provider's 0.
        if providers["primary_model"] - {"gemini"}:
            adv_val = "\n".join(f"{label}: `{config.get(key) or 0}`" for label, key in (
                ("Freq P", "frequency_penalty"), ("Pres P", "presence_penalty"),
                ("Rep P", "repetition_penalty"), ("Min P", "min_p"), ("Top A", "top_a")))
            embed.add_field(name="Advanced Sampling", value=adv_val, inline=True)

        t_summary_raw = config.get("thinking_summary_visible", "off").lower()
        t_summary = "**`ON`**" if t_summary_raw == "on" else "`OFF`"
        t_level = config.get("thinking_level", "high").title()
        t_budget = config.get("thinking_budget", -1)
        budget_display = "Dynamic" if t_budget == -1 else f"{t_budget}"

        thinking_val = (
            f"Summary: {t_summary}\n"
            f"Effort: `{t_level}`\n"
            f"Budget: `{budget_display}`"
        )
        embed.add_field(name="Thinking/Reasoning", value=thinking_val, inline=True)
        if not providers["primary_model"] - {"gemini"}:
            # A blank third, where Advanced Sampling would be: Discord fills a row with up to
            # three inline fields, so without it Tools would be pulled up beside Thinking.
            embed.add_field(name="\u200b", value="\u200b", inline=True)

        # Tools and Behaviour share a row, half each; Media, full width, closes it.
        tools, behaviour, media = feature_fields(config, providers["speech_model"])
        embed.add_field(name="Tools", value=tools[:1024], inline=True)
        embed.add_field(name="Behaviour", value=behaviour[:1024], inline=True)
        embed.add_field(name="Media", value=media[:1024], inline=False)

        train_val = (
            f"Count: `{train_count}`\n"
            f"Context Size: `{train_ctx}`\n"
            f"Relevance Threshold: `{train_rel}`"
        )
        embed.add_field(name="Training Examples", value=train_val, inline=True)

        synopsis_status = "**`ON`**" if always_reads_synopsis(config) else "`OFF`"
        stm_val = (
            f"Length: `{config.get('stm_length', defaultConfig.CHATBOT_MEMORY_LENGTH)}`\n"
            f"Always Read Synopsis: {synopsis_status}"
        )
        embed.add_field(name="Short-Term Memory", value=stm_val, inline=True)

        ltm_ctx = config.get("ltm_context_size", 3)
        ltm_rel = config.get("ltm_relevance_threshold", 0.75)
        ltm_status = "**`ON`**" if config.get("ltm_creation_enabled", False) else "`OFF`"
        # Resolved, not read: absent means on, so a profile written before the toggle
        # existed must not render as though its memory had been switched off.
        auto_recall_status = "**`ON`**" if ltm_auto_recall_enabled(config) else "`OFF`"
        # Shown beside the threshold because it changes what that number means: with
        # Memory Search on, the automatic pass runs at LTM_AUTO_THRESHOLD_WITH_TOOL
        # instead, and the stored value applies again the moment this goes off.
        recall_status = "**`ON`**" if config.get("ltm_recall_tool_enabled", False) else "`OFF`"

        ltm_val = (
            f"Auto-Creation: {ltm_status}\n"
            f"Auto-Recall: {auto_recall_status}\n"
            + (f"This Server: `{ltm_count:,} / {defaultConfig.LIMIT_LTM:,}`\n" if guild_id
               else f"All Servers: `{ltm_count:,}`\n") +
            f"Context Size: `{ltm_ctx}`\n"
            f"Relevance Threshold: `{ltm_rel}`\n"
            f"Memory Search: {recall_status}"
        )
        embed.add_field(name="Long-Term Memories", value=ltm_val, inline=True)

        if appearance_data.get("custom_avatar_url"):
            embed.set_thumbnail(url=appearance_data["custom_avatar_url"])

        return embed

    async def _build_profile_manage_embed(self, interaction: discord.Interaction, profile_name: str, target_user_id: Optional[int] = None) -> discord.Embed:
        return await self._build_profile_embed(
            target_user_id or interaction.user.id, profile_name, interaction.channel_id,
            all_models=interaction.user.id in self.cog.dashboard_all_models)
