"""desire-to-goal-driver — gateway plugin that drives the workflow-engine.

Closes the structural gap between the SKILL.md instruction
("call /opt/workflow-engine/cli.py") and what the main bot actually
does (often ignores the skill and answers from training).

Mechanism — pure plugin code, no upstream gateway/agent modification:

* Registers a ``pre_llm_call`` hook (lifecycle event documented in
  ``hermes_cli/plugins.py:VALID_HOOKS``).
* On every turn, checks the user's latest message against a copy of
  the upstream ``VAGUE_DESIRE_PATTERNS`` regex.
* When matched, shells to the workflow-engine CLI with the user's
  message + session id, captures the JSON result (phase, slots,
  confidence, mini_prompt, instructions, artifact path), and returns
  ``{"context": "<engine block>"}`` so the conversation_loop injects
  the block into the current turn's user message.

The bot then sees the engine's verbatim output in context and is
instructed to forward mini_prompt to TG. No tool call required on the
bot's side; the slot extractor, anti-pattern detectors and phase
routing have already run.

Why hook here vs gateway/run.py
    The upstream ``gateway/run.py`` does inject a static
    DESIRE_TO_GOAL_PRELOAD_NOTE when the F1-gate trigger fires, but
    that note still depends on the bot calling ``skill_view`` and then
    ``terminal_run`` against the engine CLI — two tool-call steps the
    bot often skips. ``pre_llm_call`` runs the engine eagerly so the
    output is in the conversation BEFORE the model decides whether to
    call any tool.

    Doing this via a plugin keeps upstream files untouched — important
    for fork hygiene; merges from NousResearch don't conflict on
    gateway/run.py.

Failure mode
    Engine subprocess timeout / non-zero exit / JSON parse error →
    hook returns ``None`` and the turn proceeds normally with just the
    upstream PRELOAD_NOTE. Container logs carry the engine error for
    diagnosis.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional


def functools_lru_cache_decorator():
    """Return ``functools.lru_cache(maxsize=1)`` — wrapped so the symbol is
    locally namespaced (avoids accidental ``functools.lru_cache`` collisions
    in tests that patch it)."""
    return functools.lru_cache(maxsize=1)

logger = logging.getLogger(__name__)


# Copy of upstream VAGUE_DESIRE_PATTERNS (gateway/run.py). Kept in sync
# manually — if upstream extends its list, mirror it here. Slight
# divergence is acceptable; the upstream regex remains authoritative
# for the F1 gate's sticky-state activation, this one only gates the
# engine invocation from the plugin side.
_VAGUE_DESIRE_PATTERNS = [
    # Quality adverbs
    r"\bприбыльн", r"\bэффективн", r"\bудобн", r"\bпонятн",
    r"\bнадёжн", r"\bнадежн", r"\bкрасив", r"\bбыстр",
    # Build-verb means (the proposed tool)
    r"\bбот[\s,]", r"\bбота\b", r"\bдашборд", r"\bсайт",
    r"\bмагазин", r"\bсистем", r"\bинструмент", r"\bсервис",
    r"\bприложени", r"\bпайплайн", r"\bпроект", r"\bкоманд",
    r"\bкурс\b", r"\bканал\b",
    # Vague verbs (imperative)
    r"\bпомоги", r"\bспроектируй", r"\bпридумай", r"\bразработай",
    r"\bоптимизируй", r"\bулучши", r"\bавтоматизируй",
    r"\bразберись", r"\bспланируй", r"\bнайди мне\b",
    r"\bподбирай", r"\bмониторь",
    # Multi-step "сделай мне X" / "нужна штука для Y"
    r"\bсделай мне\b", r"\bнужна штука\b", r"\bнужн[ао] что-то",
    r"\bвыучи меня",
    # Extension: infinitive forms missed by upstream regex
    # (regular complaint — "автоматизировать" vs "автоматизируй").
    r"\bавтоматизировать", r"\bспроектировать", r"\bразработать",
    r"\bоптимизировать",
]
_VAGUE_DESIRE_RE = re.compile("|".join(_VAGUE_DESIRE_PATTERNS), re.IGNORECASE)


WORKFLOW_DIR = os.environ.get(
    "DESIRE_TO_GOAL_WORKFLOW_DIR",
    "/opt/data/skills/orientation/desire-to-goal/workflow",
)
SKILL_DIR = os.environ.get(
    "DESIRE_TO_GOAL_SKILL_DIR",
    "/opt/data/skills/orientation/desire-to-goal",
)
ENGINE_CLI = os.environ.get(
    "DESIRE_TO_GOAL_ENGINE_CLI",
    "/opt/workflow-engine/cli.py",
)
PYTHON_BIN = os.environ.get(
    "DESIRE_TO_GOAL_PYTHON",
    "python3",
)
SUBPROCESS_TIMEOUT_SECONDS = float(
    os.environ.get("DESIRE_TO_GOAL_TIMEOUT", "75")
)

# Post-DONE grace period (seconds): after a desire-to-goal workflow
# finishes with reason=DONE for a given (agent_id, conversation_id),
# do NOT auto-start a new invocation for ``DESIRE_TO_GOAL_POST_DONE_GRACE``
# seconds — even if the user's next message matches the vague-desire
# regex. This window belongs to chief-manager (next_skill_hint) to
# acknowledge the hand-off and start its own phases. Without this gate,
# the user's first reply to chief ("ты передал тим-лиду?", "когда команда
# соберётся?") often contains words like "команда" or "проект" that
# re-trigger the vague-desire regex, which would start a new clarification
# invocation and re-lock all execution tools — breaking the hand-off.
POST_DONE_GRACE_SECONDS = float(
    os.environ.get("DESIRE_TO_GOAL_POST_DONE_GRACE", "300")
)

# Phases in which tool restrictions apply. DONE → release; missing state
# file → no workflow active → no restrictions. Anything else (DECOMPOSE,
# REFLECT, LOCK and any future intermediate phase the schema may add) is
# treated as "clarification — restrict tools" by default.
_CLARIFICATION_TERMINAL_PHASE = "DONE"


_DEFAULT_NEXT_SKILL = "devops/chief-manager"


def _read_artifact(artifact_path: str) -> dict[str, Any]:
    """Read and parse the artifact YAML. Returns {} on any failure."""
    try:
        import yaml  # PyYAML already in container
        with open(artifact_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "desire-to-goal-driver: artifact read failed (%s)", exc,
        )
        return {}


def _read_next_skill_hint(artifact_path: str) -> str:
    """Read ``next_skill_hint`` from artifact's ``export`` block.

    Fallback to ``_DEFAULT_NEXT_SKILL`` if missing/malformed."""
    data = _read_artifact(artifact_path)
    if not data:
        return _DEFAULT_NEXT_SKILL
    hint = (
        (data.get("export") or {}).get("next_skill_hint")
        or data.get("next_skill_hint")
        or _DEFAULT_NEXT_SKILL
    )
    return str(hint).strip() or _DEFAULT_NEXT_SKILL


def _flatten_goal_for_chief(artifact_path: str) -> str:
    """Render the artifact's ``goal`` dict as flat text the chief-manager
    can read as Phase-0 input.

    chief.requires_goal_refinement() is a heuristic on plain text — it
    checks for measurability markers and vague words. Passing structured
    YAML through it returns True (looks vague), forcing chief to
    re-clarify. By rendering each slot as ``label: value`` lines we both
    satisfy the heuristic AND give the chief everything it needs for
    decomposition without re-asking the user.

    Returns empty string if no goal slots found — caller can decide what
    to do (probably show only the hint + artifact path).
    """
    data = _read_artifact(artifact_path)
    goal = data.get("goal") or {}
    if not isinstance(goal, dict) or not goal:
        return ""
    lines = []
    # Stable ordering — preserves the schema's slot order if present.
    for key, val in goal.items():
        if val is None or str(val).strip() == "":
            continue
        lines.append(f"- {key}: {val}")
    return "\n".join(lines)


def _build_handoff_block(artifact_path: str, next_skill: str) -> str:
    """Build the post-DONE hand-off instruction block.

    Pattern: single responsible agent per phase (see
    .claude/plans/2026-05-27T06-18__desire-to-goal-to-execution-handoff.md).
    On DONE, ownership passes from Hermes (clarifier) to chief-manager
    (project executor). Hermes' job in this turn is ONE acknowledgement
    line + a single ``skill_load`` (or ``chief_spawn``) tool call. No
    decomposition, no OSINT, no kanban — chief-manager will do all that.
    """
    goal_text = _flatten_goal_for_chief(artifact_path)
    goal_section = ""
    if goal_text:
        goal_section = (
            "\nClarified goal slots (already complete — DO NOT re-clarify):\n"
            f"{goal_text}\n"
        )
    return (
        "\n"
        "[HAND-OFF — workflow DONE, ownership transfers]\n"
        f"Artifact: {artifact_path}\n"
        f"Next responsible: {next_skill} (project executor)\n"
        + goal_section
        + "\n"
        "**Your job in this turn (DO ONLY THIS):**\n"
        "1. ONE short line to the user: «Цель ясна — передаю команде проекта.» "
        "(or equivalent, ≤1 sentence). NO recap of the clarified goal — "
        "the user just confirmed it.\n"
        f"2. Load skill `{next_skill}` and start its protocol with the\n"
        "   clarified goal text above as Phase-0 input. chief-manager "
        "should SKIP its Phase 1 (clarify) — goal is already clarified "
        "by desire-to-goal — and START AT Phase 2 (decompose into "
        "subgoals + task tree).\n"
        f"   Tool call: skill_load {next_skill}\n"
        "\n"
        "**FORBIDDEN this turn**: web_search, terminal, kanban_*, "
        "github_*, write_file, chief_spawn-without-skill-load, multi-step "
        "plans, drafting deliverables in chat. chief-manager will decompose, "
        "build the capability matrix, propose a plan, and ask the user for "
        "approval on its own approval-gate phase. You are NOT chief-manager — "
        "you only hand the work off.\n"
    )


def _build_engine_block(payload: dict[str, Any]) -> str:
    """Format the engine's JSON return into a context block the bot
    should treat as authoritative for this turn."""
    mp = (payload.get("mini_prompt") or "").strip()
    phase = payload.get("phase", "")
    summary = payload.get("state_summary", {}) or {}
    slots = summary.get("slots") or {}
    instr = (payload.get("instructions_for_bot") or "").strip()
    iteration = summary.get("iteration", 0)
    completeness = summary.get("completeness", 0.0)

    is_done = str(phase).upper() == "DONE"
    # Pre-DONE phases: forbid tool calls (engine owns the turn).
    # DONE phase: REQUIRE exactly one tool call (skill_load) — ownership
    # is transferring to chief-manager and the bot must initiate that.
    tool_call_directive = (
        "On DONE phase you MUST issue exactly ONE tool call: "
        "skill_load with name=devops/chief-manager (or whatever the "
        "hand-off block below specifies). The single sentence to the "
        "user goes alongside the tool call in this same turn."
        if is_done
        else "Do NOT add tool calls in this turn."
    )

    block = (
        "[desire-to-goal workflow-engine — pre-computed reply]\n"
        f"Phase: {phase}\n"
        f"Iteration: {iteration}\n"
        f"Completeness: {completeness:.2f}\n"
        f"Slots: {json.dumps(slots, ensure_ascii=False)}\n"
        "\n"
        "**Your reply MUST be the mini_prompt below, verbatim — "
        "substitute `<placeholders>` from slots when slot value is "
        "present; otherwise keep the placeholder text. The engine "
        "already ran the slot extractor, programmatic anti-pattern "
        "detectors and phase routing for this turn. Do NOT re-derive "
        f"slots. Do NOT brand-list. {tool_call_directive}**\n"
        "\n"
        f"<mini_prompt>\n{mp}\n</mini_prompt>\n"
    )
    if instr:
        block += f"\nEngine instruction: {instr}\n"
    artifact_path = payload.get("artifact_file") or ""
    if artifact_path:
        block += f"\nArtifact YAML written: {artifact_path}\n"

    # Append the hand-off block when the engine signals DONE — at that
    # point ownership passes from Hermes (clarifier) to chief-manager.
    if is_done and artifact_path:
        next_skill = _read_next_skill_hint(artifact_path)
        block += _build_handoff_block(artifact_path, next_skill)

    return block


# ─────────────────────────────────────────────────────────────────────────
# Tool-gating policy
# ─────────────────────────────────────────────────────────────────────────
#
# The skill declares which tools are allowed at which phase via the
# ``tools:`` section in its SKILL.md YAML frontmatter. We honour that
# declaration in two places:
#
#   1. ``pre_tool_call`` hook (this file) — hard enforcement. Any tool
#      call that violates the policy gets blocked with a synthetic error
#      message instructing the bot to use only the allowed tools.
#
#   2. ``pre_llm_call`` hook (existing) — soft guidance. We append a
#      block-list line to the engine block so the model sees the
#      restriction as natural instruction (defence in depth — most calls
#      never even get attempted because the model obeys the prompt; the
#      hook only catches the misses).
#
# Pure plugin code, no upstream patches.


def _parse_skill_md_frontmatter(skill_md_path: str) -> dict[str, Any]:
    """Read and parse the YAML frontmatter block at the top of a SKILL.md.

    Frontmatter = whatever is between the first two ``---`` delimiters at
    the top of the file. Returns the parsed dict (or {} on any failure).
    """
    try:
        with open(skill_md_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if not text.startswith("---"):
            return {}
        # Strip the leading "---\n" and split on the next "---" on its own line.
        body = text[3:].lstrip("\n")
        end = body.find("\n---")
        if end < 0:
            return {}
        import yaml
        return yaml.safe_load(body[:end]) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("desire-to-goal-driver: SKILL.md parse failed: %s", exc)
        return {}


@functools_lru_cache_decorator()
def _load_skill_tools_policy() -> dict[str, Any]:
    """Load and cache the ``tools:`` section from desire-to-goal SKILL.md.

    Cached for the lifetime of the process — SKILL.md changes require a
    plugin reload anyway (i.e. hermes container restart).
    """
    skill_md = os.path.join(SKILL_DIR, "SKILL.md")
    fm = _parse_skill_md_frontmatter(skill_md)
    tools_cfg = fm.get("tools") or {}
    if not isinstance(tools_cfg, dict):
        return {}
    return tools_cfg


def _policy_for_phase(phase: str) -> Optional[dict[str, Any]]:
    """Return the allow/block dict for the given phase, or None if no policy."""
    cfg = _load_skill_tools_policy()
    if not cfg:
        return None
    by_phase = cfg.get("by_phase") or {}
    if isinstance(by_phase, dict) and phase in by_phase:
        entry = by_phase[phase]
        if isinstance(entry, dict):
            return entry
    default = cfg.get("default")
    if isinstance(default, dict):
        return default
    return None


def _tool_matches(tool_name: str, pattern: str) -> bool:
    """Glob-aware match. Supports trailing ``*`` (e.g. ``kanban_*``) and
    the wildcard ``*`` meaning "any tool"."""
    if not pattern:
        return False
    if pattern == "*":
        return True
    if pattern.endswith("*"):
        return tool_name.startswith(pattern[:-1])
    return tool_name == pattern


def _is_tool_allowed_by_policy(tool_name: str, policy: dict[str, Any]) -> bool:
    """Whitelist OR blocklist semantics, per SKILL.md tools section.

    - ``allowed``-only: tool is OK iff it matches some ``allowed`` entry.
    - ``blocked``-only: tool is OK iff it matches NO ``blocked`` entry.
    - Both present: ``blocked`` wins (defensive — a tool explicitly
      blocked is never allowed, even if also in ``allowed``).
    """
    blocked = policy.get("blocked")
    if isinstance(blocked, list):
        if any(_tool_matches(tool_name, p) for p in blocked):
            return False
    allowed = policy.get("allowed")
    if isinstance(allowed, list):
        return any(_tool_matches(tool_name, p) for p in allowed)
    # No allowed declared and not blocked → permit (no constraint).
    return True


def _read_current_phase(session_id: str) -> Optional[str]:
    """Read the workflow's current phase from the state file.

    Returns None if no active workflow for this session — caller should
    treat that as "no restrictions" and let the call through.
    """
    state_dir = Path(
        os.environ.get("WORKFLOW_STATE_DIR")
        or (Path.home() / ".hermes" / "workflow_state")
    )
    state_path = state_dir / "desire-to-goal" / f"{session_id}.json"
    if not state_path.exists():
        return None
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    phase = raw.get("phase")
    return str(phase) if phase else None


_EMPTY_FALLBACK_GUARD = (
    "\n\n"
    "CRITICAL — DO NOT respond with empty-input fallbacks like «Пустое "
    "сообщение — напиши что хотел.», «Похоже, сообщение пустое — напиши, "
    "что хотел сказать.», «Кажется, сообщение пришло пустым — напиши ещё "
    "раз?», «Кажется, сообщение пустое — что-то хотел написать?». The "
    "user's previous message IS valid input — this tool block error is "
    "NOT an empty user message. Reply directly to the user in plain "
    "Russian text (no tool call), continuing the clarification dialogue "
    "per the engine's mini_prompt. chief-manager will pick up execution "
    "after phase DONE."
)


def _format_block_message(
    tool_name: str, phase: str, policy: dict[str, Any],
) -> str:
    allowed_list = policy.get("allowed")
    blocked_list = policy.get("blocked")
    if isinstance(allowed_list, list) and allowed_list:
        return (
            f"Tool '{tool_name}' is not available while desire-to-goal "
            f"workflow is in phase '{phase}'. Allowed tools this phase: "
            f"{', '.join(map(str, allowed_list))}."
            + _EMPTY_FALLBACK_GUARD
        )
    if isinstance(blocked_list, list) and blocked_list:
        return (
            f"Tool '{tool_name}' is blocked in phase '{phase}' "
            f"(blocked: {', '.join(map(str, blocked_list))})."
            + _EMPTY_FALLBACK_GUARD
        )
    return (
        f"Tool '{tool_name}' blocked by desire-to-goal policy in "
        f"phase '{phase}'."
        + _EMPTY_FALLBACK_GUARD
    )


def _on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, str]]:
    """pre_tool_call hook — enforce SKILL.md-declared tool policy.

    Returns ``{"action": "block", "message": "..."}`` when the call
    violates policy for the current workflow phase. Returns ``None``
    otherwise (pass-through).

    Degraded behaviour on any unexpected error: pass through. The
    workflow MUST NOT break on a policy-read failure — clarification
    can proceed without enforcement (the prompt-injection in
    pre_llm_call still gives a soft hint).
    """
    try:
        tool_name = str(kwargs.get("tool_name") or "").strip()
        if not tool_name:
            return None
        session_id = str(kwargs.get("session_id") or "default")

        phase = _read_current_phase(session_id)
        if phase is None:
            return None
        if phase.upper() == _CLARIFICATION_TERMINAL_PHASE:
            # Workflow already at DONE — restrictions lifted.
            return None

        policy = _policy_for_phase(phase)
        if policy is None:
            return None

        if _is_tool_allowed_by_policy(tool_name, policy):
            return None

        msg = _format_block_message(tool_name, phase, policy)
        logger.info(
            "desire-to-goal-driver: BLOCK tool=%s phase=%s session=%s",
            tool_name, phase, session_id,
        )
        return {"action": "block", "message": msg}
    except Exception as exc:  # noqa: BLE001 — never break tool execution
        logger.warning(
            "desire-to-goal-driver pre_tool_call hook error (pass-through): %s",
            exc,
        )
        return None


_WORKFLOW_NAME = "desire-to-goal"


def _recent_done_invocation(
    registry, workflow_name: str, agent_id: str, conversation_id: str,
    grace_sec: float,
) -> Optional[float]:
    """Return finished_ts of the most recent DONE invocation for this
    (workflow, agent, conv) triple if it's within ``grace_sec`` of now,
    else None.

    Uses Registry's connection helper so we share the same DB cursor
    semantics as the rest of the engine. Failures degrade silently to
    None — plugin must never break the turn on a DB hiccup.
    """
    try:
        import time as _time
        cutoff = _time.time() - grace_sec
        with registry._conn() as c:  # noqa: SLF001 — intentional internal use
            row = c.execute(
                "SELECT finished_ts FROM workflow_invocations "
                "WHERE workflow_name=? AND agent_id=? AND conversation_id=? "
                "AND finished_reason='DONE' AND finished_ts >= ? "
                "ORDER BY finished_ts DESC LIMIT 1",
                (workflow_name, agent_id, conversation_id, cutoff),
            ).fetchone()
        return float(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("recent_done lookup failed (%s) — pass through", exc)
        return None


def _make_registry():
    """Lazy Registry import. Returns None on failure (graceful degrade)."""
    try:
        engine_root = "/opt/workflow-engine"
        import sys
        if engine_root not in sys.path:
            sys.path.insert(0, engine_root)
        from core.registry import Registry  # type: ignore  # noqa: WPS433
        return Registry()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "desire-to-goal-driver: Registry import failed (%s); "
            "engine still callable via CLI subprocess (--start --find-active)",
            exc,
        )
        return None


def _resolve_identity(kwargs: dict[str, Any]) -> tuple[str, str]:
    """Derive (agent_id, conversation_id) from gateway hook kwargs.

    agent_id        — ``HERMES_AGENT_ID`` env (default ``main``). Sub-agent
                      spawners override per-process so chief/worker
                      invocations land in their own row.
    conversation_id — derived from gateway-supplied ``session_id``. When
                      operator runs ``--clean`` the gateway gives a new
                      session_id → new conversation_id → registry starts
                      a fresh invocation (which is exactly the desired
                      test-isolation primitive).
    """
    agent_id = os.environ.get("HERMES_AGENT_ID", "main")
    session_id = str(kwargs.get("session_id") or "default")
    conversation_id = f"session-{session_id}"
    return agent_id, conversation_id


_CANCEL_PATTERNS = (
    "забудь", "сначала", "отмени ", "отмена",
    "/reset", "/start ", "/start\n", "новая задача",
    "новая тема", "переключаем", "вернёмся к", "вернемся к",
)


def _detect_cancel_intent(user_msg: str) -> bool:
    """Cancel-intent heuristic. Engine has its own ``detect_cancellation``
    inside ``core/detectors.py`` — this is the PLUGIN-side mirror so we
    can finalize the registry row BEFORE re-invoking the engine on the
    next turn (defense in depth)."""
    if not user_msg:
        return False
    msg = user_msg.lower()
    return any(p in msg for p in _CANCEL_PATTERNS)


def _on_pre_llm_call(**kwargs: Any) -> Optional[dict[str, str]]:
    """pre_llm_call hook entry point — Registry-driven lifecycle.

    Flow per turn:
      1. Compute (agent_id, conversation_id) from kwargs.
      2. Look up active invocation via Registry.find_active(...).
      3. If cancel-intent detected → mark CANCELLED + return None.
      4. If no active AND vague-desire regex doesn't match → return None.
      5. Otherwise: ensure active invocation exists (start if needed),
         pass invocation_id to engine subprocess.
      6. On terminal phase: registry already finished by engine.
    """
    user_msg = str(kwargs.get("user_message") or "").strip()
    if not user_msg:
        return None

    agent_id, conversation_id = _resolve_identity(kwargs)
    session_id = str(kwargs.get("session_id") or "default")
    registry = _make_registry()

    active = (
        registry.find_active(_WORKFLOW_NAME, agent_id, conversation_id)
        if registry is not None else None
    )

    # ── cancel-intent: finalize current invocation BEFORE engine call ───
    if active is not None and _detect_cancel_intent(user_msg):
        try:
            registry.cancel(active.invocation_id)  # type: ignore[union-attr]
            logger.info(
                "desire-to-goal-driver: cancelled invocation %s on user intent",
                active.invocation_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel failed: %s", exc)
        # Bail out for this turn — let the bot handle the user's reset
        # message normally, vague-desire regex will re-trigger if needed.
        return None

    # ── trigger decision: active continuation OR cold-start match ───────
    regex_match = bool(_VAGUE_DESIRE_RE.search(user_msg))
    if active is None and not regex_match:
        return None

    # ── post-DONE grace period: don't re-trigger after hand-off ─────────
    # If desire-to-goal recently finished DONE for this (agent, conv) and
    # we're within the grace window, the next responsible skill
    # (chief-manager) owns the turn. Don't auto-start a fresh clarification
    # workflow even if the user's reply re-trips the vague-desire regex —
    # that just locks tools again and breaks the hand-off. See
    # POST_DONE_GRACE_SECONDS for the rationale.
    if active is None and registry is not None and POST_DONE_GRACE_SECONDS > 0:
        recent_done_ts = _recent_done_invocation(
            registry, _WORKFLOW_NAME, agent_id, conversation_id,
            POST_DONE_GRACE_SECONDS,
        )
        if recent_done_ts is not None:
            import time as _time
            age = _time.time() - recent_done_ts
            logger.info(
                "desire-to-goal-driver: post-DONE grace — skipping new "
                "invocation (age=%.0fs < %s, conv=%s); chief-manager owns "
                "this turn",
                age, POST_DONE_GRACE_SECONDS, conversation_id,
            )
            return None

    # ── ensure we have an invocation_id for the engine ──────────────────
    if active is None and registry is not None:
        try:
            active = registry.start(
                workflow_name=_WORKFLOW_NAME,
                agent_id=agent_id,
                conversation_id=conversation_id,
            )
            logger.info(
                "desire-to-goal-driver: started invocation %s (agent=%s conv=%s)",
                active.invocation_id, agent_id, conversation_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Registry.start() failed: %s — falling back to legacy", exc)
            active = None

    invocation_id = active.invocation_id if active is not None else session_id

    argv = [
        PYTHON_BIN, ENGINE_CLI,
        "--workflow", WORKFLOW_DIR,
        "--invocation-id", invocation_id,
        "--user", user_msg,
    ]
    # Pass identity so engine can self-bootstrap a row if registry has gaps.
    argv.extend(["--agent-id", agent_id, "--conversation-id", conversation_id])
    if _is_test_mode():
        argv.append("--is-test")

    try:
        result = subprocess.run(
            argv,
            capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "desire-to-goal-driver: engine timeout after %ss for invocation=%s",
            SUBPROCESS_TIMEOUT_SECONDS, invocation_id,
        )
        return None
    except Exception as exc:
        logger.warning("desire-to-goal-driver: subprocess failed: %s", exc)
        return None

    if result.returncode != 0 or not (result.stdout or "").strip():
        logger.warning(
            "desire-to-goal-driver: engine rc=%s invocation=%s stderr=%r",
            result.returncode, invocation_id,
            (result.stderr or "")[:300],
        )
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.warning(
            "desire-to-goal-driver: engine returned non-JSON: %s",
            exc,
        )
        return None

    block = _build_engine_block(payload)
    logger.info(
        "desire-to-goal-driver: injected engine block "
        "(phase=%s iter=%s invocation=%s)",
        payload.get("phase"),
        payload.get("iteration"),
        invocation_id,
    )
    return {"context": block}


def _on_session_reset(session_id: str | None = None, platform: str | None = None,
                      **_kwargs: Any) -> None:
    """Gateway fires this AFTER ``/new`` or ``/reset`` finishes:
      - old session row in state.db.sessions has ``ended_at`` set
      - new sessions.json entry created with a fresh ``session_id``
      - this hook is called with the NEW session_id

    Our job: any active workflow invocation for this agent that is NOT
    for the new session belongs to the just-closed old session. Mark
    those finished with ``reason='user_reset'`` so the audit trail keeps
    them but they no longer match ``find_active`` queries.

    Production-aligned: same code path triggers for real users (``/new``)
    and for F1 tests (test runner sends ``/new`` between cases). No DB
    hacks, no manual session manipulation — Hermes' built-in primitive
    drives everything.
    """
    if not session_id:
        return
    agent_id = os.environ.get("HERMES_AGENT_ID", "main")
    registry = _make_registry()
    if registry is None:
        return
    new_conv = f"session-{session_id}"
    finished = 0
    try:
        actives = registry.list_active(
            workflow_name=_WORKFLOW_NAME, agent_id=agent_id,
        )
        for inv in actives:
            if inv.conversation_id != new_conv:
                registry.finish(inv.invocation_id, reason="CANCELLED")
                finished += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("on_session_reset registry op failed: %s", exc)
        return
    if finished:
        logger.info(
            "on_session_reset: finished %d invocation(s) for agent=%s "
            "(new session_id=%s platform=%s)",
            finished, agent_id, session_id, platform,
        )


def register(ctx) -> None:
    """Plugin entry point — called by the loader at startup.

    Subscribes to two plugin hooks:
      pre_llm_call         — main turn-driver (slot extraction + phase)
      on_session_reset     — finalize old workflow on /new or /reset
                             (precedes a fresh session, plugin marks the
                              stranded invocation as CANCELLED)

    Why no on_session_finalize subscription:
      Hermes fires on_session_finalize both on /new (line 9349 in
      gateway/run.py, with platform="telegram" etc.) AND on idle-timeout
      sweep (separate 5-min loop, with platform="gateway"). Subscribing
      would race with on_session_reset on the /new path and mislabel
      reasons. For idle-timeout we rely on the janitor cron sweep
      (last_active_ts > cutoff → ABANDONED_TIMEOUT) which gives a clean
      separation: explicit user reset → CANCELLED via on_session_reset,
      passive abandonment → ABANDONED_TIMEOUT via janitor cron.
    """
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("on_session_reset", _on_session_reset)
    logger.info(
        "desire-to-goal-driver plugin registered hooks: "
        "pre_llm_call + pre_tool_call + on_session_reset"
    )
