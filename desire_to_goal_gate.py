"""F1 desire-to-goal gate — shared state + tool-level enforcement.

Two independent process-wide sets keyed by ``session_id``:

* ``_ACTIVE``        — sessions where the user's message matched a
                       vague-desire pattern (gateway sets this at message
                       intake; cleared on `--clean` since session_id is
                       per-session).
* ``_SKILL_LOADED``  — sessions where the agent has already called
                       ``skill_view(name="orientation/desire-to-goal")``
                       at least once. Set by ``registry.dispatch`` hook.

Enforcement (in ``registry.dispatch``):

* If the session is ``_ACTIVE`` **and** skill not yet loaded → block
  side-effecting tools (write_file / terminal / execute_code /
  chief_spawn / cron_create / kanban_block / mc_project_create /
  workflow_run) and return a structured tool-error asking the agent
  to call ``skill_view`` first.
* All read-only and information-gathering tools (read_file,
  hindsight_recall, kanban_read, *_view, *_list, etc.) pass through.

This is the gateway-level injection's hard backstop: prompt text can
be ignored by sonnet, but a tool that returns an error and asks for
``skill_view`` cannot be quietly skipped.
"""
from __future__ import annotations

from typing import Iterable, Set

_ACTIVE: Set[str] = set()
_SKILL_LOADED: Set[str] = set()

# Tools blocked while F1 gate is active and skill not yet loaded.
# This list intentionally targets ANY side-effecting / committing tool —
# the agent must FIRST see the decomposition.
BLOCKED_TOOLS_BEFORE_SKILL: frozenset[str] = frozenset({
    "write_file", "edit_file",
    "terminal", "execute_code", "run_python", "run_shell",
    "chief_spawn", "chief_terminate",
    "mc_project_create", "mc_task_create",
    "cron_create",
    "kanban_block", "kanban_task_create",
    "workflow_run", "workflow_dispatch",
    "github_repo_create", "github_repo_delete",
    "google_sheet_create", "google_doc_create",
    "hindsight_retain",
})

# Skill names whose `skill_view` invocation marks the gate as "skill-seen".
DESIRE_TO_GOAL_SKILL_NAMES: frozenset[str] = frozenset({
    "desire-to-goal",
    "orientation/desire-to-goal",
})


def activate(session_id: str | None) -> None:
    if session_id:
        _ACTIVE.add(session_id)


def is_active(session_id: str | None) -> bool:
    return bool(session_id) and session_id in _ACTIVE


def mark_skill_loaded(session_id: str | None) -> None:
    if session_id:
        _SKILL_LOADED.add(session_id)


def has_skill_loaded(session_id: str | None) -> bool:
    return bool(session_id) and session_id in _SKILL_LOADED


def reset(session_id: str | None) -> None:
    if session_id:
        _ACTIVE.discard(session_id)
        _SKILL_LOADED.discard(session_id)


def block_message(tool_name: str) -> str:
    """Returns the directive string the dispatcher should hand back to the
    agent in place of a real tool result. Intentionally specific about
    WHAT to do next so the agent doesn't re-try the same blocked tool."""
    return (
        f"BLOCKED by F1 desire-to-goal gate. The tool '{tool_name}' is a "
        f"side-effecting / committing action and cannot run before you "
        f"have shown the «истинная цель / средство / место / контекст» "
        f"decomposition for this vague-desire request.\n\n"
        f"DO THIS NEXT (and ONLY this):\n"
        f"  skill_view(name=\"orientation/desire-to-goal\")\n\n"
        f"Then follow its «ПЕРВЫЙ ход» section: output the 4-7 line "
        f"decomposition in chat and ask the user to confirm. Only AFTER "
        f"confirmation may you call '{tool_name}' or any other blocked "
        f"tool below:\n  "
        + ", ".join(sorted(BLOCKED_TOOLS_BEFORE_SKILL))
    )


def housekeep(max_sessions: int = 1000) -> None:
    """Cap memory growth (sessions are per-message, can accumulate)."""
    if len(_ACTIVE) > max_sessions:
        _ACTIVE.clear()
    if len(_SKILL_LOADED) > max_sessions:
        _SKILL_LOADED.clear()


def known_blocked() -> Iterable[str]:
    return sorted(BLOCKED_TOOLS_BEFORE_SKILL)
