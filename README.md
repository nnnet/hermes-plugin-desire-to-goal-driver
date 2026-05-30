# desire-to-goal-driver

Hermes `pre_llm_call` plugin that bridges vague-desire user messages to
the deterministic [workflow-engine](https://github.com/nnnet/hermes-plugin-workflow-engine).

## What it does

When the bot's `pre_llm_call` hook fires:

1. Detect whether the most recent user turn looks like a vague desire
   ("хочу сайт", "want to launch a SaaS", "помоги с продуктом", ...) — i.e.
   needs clarification before the LLM tries to act.
2. If yes, invoke the workflow-engine's `cli.py` (mounted at
   `/opt/workflow-engine/`) with the active `invocation_id` from the
   plugin's Registry-backed store.
3. Inject the engine's `mini_prompt` + `state_summary` into the LLM
   context as a system-priority hint, so the bot's next turn lands on
   the right clarification question rather than free-improvised "Sure,
   I'll build it" replies.

This is the production answer to a long-standing failure mode where
`orientation/desire-to-goal` SKILL.md instructions kept getting ignored
by the model under load. Driver runs deterministically every turn; the
model can drift, but the engine's slot-extraction and phase routing
won't.

## Configuration

`plugin.yaml` is bundled-defaults; runtime overrides go into Hermes'
gateway config (`~/.hermes/config.yaml` → `plugins.desire-to-goal-driver`).

| Key | Default | Purpose |
|-----|---------|---------|
| `enabled` | `true` | Master switch |
| `workflow_name` | `desire-to-goal` | Which `workflow/` skill to run |
| `min_chars_for_engine` | `8` | Skip very short turns (commands like "ок") |

See `__init__.py` for the full extension-point surface.

## Mounting

External plugin: pulled by
[`nnnet/AiManager:infra/hermes/scripts/sync-external-plugins.sh`](https://github.com/nnnet/AiManager/blob/prod/infra/hermes/scripts/sync-external-plugins.sh)
into `sources/hermes-external-plugins/desire-to-goal-driver/`, then
bind-mounted into the container at
`/opt/hermes/plugins/desire-to-goal-driver/`.

## Related

- [`hermes-plugin-workflow-engine`](https://github.com/nnnet/hermes-plugin-workflow-engine) — the engine this driver invokes
- `infra/hermes/.hermes/skills/orientation/desire-to-goal/` (in
  AiManager) — the skill carrying the workflow definition
- `nnnet/hermes-agent` — gateway fork; this plugin overlays the
  baked-in plugin dir via bind-mount, no image rebuild required for
  edits.

## License

MIT — see LICENSE.
