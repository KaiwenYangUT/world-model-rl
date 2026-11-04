# Third-party patches

Every local change to code under `third_party/` is listed here (CONTRIBUTING.md rule 3).

| Submodule | Pinned commit | Upstream |
|---|---|---|
| `third_party/dreamerv3` | `e01491fad6434b2245a3b8ca201dd7faedcc458c` (2026-10-03, "Pin Bazel 7 for the DMLab install script (#223)") | https://github.com/danijar/dreamerv3 |

## Patches
None yet. (Planned: `wm_eval` next to `report()` in `dreamerv3/agent.py`, PLAN.md §8.)

## Behaviour overrides in our code (no third-party files edited)
- **crafter 1.8.3 (pip):** `rlproj/envs/crafter_task.py` subclasses `crafter.Env` as `_DeterministicCrafterEnv`. It overrides `_balance_object` to sort each chunk's objects by position before Crafter's `random.randint` picks one to despawn. Upstream iterates a `set` whose order follows memory addresses, so identical seeds and actions could diverge within one process (seen at steps 50–143 in T03). RNG draws are unchanged, so the distribution is the same. It also sets `_seed`/`_episode` before each reset to force the world seed.
