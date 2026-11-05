# Third-party patches

Every local change to code under `third_party/` is listed here (CONTRIBUTING.md rule 3).

| Submodule | Pinned commit | Upstream |
|---|---|---|
| `third_party/dreamerv3` | `e01491fad6434b2245a3b8ca201dd7faedcc458c` (2026-10-03, "Pin Bazel 7 for the DMLab install script (#223)") | https://github.com/danijar/dreamerv3 |

## Patches
Apply after a fresh submodule checkout: `git -C third_party/dreamerv3 apply ../patches/dreamerv3-report-wm-horizon.patch`.

1. **`dreamerv3/agent.py`: open-loop world-model metrics in `Agent.report`** (2026-11-04, RQ3 / PLAN.md §8). File: `patches/dreamerv3-report-wm-horizon.patch`.
   - Adds `_wm_horizon_metrics()` (module level) and one call to it in `report()`.
   - On the full report batch (T = `report_length` = 32), it observes the first half with `dyn.observe` and imagines the second half with `dyn.imagine` using the recorded actions, as upstream's open-loop video does. It then compares `rew(feat, 2).pred()` with the reward and `con(feat, 2).prob(1)` with 1 − is_terminal.
   - Head semantics checked in `agent.py`: `imag_loss` consumes `rew(...).pred()` as rewards and `con(...).prob(1)` as continuation.
   - It emits scalars `wm/k{k:02d}/<sum or count>`: k = 0 is the posterior over the context, k = 1..16 is open loop; steps after a new episode starts are masked. They land in `metrics.jsonl` (prefix `report/`, or `eval/` for the eval buffer) during training, and `scripts/wm_eval.py` uses them post hoc.
   - It does not change training; it only adds outputs to `report()`. Not present in runs started before the patch (e.g. `main/dreamerv3/seed0`, whose metrics.jsonl has no `wm/` keys; `wm_eval.py` still works on its checkpoint).
   - Upstream note, not changed: `report()` does `mets.update(mets)`, so upstream's own report returns no loss scalars.

## Behaviour overrides in our code (no third-party files edited)
- **crafter 1.8.3 (pip):** `rlproj/envs/crafter_task.py` subclasses `crafter.Env` as `_DeterministicCrafterEnv`. It overrides `_balance_object` to sort each chunk's objects by position before Crafter's `random.randint` picks one to despawn. Upstream iterates a `set` whose order follows memory addresses, so identical seeds and actions could diverge within one process (seen at steps 50–143 in T03). RNG draws are unchanged, so the distribution is the same. It also sets `_seed`/`_episode` before each reset to force the world seed.
- **crafter 1.8.3 (pip), world cache:** `_DeterministicCrafterEnv.reset` caches each freshly generated world per process, keyed by (world RNG seed, area). The cached copy is a deep copy of `world.__dict__` plus the player, which includes the shared `world.random`. Later resets restore a copy into the env's existing `World` object and repeat the non-generation lines of upstream `Env.reset`. Reset: ~490 ms → ~1 ms. Verified: trajectories identical to fresh generation (tests), and T04 episode CSVs byte-identical with and without the cache. **Re-check `reset()` if the crafter version changes.** Disable with `CrafterTask(cache_worlds=False)`.
