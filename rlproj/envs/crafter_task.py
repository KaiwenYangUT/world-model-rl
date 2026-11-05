"""CrafterTask: Crafter as a gymnasium env with a target achievement, two reward modes,
and fixed world-seed pools for the generalization split (PLAN.md §3).

Imports only numpy, crafter, and gymnasium, so it works in both venvs.

World seeding (checked in crafter 1.8.3, crafter/env.py `Env.reset` and crafter/engine.py
`World.reset`): every reset does `self._episode += 1` and then
`self._world.reset(seed=hash((self._seed, self._episode)) % (2**31 - 1))`, which creates
`world.random = np.random.RandomState(seed)`. World generation (crafter/worldgen.py) and all
object behaviour (crafter/objects.py) draw only from `world.random`. So pinning the private
attributes `_seed = world_seed` and `_episode = 0` before each reset makes the map and the
in-episode dynamics a function of `world_seed` and the actions alone. `hash()` of a tuple of
ints does not depend on PYTHONHASHSEED.

One exception, found by tests/test_crafter_task.py: `World` keeps the objects of each chunk in a
`set` (crafter/engine.py), and `Env._balance_object` picks the creature to despawn with
`creatures[random.randint(0, len(creatures))]`, where `creatures` is built by iterating that set.
Set order follows object ids (memory addresses), so the same seed and actions can diverge within
one process. `_DeterministicCrafterEnv` passes the chunk's objects sorted by position instead.
The RNG draws are unchanged, so the despawn distribution is the same; only the order is fixed.

World cache (T04 measured `Env.reset` at ~0.5 s, ~84% of env time for a random policy, almost all
of it Python world generation in crafter/worldgen.py). The world RNG seed fully determines the
freshly generated world, and our seeds come from fixed pools, so `_DeterministicCrafterEnv`
generates each world once per process, stores a deep copy of the world state and player, and
restores a copy on later resets. The copy includes `world.random`, which every object shares, so
the episode continues exactly as after a fresh generation (tests check this). The restore
mirrors the non-generation lines of crafter 1.8.3 `Env.reset`; re-check it if crafter changes.
"""

import copy

import crafter
import gymnasium as gym
import numpy as np

TRAIN_SEEDS = tuple(range(0, 50))
HELDOUT_SEEDS = tuple(range(10000, 10100))
SPLITS = {"train": TRAIN_SEEDS, "heldout": HELDOUT_SEEDS}
REWARD_MODES = ("crafter", "sparse")
# Success is always logged for these, whatever the target (PLAN.md §3).
TRACKED_ACHIEVEMENTS = ("collect_wood", "place_table", "make_wood_pickaxe")
NOT_UNLOCKED = -1  # first_unlock value for an achievement not unlocked this episode

# (world RNG seed, area) -> deep copy of (world.__dict__, player), taken right after generation.
# Objects point at their World; in the snapshot that pointer is _WORLD_PLACEHOLDER, so the cache
# keeps no env alive, and on restore it is remapped to the restoring env's own World.
_WORLD_CACHE = {}
_WORLD_CACHE_MAX = 1024  # ~150 pool seeds in practice; beyond this, just generate
_WORLD_PLACEHOLDER = object()


def clear_world_cache():
    _WORLD_CACHE.clear()


class _DeterministicCrafterEnv(crafter.Env):
    """crafter.Env with a fixed iteration order over chunk objects and a per-process cache of
    generated worlds (see module docstring)."""

    cache_worlds = True

    def reset(self):
        if not self.cache_worlds:
            return super().reset()
        # Same seed expression as crafter 1.8.3 Env.reset, after its `_episode += 1`.
        key = (hash((self._seed, self._episode + 1)) % (2 ** 31 - 1), tuple(self._area))
        snapshot = _WORLD_CACHE.get(key)
        if snapshot is None:
            obs = super().reset()
            if len(_WORLD_CACHE) < _WORLD_CACHE_MAX:
                _WORLD_CACHE[key] = copy.deepcopy(
                    (self._world.__dict__, self._player),
                    memo={id(self._world): _WORLD_PLACEHOLDER})
            return obs
        # One deepcopy call, so the world and every object keep sharing one copied RNG.
        state, player = copy.deepcopy(snapshot, memo={id(_WORLD_PLACEHOLDER): self._world})
        # The World object itself is kept: the local, item, and semantic views hold it.
        self._world.__dict__.clear()
        self._world.__dict__.update(state)
        # The rest of crafter 1.8.3 Env.reset, minus world.reset and generate_world.
        self._episode += 1
        self._step = 0
        self._update_time()
        self._player = player
        self._last_health = player.health
        self._unlocked = set()
        return self._obs()

    def _balance_object(self, chunk, objs, *args, **kwargs):
        # One object per cell, so position is a unique sort key.
        objs = sorted(objs, key=lambda obj: (int(obj.pos[0]), int(obj.pos[1])))
        return super()._balance_object(chunk, objs, *args, **kwargs)


class CrafterTask(gym.Env):
    """Crafter with a target achievement.

    reward_mode="crafter": the standard Crafter reward (+1 per first unlock of each
        achievement, ±0.1 per health change). The episode ends on death (terminated) or at
        `length` steps (truncated).
    reward_mode="sparse": +1 only on the first unlock of `target`, nothing else. The episode
        also ends (terminated) on that unlock.

    Each reset picks a world seed uniformly from `seed_pool` using the gymnasium `np_random`
    generator. Pass `options={"world_seed": s}` to force one, e.g. for evaluation.

    Extra `info` keys on every step:
        world_seed, episode_step, success (target unlocked so far this episode),
        first_unlock ({achievement: episode step of its first unlock, or -1 if not yet}; ints
            only, so gymnasium vector envs can stack it into arrays),
        crafter_reward (Crafter's reward, whatever reward_mode is).
    Crafter's own keys (inventory, achievements, discount, semantic, player_pos, reward) are
    passed through unchanged.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        target="place_table",
        reward_mode="crafter",
        length=10000,
        seed_pool=TRAIN_SEEDS,
        seed=None,
        size=64,
        area=(64, 64),
        view=(9, 9),
        render_mode=None,
        cache_worlds=True,
    ):
        if target not in crafter.constants.achievements:
            raise ValueError(f"unknown Crafter achievement: {target!r}")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
        if len(seed_pool) == 0:
            raise ValueError("seed_pool is empty")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        self.target = target
        self.reward_mode = reward_mode
        self.length = length
        self.seed_pool = tuple(int(s) for s in seed_pool)
        self.render_mode = render_mode
        self.tracked = tuple(dict.fromkeys(TRACKED_ACHIEVEMENTS + (target,)))
        # The seed passed to the constructor is used for the first reset that has no seed,
        # so make_env(cfg, split, seed) is reproducible without the caller seeding reset().
        self._init_seed = seed
        self._seeded = False
        # Crafter's own seed is overwritten on every reset, so 0 here is a placeholder.
        self._env = _DeterministicCrafterEnv(
            area=area, view=view, size=(size, size), reward=True, length=length, seed=0)
        self._env.cache_worlds = cache_worlds
        self.observation_space = gym.spaces.Box(0, 255, (size, size, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(len(crafter.constants.actions))
        self.world_seed = None
        self._step = 0
        self._first_unlock = {}

    @property
    def action_names(self):
        return list(crafter.constants.actions)

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._seeded:
            seed = self._init_seed
        super().reset(seed=seed)
        self._seeded = True
        options = options or {}
        world_seed = options.get("world_seed")
        if world_seed is None:
            world_seed = self.seed_pool[int(self.np_random.integers(len(self.seed_pool)))]
        self.world_seed = int(world_seed)
        # See the module docstring: this makes the episode depend only on world_seed.
        self._env._seed = self.world_seed
        self._env._episode = 0
        obs = self._env.reset()
        self._step = 0
        self._first_unlock = dict.fromkeys(self.tracked, NOT_UNLOCKED)
        info = {"world_seed": self.world_seed, "episode_step": 0, "success": False,
                "first_unlock": dict(self._first_unlock)}
        return obs, info

    def step(self, action):
        obs, crafter_reward, done, info = self._env.step(int(action))
        self._step += 1
        for name in self.tracked:
            if self._first_unlock[name] == NOT_UNLOCKED and info["achievements"][name] > 0:
                self._first_unlock[name] = self._step
        success = self._first_unlock[self.target] != NOT_UNLOCKED
        dead = info["discount"] == 0
        if self.reward_mode == "sparse":
            just_unlocked = self._first_unlock[self.target] == self._step
            reward = 1.0 if just_unlocked else 0.0
            terminated = dead or just_unlocked
        else:
            reward = float(crafter_reward)
            terminated = dead
        truncated = bool(done) and not terminated
        info.update(
            # Crafter returns the live `player.pos` array; copy it so infos don't alias.
            player_pos=np.array(info["player_pos"]),
            world_seed=self.world_seed,
            episode_step=self._step,
            success=success,
            first_unlock=dict(self._first_unlock),
            crafter_reward=float(crafter_reward),
        )
        return obs, reward, bool(terminated), truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.render()
        return None


def make_env(cfg, split="train", seed=None):
    """Build a CrafterTask from an env config dict and a split name ("train" or "heldout").

    cfg keys (all optional): target, reward_mode, length.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {tuple(SPLITS)}, got {split!r}")
    return CrafterTask(
        target=cfg.get("target", "place_table"),
        reward_mode=cfg.get("reward_mode", "crafter"),
        length=cfg.get("length", 10000),
        seed_pool=SPLITS[split],
        seed=seed,
    )
