"""Policy adapters for the shared eval harness (rlproj.eval.harness).

Adapter protocol (every method implements it; numpy in, numpy out):

    reset(n, seed)   Start n parallel episode slots. Clear any recurrent state and seed the
                     policy's action sampling, so a re-run gives the same numbers.
    act(obs, first)  obs: uint8 array (n, 64, 64, 3); first: bool array (n,), True on the
                     first step of a new episode in that slot (reset recurrent state there).
                     Returns an int array (n,) of actions in [0, 17).

Slots the harness has finished with keep receiving their last observation with
first=False; their actions are ignored. Method adapters (sb3.py, dreamer.py) live in their
own modules so this package imports only numpy.
"""

import numpy as np

N_ACTIONS = 17


class Adapter:
    """Base class documenting the protocol above."""

    def reset(self, n, seed):
        raise NotImplementedError

    def act(self, obs, first):
        raise NotImplementedError


class RandomAdapter(Adapter):
    """Uniform random actions; the random baseline and harness tests."""

    def reset(self, n, seed):
        self._n = n
        self._rng = np.random.default_rng(seed)

    def act(self, obs, first):
        return self._rng.integers(N_ACTIONS, size=self._n)
