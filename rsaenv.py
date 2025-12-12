from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nwutil import (
    Request,
    LinkState,
    PATHS,
    PAIR_TO_ACTIONS,
    generate_sample_graph,
    load_requests_csv,
)


class RSAEnv(gym.Env):
    """Gymnasium environment for the Routing and Spectrum Allocation problem."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        request_files: Sequence[str | Path],
        capacity: int = 20,
        max_holding_time: int = 40,
        use_random_file_per_episode: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        request_files : list of paths
            Each file contains one episode of 100 requests.
        capacity : int
            Number of wavelengths on each link.
        max_holding_time : int
            Upper bound used to normalize holding_time in observations.
        use_random_file_per_episode : bool
            If True, each reset() picks a random training file.
            If False (for evaluation), files are used sequentially.
        """
        super().__init__()

        self.request_files = [str(p) for p in request_files]
        if not self.request_files:
            raise ValueError("RSAEnv needs at least one request file.")

        self.capacity = int(capacity)
        self.max_holding_time = int(max_holding_time)
        self.use_random_file = bool(use_random_file_per_episode)
        self._next_file_index = 0  # used when not random

        # Build the graph and link states.
        self.graph, self.links = generate_sample_graph(self.capacity)
        self.num_links = len(self.links)

        # Action: choose one of the 8 candidate paths
        self.action_space = spaces.Discrete(len(PATHS))

        # Observation: flattened link spectra + (src, dst, holding_time)
        obs_len = self.num_links * self.capacity + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32,
        )

        # Per-episode state:
        self.current_requests: List[Request] = []
        self.current_request_index: int = 0
        self.logical_time: int = 0

        # For metrics / objective:
        self.num_blocked_requests: int = 0
        self.num_total_requests: int = 0

    # ---------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Start a new episode.

        - Pick a request file (random or sequential).
        - Reset all LinkStates.
        - Load all 100 requests.
        - Return observation for the first request.
        """
        super().reset(seed=seed)

        # 1. Reset link states and counters.
        for link in self.links.values():
            link.reset()

        self.logical_time = 0
        self.current_request_index = 0
        self.num_blocked_requests = 0
        self.num_total_requests = 0

        # 2. Choose which CSV file to use this episode.
        if self.use_random_file:
            file_path = random.choice(self.request_files)
        else:
            file_path = self.request_files[self._next_file_index % len(self.request_files)]
            self._next_file_index += 1

        self.current_requests = load_requests_csv(file_path)

        # 3. Build and return the initial observation.
        obs = self._build_observation()
        info = {"file": file_path}
        return obs, info

    def step(self, action: int):
        """Process one request with the given action.

        Steps:
        1. Take the current request.
        2. Try to allocate along the chosen path (respecting valid paths).
        3. Advance the logical time on all links.
        4. Move to the next request and construct the new observation.
        """
        assert self.current_requests, "reset() must be called before step()."
        done = False

        # 1. Get current request.
        req = self.current_requests[self.current_request_index]
        self.num_total_requests += 1

        # 2. Attempt allocation using the selected path.
        success = self._try_allocate(action, req)

        if not success:
            self.num_blocked_requests += 1
            reward = -1.0
        else:
            reward = +1.0

        # 3. Advance logical time.
        self.logical_time += 1
        for link in self.links.values():
            link.advance_time()

        # 4. Move to next request or terminate episode.
        self.current_request_index += 1
        if self.current_request_index >= len(self.current_requests):
            done = True

        if done:
            obs = self._empty_observation()
        else:
            obs = self._build_observation()

        info = {
            "time": self.logical_time,
            "blocked": not success,
            "episode_blocking_rate": self.get_episode_blocking_rate(),
        }

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    # ---------- Helper methods ----------

    def _build_observation(self) -> np.ndarray:
        """Construct the flat observation vector for the current request."""
        # Flatten link spectra in a fixed order of edges.
        spectra = []
        for key in sorted(self.links.keys()):
            link = self.links[key]
            # Normalize to [0, 1]
            spec_norm = np.minimum(link.spectrum, self.max_holding_time) / float(
                self.max_holding_time
            )
            spectra.append(spec_norm)

        spectra_flat = np.concatenate(spectra, dtype=np.float32)

        # Append normalized request features.
        req = self.current_requests[self.current_request_index]
        src_norm = req.src / 8.0  # nodes are 0..8
        dst_norm = req.dst / 8.0
        ht_norm = min(req.holding_time, self.max_holding_time) / float(self.max_holding_time)

        extra = np.array([src_norm, dst_norm, ht_norm], dtype=np.float32)

        return np.concatenate([spectra_flat, extra]).astype(np.float32)

    def _empty_observation(self) -> np.ndarray:
        """Observation after the episode is done (not really used by DQN)."""
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _try_allocate(self, action: int, req: Request) -> bool:
        """Try to allocate a lightpath for a single request.

        Returns True if allocated, False if blocked.
        """
        src, dst, ht = req.src, req.dst, req.holding_time

        # 1. Determine which path indices are valid for this (src, dst).
        valid_actions = PAIR_TO_ACTIONS.get((src, dst), [])
        if action not in valid_actions:
            # Path-choice doesn't match this request's node pair -> block.
            return False

        path_nodes = PATHS[action]
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]

        # 2. Find first wavelength index that is free on ALL links.
        for w in range(self.capacity):  # "smallest index first" rule
            all_free = True
            for u, v in path_edges:
                key = (min(u, v), max(u, v))
                link = self.links[key]
                if not link.is_wavelength_free(w):
                    all_free = False
                    break
            if all_free:
                # 3. Allocate this wavelength on all links.
                for u, v in path_edges:
                    key = (min(u, v), max(u, v))
                    link = self.links[key]
                    link.allocate(w, ht)
                return True

        # No wavelength available on each link of the path -> blocked.
        return False

    # ---------- Metrics helpers ----------

    def get_episode_blocking_rate(self) -> float:
        """Return current blocking rate within the episode."""
        if self.num_total_requests == 0:
            return 0.0
        return self.num_blocked_requests / float(self.num_total_requests)
