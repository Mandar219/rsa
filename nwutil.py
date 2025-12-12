from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ---------- Request representation ----------

@dataclass
class Request:
    """Single traffic request.

    Attributes
    ----------
    src : int
        Source node id.
    dst : int
        Destination node id.
    holding_time : int
        Number of time slots the connection stays active.
    """
    src: int
    dst: int
    holding_time: int


def load_requests_csv(path: str | Path) -> List[Request]:
    """Load one request file (100 rows) into Request objects.

    The CSV is expected to have header:
        source,destination,holding_time

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    list[Request]
    """
    df = pd.read_csv(path)
    return [
        Request(int(row["source"]), int(row["destination"]), int(row["holding_time"]))
        for _, row in df.iterrows()
    ]


# ---------- Link representation ----------

class LinkState:
    """Track spectrum usage for a single undirected link.

    We store a vector of length `capacity`.  Each entry is the remaining
    holding time on that wavelength (0 means free).
    """

    def __init__(self, u: int, v: int, capacity: int) -> None:
        self.u = min(u, v)
        self.v = max(u, v)
        self.capacity = int(capacity)
        # spectrum[i] == remaining holding time on wavelength i
        self.spectrum = np.zeros(self.capacity, dtype=np.int32)

    @property
    def endpoints(self) -> Tuple[int, int]:
        return (self.u, self.v)

    def reset(self) -> None:
        """Free all wavelengths on this link."""
        self.spectrum[:] = 0

    def advance_time(self) -> None:
        """Advance logical time by one slot.

        Decrease holding times; any slot that hits 0 becomes free.
        """
        # subtract 1 from positive entries, clip at 0
        self.spectrum[self.spectrum > 0] -= 1

    def is_wavelength_free(self, wavelength: int) -> bool:
        """Check if one wavelength is free on this link."""
        return self.spectrum[wavelength] == 0

    def allocate(self, wavelength: int, holding_time: int) -> None:
        """Allocate a wavelength for a given holding time."""
        self.spectrum[wavelength] = int(holding_time)


# ---------- Network topology and paths ----------

# Edges of the 9-node network.
# This includes a ring plus three shortcut edges.
EDGE_LIST: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 0),
    (1, 5),
    (1, 7),
    (3, 6),
]

# Pre-defined paths P1–P8 (node sequences).
PATHS: List[List[int]] = [
    [0, 1, 2, 3],          # P1 : 0 -> 3
    [0, 8, 7, 6, 3],       # P2 : 0 -> 3
    [0, 1, 5, 4],          # P3 : 0 -> 4
    [0, 8, 7, 6, 3, 4],    # P4 : 0 -> 4
    [7, 1, 2, 3],          # P5 : 7 -> 3
    [7, 6, 3],             # P6 : 7 -> 3
    [7, 1, 5, 4],          # P7 : 7 -> 4
    [7, 6, 3, 4],          # P8 : 7 -> 4
]

# Map (src, dst) -> indices into PATHS list.
PAIR_TO_ACTIONS: Dict[Tuple[int, int], List[int]] = {
    (0, 3): [0, 1],    # P1, P2
    (0, 4): [2, 3],    # P3, P4
    (7, 3): [4, 5],    # P5, P6
    (7, 4): [6, 7],    # P7, P8
}


def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
    """Convert [0,1,2,3] -> [(0,1), (1,2), (2,3)]."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def generate_sample_graph(capacity: int) -> Tuple[nx.Graph, Dict[Tuple[int, int], LinkState]]:
    """Create the sample topology and LinkState objects.

    Parameters
    ----------
    capacity : int
        Number of wavelengths per link.

    Returns
    -------
    (G, links)
        G is a networkx.Graph with the given edges.
        links maps (u, v) with u < v to LinkState objects.
    """
    G = nx.Graph()
    G.add_nodes_from(range(9))
    links: Dict[Tuple[int, int], LinkState] = {}

    for u, v in EDGE_LIST:
        G.add_edge(u, v)
        key = (min(u, v), max(u, v))
        links[key] = LinkState(u, v, capacity)

    return G, links
