from __future__ import annotations
import heapq
from typing import Dict, List, Tuple
import networkx as nx
from satcomrl.envs.routing_env import SatelliteNetworkEnv

def build_nx_graph(env: SatelliteNetworkEnv) -> nx.DiGraph:
    G = nx.DiGraph()
    for u, nbrs in env.adj.items():
        for v in nbrs:
            # edge weight = propagation only (queue-unaware)
            G.add_edge(u, v, weight=env.cfg.prop_ms)
    return G

def dijkstra_route(env: SatelliteNetworkEnv) -> List[int]:
    G = build_nx_graph(env)
    path = nx.shortest_path(G, source=env.cfg.src, target=env.cfg.dst, weight="weight")
    return list(path)

def dijkstra_mq_route(env: SatelliteNetworkEnv, mean_queue_ms: float = 2.5) -> List[int]:
    # "God mode": approximate queue info by using mean queue on every edge
    G = nx.DiGraph()
    for u, nbrs in env.adj.items():
        for v in nbrs:
            G.add_edge(u, v, weight=env.cfg.prop_ms + mean_queue_ms)
    path = nx.shortest_path(G, source=env.cfg.src, target=env.cfg.dst, weight="weight")
    return list(path)
