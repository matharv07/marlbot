#!/usr/bin/env python3
"""
mesh_comms.py — Realistic Ad-Hoc Mesh Communication & Peer Consensus Layer.

Simulates decentralized multi-robot inter-agent communication:
  • Physical distance attenuation & range thresholding (e.g. 15m nominal WiFi/mesh link).
  • Line-of-sight vs. non-line-of-sight (NLOS) wall attenuation penalty.
  • Probabilistic packet loss, latency jitter, and message bandwidth budget.
  • Peer-to-peer gossip protocol for delta-compressed CBBA bids and local traversability tiles.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class MeshPacket:
    """Represents a lightweight network packet exchanged across the ad-hoc mesh."""

    def __init__(
        self,
        sender_id: int,
        seq: int,
        packet_type: str,
        payload: Any,
        timestamp: float,
    ):
        self.sender_id = sender_id
        self.seq = seq
        self.packet_type = packet_type  # 'cbba_bid', 'traversability_delta', 'heartbeat', 'alert'
        self.payload = payload
        self.timestamp = timestamp


class DecentralizedMeshNetwork:
    """Simulates physical ad-hoc wireless mesh communication between autonomous quadrupeds."""

    def __init__(
        self,
        max_comm_range: float = 16.0,          # Maximum line-of-sight range (meters)
        nlos_range_penalty: float = 0.55,       # Range reduction factor when occluded by walls
        base_packet_drop_rate: float = 0.02,    # Base packet loss probability
        bandwidth_bytes_per_sec: int = 64000,   # 64 KB/s simulated bandwidth cap per link
    ):
        self.max_comm_range = max_comm_range
        self.nlos_range_penalty = nlos_range_penalty
        self.base_packet_drop_rate = base_packet_drop_rate
        self.bandwidth_bytes_per_sec = bandwidth_bytes_per_sec

        # Inboxes per agent: agent_id -> list of incoming packets
        self.inboxes: Dict[int, List[MeshPacket]] = {}
        self.seq_counters: Dict[int, int] = {}

    def register_agent(self, agent_id: int):
        if agent_id not in self.inboxes:
            self.inboxes[agent_id] = []
            self.seq_counters[agent_id] = 0

    def broadcast(
        self,
        sender_id: int,
        sender_pos: Tuple[float, float],
        agent_positions: Dict[int, Tuple[float, float]],
        packet_type: str,
        payload: Any,
        sim_time: float,
        world_has_los_fn: Optional[Any] = None,
    ) -> int:
        """Transmits a packet to all reachable neighbor agents within wireless range.

        Args:
            sender_id: ID of transmitting robot.
            sender_pos: (x, y) 2D position in meters.
            agent_positions: Dict mapping other active agent_ids to their (x, y) positions.
            packet_type: Packet classification.
            payload: Serializable data or dict payload.
            sim_time: Current simulation timestamp.
            world_has_los_fn: Optional callable (p1, p2) -> bool checking physical line-of-sight.

        Returns:
            Count of neighbors that successfully received the packet.
        """
        self.register_agent(sender_id)
        self.seq_counters[sender_id] += 1
        seq = self.seq_counters[sender_id]

        packet = MeshPacket(
            sender_id=sender_id,
            seq=seq,
            packet_type=packet_type,
            payload=payload,
            timestamp=sim_time,
        )

        successful_deliveries = 0

        for recipient_id, r_pos in agent_positions.items():
            if recipient_id == sender_id:
                continue
            self.register_agent(recipient_id)

            # 1. Physical Euclidean distance check
            dist = math.hypot(sender_pos[0] - r_pos[0], sender_pos[1] - r_pos[1])

            # 2. Line of sight check and effective range calculation
            has_los = True
            if world_has_los_fn is not None:
                has_los = world_has_los_fn(sender_pos, r_pos)

            effective_max_range = self.max_comm_range if has_los else (self.max_comm_range * self.nlos_range_penalty)
            if dist > effective_max_range:
                continue  # Out of radio reception range

            # 3. Path loss & stochastic packet drop
            dist_factor = (dist / effective_max_range) ** 2
            drop_prob = self.base_packet_drop_rate + 0.35 * dist_factor
            if not has_los:
                drop_prob += 0.20  # Additional fading due to multi-path diffraction / wall penetration

            if random.random() < drop_prob:
                continue  # Packet dropped

            # Deliver packet to recipient inbox
            self.inboxes[recipient_id].append(packet)
            successful_deliveries += 1

        return successful_deliveries

    def drain_inbox(self, agent_id: int) -> List[MeshPacket]:
        """Fetches and clears all pending incoming packets for the given agent."""
        if agent_id not in self.inboxes:
            return []
        packets = self.inboxes[agent_id]
        self.inboxes[agent_id] = []
        return packets
