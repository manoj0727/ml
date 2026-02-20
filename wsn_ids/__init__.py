"""
WSN-IDS: Machine Learning Based Intrusion Detection System
for Wireless Sensor Networks
"""

ATTACK_LABELS = {
    0: "Normal",
    1: "Sinkhole",
    2: "Sybil",
    3: "Selective Forwarding",
    4: "Hello Flood",
    5: "DoS",
    6: "Node Compromise",
}

FEATURE_NAMES = [
    "packet_forwarding_rate",
    "packet_drop_ratio",
    "residual_energy",
    "neighbor_count",
    "route_change_frequency",
    "signal_strength_variation",
    "transmission_interval",
]
