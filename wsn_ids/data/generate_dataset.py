"""
WSN Dataset Generator

Simulates sensor node behavior under normal conditions and during
6 different attack scenarios in Wireless Sensor Networks.

Features generated per node observation:
  - packet_forwarding_rate    : Fraction of received packets forwarded (0-1)
  - packet_drop_ratio         : Fraction of packets dropped (0-1)
  - residual_energy           : Remaining battery percentage (0-100)
  - neighbor_count            : Number of neighbouring nodes detected
  - route_change_frequency    : How often the node changes its route (0-1)
  - signal_strength_variation : Std-dev of RSSI readings (dBm)
  - transmission_interval     : Average time between transmissions (ms)

Labels:
  0 - Normal
  1 - Sinkhole Attack
  2 - Sybil Attack
  3 - Selective Forwarding
  4 - Hello Flood Attack
  5 - DoS Attack
  6 - Node Compromise
"""

import numpy as np
import pandas as pd
from pathlib import Path

from wsn_ids import ATTACK_LABELS, FEATURE_NAMES

RNG_SEED = 42


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def _generate_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Healthy sensor nodes."""
    return np.column_stack([
        _clip(rng.normal(0.92, 0.05, n), 0.75, 1.00),   # forwarding rate
        _clip(rng.normal(0.05, 0.03, n), 0.00, 0.20),   # drop ratio
        _clip(rng.normal(60.0, 15.0, n), 20.0, 100.0),  # residual energy %
        _clip(rng.normal(5.5,  1.5,  n), 2,   10),      # neighbor count
        _clip(rng.normal(0.05, 0.03, n), 0.00, 0.20),   # route change freq
        _clip(rng.normal(3.0,  1.0,  n), 1.0, 6.0),     # signal variation dBm
        _clip(rng.normal(300,  60,   n), 100, 600),      # tx interval ms
    ])


def _generate_sinkhole(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sinkhole: node advertises best route to attract traffic,
    then silently drops or manipulates packets.
    High forwarding-rate claim but elevated drop; many neighbours attracted.
    """
    return np.column_stack([
        _clip(rng.normal(0.55, 0.12, n), 0.25, 0.80),  # drops packets after attracting
        _clip(rng.normal(0.45, 0.12, n), 0.20, 0.75),  # high drop ratio
        _clip(rng.normal(65.0, 10.0, n), 30.0, 95.0),  # normal-ish energy
        _clip(rng.normal(14.0,  3.0, n), 8,   25),     # attracts many neighbours
        _clip(rng.normal(0.08,  0.04, n), 0.01, 0.25), # low route churn (stable lure)
        _clip(rng.normal(4.0,   1.5, n), 1.5,  8.0),   # moderate signal variation
        _clip(rng.normal(290,   55,  n), 100,  550),    # normal interval
    ])


def _generate_sybil(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sybil: one physical node impersonates many identities.
    Extremely high apparent neighbour count; inconsistent signal.
    """
    return np.column_stack([
        _clip(rng.normal(0.80, 0.10, n), 0.50, 0.98),  # plausible forwarding
        _clip(rng.normal(0.20, 0.08, n), 0.05, 0.45),  # moderate drops
        _clip(rng.normal(55.0, 12.0, n), 15.0, 90.0),  # normal energy
        _clip(rng.normal(28.0,  5.0, n), 18,   45),    # very high fake neighbours
        _clip(rng.normal(0.12,  0.05, n), 0.02, 0.30), # slightly elevated route churn
        _clip(rng.normal(8.0,   2.5, n), 3.0,  15.0),  # high signal variation (many IDs)
        _clip(rng.normal(310,   70,  n), 100,  600),    # normal interval
    ])


def _generate_selective_forwarding(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Selective Forwarding: node deliberately drops chosen packets.
    Low forwarding rate and high drop ratio are the key signatures.
    """
    return np.column_stack([
        _clip(rng.normal(0.38, 0.12, n), 0.10, 0.65),  # low forwarding
        _clip(rng.normal(0.62, 0.12, n), 0.35, 0.90),  # high drop
        _clip(rng.normal(58.0, 14.0, n), 20.0, 90.0),  # normal energy
        _clip(rng.normal(5.5,  1.5,  n), 2,   10),     # normal neighbours
        _clip(rng.normal(0.06, 0.03, n), 0.01, 0.18),  # low route churn
        _clip(rng.normal(3.5,  1.0, n),  1.5,  6.5),   # normal signal
        _clip(rng.normal(305,  60,  n),  100,  580),    # normal interval
    ])


def _generate_hello_flood(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Hello Flood: broadcasts high-power HELLO messages to confuse routing.
    Short tx interval (flooding), many apparent neighbours, rapid route changes.
    """
    return np.column_stack([
        _clip(rng.normal(0.75, 0.10, n), 0.50, 0.95),  # mostly forwards
        _clip(rng.normal(0.25, 0.08, n), 0.05, 0.50),  # moderate drop
        _clip(rng.normal(40.0, 12.0, n), 10.0, 70.0),  # drains energy fast
        _clip(rng.normal(18.0,  4.0, n), 10,   35),    # many confused neighbours
        _clip(rng.normal(0.45,  0.10, n), 0.25, 0.70), # high route churn
        _clip(rng.normal(12.0,  3.0, n), 5.0,  20.0),  # very high signal variation
        _clip(rng.normal(35,    12,  n), 10,   80),    # very short tx interval
    ])


def _generate_dos(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    DoS: floods network with excessive traffic, exhausting resources.
    Very short tx interval, depleted energy, high drop, route instability.
    """
    return np.column_stack([
        _clip(rng.normal(0.30, 0.12, n), 0.05, 0.60),  # low forwarding (overwhelmed)
        _clip(rng.normal(0.70, 0.12, n), 0.40, 0.95),  # high drop ratio
        _clip(rng.normal(22.0,  8.0, n), 5.0,  45.0),  # very low residual energy
        _clip(rng.normal(6.0,   2.0, n), 2,    12),    # normal-ish neighbours
        _clip(rng.normal(0.55,  0.12, n), 0.25, 0.80), # high route churn
        _clip(rng.normal(9.0,   2.5, n), 3.5,  16.0),  # high signal variation
        _clip(rng.normal(18,     8,  n), 5,    50),    # very short interval
    ])


def _generate_node_compromise(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Node Compromise: physically captured node behaves erratically or
    relays adversary commands.  Features are highly irregular.
    """
    return np.column_stack([
        _clip(rng.uniform(0.10, 0.90, n), 0.05, 0.95),  # erratic forwarding
        _clip(rng.uniform(0.10, 0.80, n), 0.05, 0.90),  # erratic drops
        _clip(rng.normal(50.0, 20.0, n), 5.0, 95.0),    # unpredictable energy
        _clip(rng.normal(7.0,   3.0, n), 1,   20),      # variable neighbours
        _clip(rng.normal(0.35,  0.15, n), 0.05, 0.70),  # high route change
        _clip(rng.normal(7.5,   3.0, n), 1.5,  18.0),   # erratic signal
        _clip(rng.uniform(20,   500, n), 10,   600),    # random interval
    ])


_GENERATORS = {
    0: _generate_normal,
    1: _generate_sinkhole,
    2: _generate_sybil,
    3: _generate_selective_forwarding,
    4: _generate_hello_flood,
    5: _generate_dos,
    6: _generate_node_compromise,
}


def generate_dataset(
    samples_per_class: int = 500,
    seed: int = RNG_SEED,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic WSN intrusion-detection dataset.

    Parameters
    ----------
    samples_per_class : int
        Number of observations to generate per attack type (and normal).
    seed : int
        Random seed for reproducibility.
    save_path : str | None
        If given, saves the CSV to this path.

    Returns
    -------
    pd.DataFrame with FEATURE_NAMES columns + 'label' + 'attack_type'.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for label, generator in _GENERATORS.items():
        data = generator(samples_per_class, rng)
        df = pd.DataFrame(data, columns=FEATURE_NAMES)
        df["label"] = label
        df["attack_type"] = ATTACK_LABELS[label]
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Enforce sensible dtypes
    dataset["neighbor_count"] = dataset["neighbor_count"].round().astype(int)
    dataset["residual_energy"] = dataset["residual_energy"].round(2)
    dataset["transmission_interval"] = dataset["transmission_interval"].round(1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(save_path, index=False)
        print(f"[Dataset] Saved {len(dataset)} rows → {save_path}")

    return dataset


if __name__ == "__main__":
    df = generate_dataset(
        samples_per_class=500,
        save_path="wsn_ids/data/wsn_dataset.csv",
    )
    print(df["attack_type"].value_counts())
    print(df.describe().round(3))
