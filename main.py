"""
WSN-IDS Entry Point

Run the full pipeline:
    python main.py

Or customise:
    python main.py --samples 1000 --results my_results
"""

import argparse
from wsn_ids.ids import WSNIDS


def parse_args():
    parser = argparse.ArgumentParser(
        description="WSN Intrusion Detection System — ML Pipeline"
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Samples per attack class (default: 500)"
    )
    parser.add_argument(
        "--results", type=str, default="results",
        help="Output directory (default: results/)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def demo_inference(ids: WSNIDS) -> None:
    """Demonstrate real-time node classification."""
    print("\n--- Real-Time Inference Demo ---")

    scenarios = [
        {
            "name": "Healthy Sensor",
            # pkt_fwd  drop  energy  nbrs  rt_chg  sig_var  tx_int
            "obs": [0.95, 0.04,  65.0,   5,   0.04,   2.5,    310],
        },
        {
            "name": "Suspected Sinkhole",
            "obs": [0.52, 0.48,  70.0,  15,   0.07,   4.0,    295],
        },
        {
            "name": "Suspected Sybil",
            "obs": [0.82, 0.18,  58.0,  32,   0.11,   9.5,    305],
        },
        {
            "name": "Suspected DoS",
            "obs": [0.28, 0.72,  18.0,   6,   0.58,   9.2,     20],
        },
        {
            "name": "Hello Flood",
            "obs": [0.76, 0.24,  38.0,  20,   0.48,  13.0,     30],
        },
    ]

    for s in scenarios:
        alert = ids.alert(s["obs"])
        print(f"\n  Node: {s['name']}")
        print(f"  {alert}")


if __name__ == "__main__":
    args = parse_args()

    ids = WSNIDS(
        samples_per_class=args.samples,
        results_dir=args.results,
        random_state=args.seed,
    )
    ids.run()
    demo_inference(ids)
