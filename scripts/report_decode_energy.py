#!/usr/bin/env python3
"""Generate a comparable decode-energy proxy report for model configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minillm.config import MiniLLMConfig
from minillm.energy import HardwareEnergyProfile, estimate_decode_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--kv-bits", type=int, default=8)
    parser.add_argument("--memory-pj-per-byte", type=float, default=60.0)
    parser.add_argument("--mac-pj", type=float, default=0.5)
    parser.add_argument("--output", default="results/decode_energy_proxy.json")
    args = parser.parse_args()

    hardware = HardwareEnergyProfile(
        memory_pj_per_byte=args.memory_pj_per_byte,
        mac_pj=args.mac_pj,
    )
    estimates = []
    for path_string in args.configs:
        path = Path(path_string)
        config = MiniLLMConfig.load(path)
        for context in args.contexts:
            estimate = estimate_decode_energy(
                config,
                context_length=context,
                weight_bits=args.weight_bits,
                kv_bits=args.kv_bits,
                hardware=hardware,
            )
            estimates.append({"config": str(path), **estimate.to_dict()})

    report = {
        "hardware": {
            "name": hardware.name,
            "memory_pj_per_byte": hardware.memory_pj_per_byte,
            "mac_pj": hardware.mac_pj,
        },
        "estimates": estimates,
        "limitations": [
            "Fermi estimate, not measured power.",
            "Assumes each active weight and full attention KV history is read once per token.",
            "Charges one read and one write for recurrent state at the same memory-tier price.",
            "Omits activation traffic, cache residency, kernel inefficiency, and sampling.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
