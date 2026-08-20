#!/usr/bin/env python3
"""Short CPU forward benchmark for matched proxy layer mixes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from minillm.analysis import profile_model
from minillm.config import MiniLLMConfig
from minillm.model import MiniLLM

CONFIGS = {
    "2_attention_4_conv": "configs/proxy_3m.json",
    "3_attention_3_conv": "configs/proxy_hybrid3_3m.json",
    "4_attention_2_conv": "configs/proxy_hybrid4_3m.json",
    "6_attention_0_conv": "configs/proxy_attention_3m.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30, choices=range(1, 301))
    parser.add_argument("--output", default="results/proxy_forward_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(77)
    rows = []
    for name, path in CONFIGS.items():
        config = MiniLLMConfig.load(path)
        model = MiniLLM(config).eval()
        batch = torch.randint(0, config.vocab_size, (4, 128))
        with torch.no_grad():
            for _ in range(5):
                model(batch)
            timings = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                model(batch)
                timings.append(time.perf_counter() - started)
        median = statistics.median(timings)
        profile = profile_model(config, context_length=128)
        rows.append(
            {
                "variant": name,
                "config": path,
                "parameters": profile.unique_parameters,
                "median_milliseconds": median * 1000,
                "tokens_per_second": batch.numel() / median,
                "iterations": args.iterations,
            }
        )
    payload = {
        "warning": "Reference PyTorch CPU benchmark, not a mobile-kernel result.",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "batch": 4,
        "sequence_length": 128,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
