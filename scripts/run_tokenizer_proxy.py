#!/usr/bin/env python3
"""Train several byte-BPE proxy tokenizers and compare multilingual economics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from minillm.analysis import profile_model
from minillm.config import MiniLLMConfig
from minillm.corpus import read_jsonl
from minillm.tokenization import tokenizer_report, train_byte_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/tokenizer-proxy")
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[4096, 8192, 16384]
    )
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--model-config", default="configs/edge_dense_350m.json")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--train-documents-per-language", type=int, default=100)
    parser.add_argument("--output", default="results/tokenizer_proxy.json")
    parser.add_argument("--artifacts", default="runs/tokenizers")
    args = parser.parse_args()

    corpus = Path(args.corpus)
    train_all = list(read_jsonl(corpus / "train.jsonl"))
    by_language = defaultdict(list)
    for document in train_all:
        by_language[document.language].append(document)
    train = [
        document
        for language in sorted(by_language)
        for document in sorted(by_language[language], key=lambda item: item.id)[
            : args.train_documents_per_language
        ]
    ]
    evaluation = list(read_jsonl(corpus / "validation.jsonl")) + list(
        read_jsonl(corpus / "test.jsonl")
    )
    model_config = MiniLLMConfig.load(args.model_config)
    model_profile = profile_model(model_config, context_length=args.context)
    target_lm_head_flops = 2 * model_config.vocab_size * model_config.d_model
    backbone_flops = model_profile.approximate_decode_flops - target_lm_head_flops
    reports = []
    for requested_size in args.vocab_sizes:
        path = Path(args.artifacts) / f"byte_bpe_{requested_size}.json"
        tokenizer = train_byte_bpe(
            (document.text for document in train),
            vocab_size=requested_size,
            output_path=path,
        )
        report = tokenizer_report(tokenizer, evaluation, d_model=args.d_model)
        report["requested_vocab_size"] = requested_size
        report["artifact"] = str(path)
        report["embedding_q4_mib"] = report["embedding_parameters_tied"] / 2**21
        report["estimated_decode_flops_per_utf8_byte"] = (
            backbone_flops + report["lm_head_flops_per_token"]
        ) / report["overall"]["bytes_per_token"]
        reports.append(report)
    payload = {
        "warning": "Small tokenizer proxy snapshot; final vocabulary needs a much larger target-domain corpus.",
        "train_documents_available": len(train_all),
        "train_documents_selected": len(train),
        "train_documents_by_language": {
            language: min(len(documents), args.train_documents_per_language)
            for language, documents in sorted(by_language.items())
        },
        "evaluation_documents": len(evaluation),
        "d_model_for_economics": args.d_model,
        "backbone_config": args.model_config,
        "context_for_compute_estimate": args.context,
        "backbone_flops_excluding_lm_head": backbone_flops,
        "reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
