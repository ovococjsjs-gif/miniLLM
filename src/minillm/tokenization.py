"""Train and measure multilingual byte-level BPE tokenizers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

from .corpus import CorpusDocument

SPECIAL_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
)
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class TokenizerMetrics:
    documents: int
    utf8_bytes: int
    characters: int
    words: int
    tokens: int
    unknown_tokens: int
    bytes_per_token: float
    characters_per_token: float
    tokens_per_word: float
    unknown_rate: float


def train_byte_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int = 2,
    output_path: str | Path | None = None,
) -> Tokenizer:
    if vocab_size < 512:
        raise ValueError("byte BPE vocabulary must be at least 512")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=None)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def measure_tokenizer(
    tokenizer: Tokenizer, documents: Sequence[CorpusDocument]
) -> TokenizerMetrics:
    byte_count = character_count = word_count = token_count = unknown_count = 0
    unknown_id = tokenizer.token_to_id("<unk>")
    for document in documents:
        byte_count += len(document.text.encode("utf-8"))
        character_count += len(document.text)
        word_count += len(_WORD.findall(document.text))
        ids = tokenizer.encode(document.text).ids
        token_count += len(ids)
        unknown_count += sum(index == unknown_id for index in ids)
    return TokenizerMetrics(
        documents=len(documents),
        utf8_bytes=byte_count,
        characters=character_count,
        words=word_count,
        tokens=token_count,
        unknown_tokens=unknown_count,
        bytes_per_token=byte_count / max(1, token_count),
        characters_per_token=character_count / max(1, token_count),
        tokens_per_word=token_count / max(1, word_count),
        unknown_rate=unknown_count / max(1, token_count),
    )


def measure_by_language(
    tokenizer: Tokenizer, documents: Sequence[CorpusDocument]
) -> dict[str, TokenizerMetrics]:
    groups: dict[str, list[CorpusDocument]] = defaultdict(list)
    for document in documents:
        groups[document.language].append(document)
    return {
        language: measure_tokenizer(tokenizer, group)
        for language, group in sorted(groups.items())
    }


def tokenizer_report(
    tokenizer: Tokenizer,
    documents: Sequence[CorpusDocument],
    *,
    d_model: int,
) -> dict[str, object]:
    vocabulary = tokenizer.get_vocab_size()
    overall = measure_tokenizer(tokenizer, documents)
    languages = measure_by_language(tokenizer, documents)
    return {
        "vocab_size": vocabulary,
        "embedding_parameters_tied": vocabulary * d_model,
        "lm_head_flops_per_token": 2 * vocabulary * d_model,
        "overall": asdict(overall),
        "by_language": {key: asdict(value) for key, value in languages.items()},
    }


def save_report(path: str | Path, report: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
