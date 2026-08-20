from pathlib import Path

import torch

from minillm.config import EngramConfig, MiniLLMConfig
from minillm.evaluation import check_completion, load_completion_cases
from minillm.generation import (
    SamplingConfig,
    generate_ids,
    load_model_checkpoint,
    save_inference_checkpoint,
)
from minillm.model import MiniLLM

ROOT = Path(__file__).parents[1]


def cached_config(*, engram: bool = False) -> MiniLLMConfig:
    return MiniLLMConfig(
        vocab_size=48,
        d_model=16,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        ffn_hidden=32,
        max_seq_len=32,
        conv_kernel=3,
        sandwich_norm=True,
        prelude_layers=("attention",),
        core_layers=("conv", "gdn2"),
        coda_layers=("attention",),
        core_repetitions=2,
        max_core_repetitions=2,
        recurrent_input_injection=True,
        mtp_depth=0,
        engram=EngramConfig(
            enabled=engram,
            ngram_orders=(2,),
            num_hash_heads=1,
            table_size=31,
            embedding_dim=4,
            conv_kernel=2,
        ),
    ).validate()


def test_cached_chunks_match_full_forward() -> None:
    torch.manual_seed(8)
    model = MiniLLM(cached_config()).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 11))
    with torch.inference_mode():
        expected = model(token_ids).logits
        first, cache = model.forward_cached(token_ids[:, :4])
        second, cache = model.forward_cached(token_ids[:, 4:7], cache)
        pieces = [first.logits, second.logits]
        for position in range(7, token_ids.shape[1]):
            current, cache = model.forward_cached(
                token_ids[:, position : position + 1], cache
            )
            pieces.append(current.logits)
    actual = torch.cat(pieces, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert cache.token_count == token_ids.shape[1]


def test_generation_is_seeded_and_cache_matches_full_prefix() -> None:
    torch.manual_seed(9)
    model = MiniLLM(cached_config()).eval()
    prompt = [1, 4, 7, 2]
    sampling = SamplingConfig(
        max_new_tokens=6,
        temperature=0.8,
        top_k=12,
        top_p=0.9,
        seed=123,
    )
    first = generate_ids(model, prompt, sampling)
    second = generate_ids(model, prompt, sampling)
    uncached = generate_ids(
        model,
        prompt,
        SamplingConfig(
            max_new_tokens=6,
            temperature=0.8,
            top_k=12,
            top_p=0.9,
            seed=123,
            use_cache=False,
        ),
    )
    assert first.generated_token_ids == second.generated_token_ids
    assert first.generated_token_ids == uncached.generated_token_ids
    assert first.used_cache and not uncached.used_cache


def test_engram_generation_falls_back_to_full_prefix() -> None:
    torch.manual_seed(10)
    model = MiniLLM(cached_config(engram=True)).eval()
    result = generate_ids(
        model,
        [1, 2, 3],
        SamplingConfig(max_new_tokens=2, temperature=0),
    )
    assert len(result.generated_token_ids) == 2
    assert not result.used_cache


def test_bilingual_smoke_suite_is_valid_and_checks_are_transparent() -> None:
    cases = load_completion_cases(ROOT / "eval" / "bilingual_smoke.json")
    assert len(cases) == 8
    english_capital = next(case for case in cases if case.id == "en-capital")
    assert all(check_completion(english_capital, " Paris.").values())
    assert not all(check_completion(english_capital, " London.").values())


def test_inference_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = MiniLLM(cached_config()).eval()
    path = tmp_path / "inference.pt"
    save_inference_checkpoint(path, model, step=17, metadata={"purpose": "test"})
    loaded = load_model_checkpoint(path)
    token_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(token_ids).logits
        actual = loaded.model(token_ids).logits
    torch.testing.assert_close(actual, expected)
    assert loaded.step == 17
    assert loaded.metadata == {"purpose": "test"}
    assert loaded.config.to_dict() == model.config.to_dict()
