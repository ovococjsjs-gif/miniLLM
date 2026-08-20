from minillm.corpus import CorpusDocument
from minillm.tokenization import measure_tokenizer, train_byte_bpe


def test_byte_bpe_roundtrip_and_metrics() -> None:
    texts = [
        "Маленькая модель вызывает точный инструмент.",
        "A small model calls an exact tool.",
        "Користувач отримує перевірену відповідь.",
        '{"type":"tool_call","tool":"calculator"}',
    ] * 20
    tokenizer = train_byte_bpe(texts, vocab_size=512, min_frequency=1)
    sample = "Привет, Welt! calculator: 17*19 🚀"
    encoded = tokenizer.encode(sample)
    assert tokenizer.decode(encoded.ids) == sample
    documents = [
        CorpusDocument(str(i), text, "test", "CC0", "ru", "test", "2026-08-20")
        for i, text in enumerate(texts[:4])
    ]
    metrics = measure_tokenizer(tokenizer, documents)
    assert metrics.tokens > 0
    assert metrics.unknown_tokens == 0
    assert metrics.bytes_per_token > 0
