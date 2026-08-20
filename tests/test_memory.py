from minillm.memory import EpisodicMemoryStore, MemoryFact


def test_temporal_fact_supersession_and_privacy() -> None:
    with EpisodicMemoryStore() as store:
        old = store.add(
            MemoryFact(
                subject="user",
                predicate="city",
                object="Amsterdam",
                source_turn="turn-10",
                valid_from="2025-01-01",
            )
        )
        new = store.supersede(
            old.id or -1,
            MemoryFact(
                subject="user",
                predicate="city",
                object="Eygelshoven",
                source_turn="turn-20",
                valid_from="2026-01-01",
            ),
        )
        assert store.relation("user", "city")[0].object == "Eygelshoven"
        assert store.relation("user", "city", at="2025-06-01")[0].object == "Amsterdam"
        assert store.get(old.id or -1).superseded_by == new.id

        secret = store.add(
            MemoryFact(
                subject="user",
                predicate="private-note",
                object="never expose this",
                source_turn="turn-21",
                privacy_class="sensitive",
            )
        )
        assert not store.search("never expose", allowed_privacy=("public", "private"))
        assert (
            store.search("never expose", allowed_privacy=("sensitive",))[0].id
            == secret.id
        )


def test_memory_is_deletable_and_searchable() -> None:
    with EpisodicMemoryStore() as store:
        fact = store.add(
            MemoryFact(
                subject="project",
                predicate="goal",
                object="small fast language model",
                source_turn="turn-1",
                confidence=0.9,
            )
        )
        assert store.search("language model")[0].id == fact.id
        store.delete(fact.id or -1)
        assert store.search("language model") == []
