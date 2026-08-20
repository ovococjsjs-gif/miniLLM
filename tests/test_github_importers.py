from pathlib import Path

from minillm.corpus import stable_split
from minillm.github_importers import (
    extract_tei_text,
    import_oanc_checkout,
    import_rusdracor_checkout,
    import_ruslit_checkout,
)

REPOSITORY = "https://github.com/example/corpus.git"
REVISION = "a" * 40


def test_oanc_import_is_chunked_and_pinned(tmp_path: Path) -> None:
    directory = tmp_path / "oanc"
    directory.mkdir()
    (directory / "sample.txt").write_text(
        ("A modern English corpus paragraph with useful prose. " * 20)
        + "\n\n"
        + ("A second paragraph remains in the same document. " * 20),
        encoding="utf-8",
    )
    documents = tuple(
        import_oanc_checkout(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
            acquisition_date="2026-08-20",
            sample_parts_per_million=1_000_000,
            maximum_characters=400,
        )
    )
    assert len(documents) > 1
    assert {item.source for item in documents} == {"oanc-github-mirror"}
    assert {item.license for item in documents} == {"OANC-Unrestricted"}
    assert len({stable_split(item) for item in documents}) == 1
    assert all(REVISION in (item.url or "") for item in documents)


def test_ruslit_import_restricts_authors(tmp_path: Path) -> None:
    accepted = tmp_path / "prose" / "Pushkin"
    rejected = tmp_path / "prose" / "Unknown"
    accepted.mkdir(parents=True)
    rejected.mkdir(parents=True)
    (accepted / "Дубровский.txt").write_text(
        "Русский литературный текст. " * 40, encoding="utf-8"
    )
    (rejected / "Recent.txt").write_text("Restricted text. " * 40, encoding="utf-8")
    documents = tuple(
        import_ruslit_checkout(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
            acquisition_date="2026-08-20",
            allowed_authors=frozenset({"Pushkin"}),
            sample_parts_per_million=1_000_000,
            maximum_characters=10_000,
        )
    )
    assert len(documents) == 1
    assert documents[0].metadata["creator"] == "Pushkin"
    assert documents[0].license == "Public-Domain"


def test_rusdracor_tei_extracts_dialogue_and_metadata(tmp_path: Path) -> None:
    tei = tmp_path / "tei"
    tei.mkdir()
    path = tei / "play.xml"
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader><fileDesc><titleStmt><title>Пьеса</title><author>Автор</author>
          </titleStmt></fileDesc></teiHeader>
          <text><body><head>Первое действие</head><sp><speaker>ГЕРОЙ</speaker>
          <p>Первая реплика.</p><stage>Входит.</stage></sp>
          <sp><speaker>ДРУГОЙ</speaker><l>Стихотворная реплика.</l></sp></body></text>
        </TEI>""",
        encoding="utf-8",
    )
    text, metadata = extract_tei_text(path)
    assert "Первая реплика" in text and "Стихотворная реплика" in text
    assert metadata["title"] == "Пьеса" and metadata["creator"] == "Автор"
    documents = tuple(
        import_rusdracor_checkout(
            tmp_path,
            repository=REPOSITORY,
            revision=REVISION,
            acquisition_date="2026-08-20",
            sample_parts_per_million=1_000_000,
            maximum_characters=10_000,
        )
    )
    assert len(documents) == 1
    assert documents[0].source == "rusdracor-github"
    assert documents[0].license == "CC0-1.0"
