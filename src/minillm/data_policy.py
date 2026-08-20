"""Machine-enforced source and license policy for corpus construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import CorpusDocument

_SOURCE_STATUSES = frozenset({"approved", "conditional", "research-only"})
_ATTRIBUTION_FREE = frozenset({"Public-Domain", "CC0-1.0"})


@dataclass(frozen=True)
class SourceSpec:
    id: str
    status: str
    licenses: frozenset[str]
    languages: frozenset[str]
    domains: frozenset[str]
    provenance_required: bool
    url: str
    notes: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SourceSpec:
        source = cls(
            id=str(raw["id"]),
            status=str(raw["status"]),
            licenses=frozenset(str(item) for item in raw["licenses"]),
            languages=frozenset(str(item) for item in raw.get("languages", ())),
            domains=frozenset(str(item) for item in raw.get("domains", ())),
            provenance_required=bool(raw.get("provenance_required", True)),
            url=str(raw.get("url", "")),
            notes=str(raw.get("notes", "")),
        )
        if (
            not source.id
            or source.status not in _SOURCE_STATUSES
            or not source.licenses
        ):
            raise ValueError("source registry entry is invalid")
        return source


@dataclass(frozen=True)
class DataPolicy:
    id: str
    allowed_source_statuses: frozenset[str]
    allowed_licenses: frozenset[str]
    license_aliases: dict[str, str]
    require_provenance_for_attribution: bool
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> DataPolicy:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported data policy schema")
        policy = cls(
            id=str(raw["id"]),
            allowed_source_statuses=frozenset(raw["allowed_source_statuses"]),
            allowed_licenses=frozenset(raw["allowed_licenses"]),
            license_aliases={
                str(key).casefold(): str(value)
                for key, value in raw.get("license_aliases", {}).items()
            },
            require_provenance_for_attribution=bool(
                raw.get("require_provenance_for_attribution", True)
            ),
            notes=str(raw.get("notes", "")),
        )
        if not policy.id or not policy.allowed_source_statuses <= _SOURCE_STATUSES:
            raise ValueError("data policy contains invalid source statuses")
        if not policy.allowed_licenses:
            raise ValueError("data policy must allow at least one license")
        if not set(policy.license_aliases.values()) <= policy.allowed_licenses:
            raise ValueError("license alias points outside policy allowlist")
        return policy

    def canonicalize_license(self, value: str) -> str:
        stripped = value.strip()
        if stripped in self.allowed_licenses:
            return stripped
        return self.license_aliases.get(stripped.casefold(), stripped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "allowed_source_statuses": sorted(self.allowed_source_statuses),
            "allowed_licenses": sorted(self.allowed_licenses),
            "require_provenance_for_attribution": self.require_provenance_for_attribution,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class PolicyDecision:
    accepted: bool
    reason: str | None
    canonical_license: str
    source: SourceSpec | None


class SourceRegistry:
    def __init__(
        self, sources: tuple[SourceSpec, ...], *, snapshot_sha256: str
    ) -> None:
        if len({source.id for source in sources}) != len(sources):
            raise ValueError("source IDs must be unique")
        self.sources = {source.id: source for source in sources}
        self.snapshot_sha256 = snapshot_sha256

    @classmethod
    def load(cls, path: str | Path) -> SourceRegistry:
        registry_path = Path(path)
        content = registry_path.read_bytes()
        raw = json.loads(content)
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported source registry schema")
        sources = tuple(SourceSpec.from_dict(item) for item in raw["sources"])
        return cls(sources, snapshot_sha256=hashlib.sha256(content).hexdigest())

    def decide(self, document: CorpusDocument, policy: DataPolicy) -> PolicyDecision:
        source = self.sources.get(document.source)
        canonical = policy.canonicalize_license(document.license)
        if source is None:
            return PolicyDecision(False, "unknown_source", canonical, None)
        if source.status not in policy.allowed_source_statuses:
            return PolicyDecision(
                False, f"source_status_{source.status}", canonical, source
            )
        if canonical not in source.licenses:
            return PolicyDecision(
                False, "license_not_declared_for_source", canonical, source
            )
        if canonical not in policy.allowed_licenses:
            return PolicyDecision(False, "license_not_allowed", canonical, source)
        if source.languages and document.language not in source.languages:
            return PolicyDecision(
                False, "language_not_declared_for_source", canonical, source
            )
        if source.domains and document.domain not in source.domains:
            return PolicyDecision(
                False, "domain_not_declared_for_source", canonical, source
            )
        requires_provenance = source.provenance_required or (
            policy.require_provenance_for_attribution
            and canonical not in _ATTRIBUTION_FREE
        )
        if requires_provenance and not document.url:
            return PolicyDecision(False, "missing_provenance_url", canonical, source)
        return PolicyDecision(True, None, canonical, source)
