from __future__ import annotations

import pytest

from tempprov_audit.cli.run_civic_llm_reranking import parse_score, percentile


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"score": 87}', 87),
        ("```json\n{\"score\": 42}\n```", 42),
        ("score: 63", 63),
        ("99", 99),
        ('{"score": 71.6}', 72),
    ],
)
def test_parse_score_accepts_supported_formats(text: str, expected: int) -> None:
    assert parse_score(text) == expected


@pytest.mark.parametrize("text", ['{"score": 101}', '{"value": 50}', "not a score"])
def test_parse_score_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_score(text)


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.95) == 0.0
