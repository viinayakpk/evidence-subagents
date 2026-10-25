import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evcarry.providers.identity import served_identity
from evcarry.providers.registry import load_models
from evcarry.providers.usage import PriceTable, Rates, TokenUsage, call_cost, cost_usd, is_peak, token_usage_from_body
from evcarry.runs.budget import BudgetGuard, PeakWindowError, SpendCapExceeded


def _price() -> PriceTable:
    return PriceTable.from_mapping(
        {
            "peak_multiplier": 2,
            "weekdays_only": True,
            "peak_windows_utc": [["01:00", "04:00"], ["06:00", "10:00"]],
            "per_million_tokens": {"input_cache_miss": 0.15, "input_cache_hit": 0.003, "output": 0.60},
        }
    )


def test_cost_matches_hand_values() -> None:
    price = _price()
    tokens = TokenUsage(10_000_000, 8_000_000, 2_000_000, 40_000)
    off = datetime(2026, 10, 24, 2, 0, tzinfo=timezone.utc)  # Saturday
    peak = datetime(2026, 10, 21, 2, 0, tzinfo=timezone.utc)  # Wednesday
    assert cost_usd(tokens, price, off) == pytest.approx(0.348)
    assert cost_usd(tokens, price, peak) == pytest.approx(0.696)
    body = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        }
    }
    parsed = token_usage_from_body(body)
    assert cost_usd(parsed, price, off) == pytest.approx(0.00000924)


def test_peak_window_boundaries() -> None:
    price = _price()
    wednesday = datetime(2026, 10, 21, tzinfo=timezone.utc)
    assert is_peak(wednesday.replace(hour=0, minute=59), price) is False
    assert is_peak(wednesday.replace(hour=1, minute=0), price) is True
    assert is_peak(wednesday.replace(hour=2, minute=0), price) is True
    assert is_peak(wednesday.replace(hour=4, minute=0), price) is False
    assert is_peak(datetime(2026, 10, 24, 2, 0, tzinfo=timezone.utc), price) is False


def test_budget_gates() -> None:
    price = _price()
    guard = BudgetGuard(cap_usd=0.5, schedule="offpeak_only", price=price, spent_usd=1.0)
    with pytest.raises(SpendCapExceeded):
        guard.check(billable=True, now=datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc))
    peak_guard = BudgetGuard(cap_usd=10, schedule="offpeak_only", price=price, spent_usd=0)
    with pytest.raises(PeakWindowError):
        peak_guard.check(billable=True, now=datetime(2026, 10, 21, 2, 0, tzinfo=timezone.utc))
    guard.check(billable=False, now=datetime(2026, 10, 21, 2, 0, tzinfo=timezone.utc))


def test_experiment_config_rejects_typos(tmp_path) -> None:
    """A misspelt schedule used to disable the peak-window refusal; a misspelt key fell back to its default."""
    from pydantic import ValidationError

    from evcarry.runs.config import load_experiment

    base = (Path(__file__).resolve().parents[2] / "configs/experiments/pilot_deepseek.yaml").read_text(encoding="utf-8")
    for broken in (
        base.replace("schedule: offpeak_only", "schedule: offpeak-only"),
        base.replace("judge_model: openrouter_ds_1017", "judge_modle: openrouter_ds_1017"),
        base.replace("sampler: risk", "sampler: riks"),
    ):
        assert broken != base
        path = tmp_path / "broken.yaml"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_experiment(path)


def test_openrouter_identity_and_models_yaml() -> None:
    body = {
        "model": "deepseek/deepseek-chat",
        "openrouter_metadata": {
            "requested": "deepseek/deepseek-chat",
            "strategy": "order",
            "attempt": 1,
            "summary": "pinned",
            "endpoints": {
                "available": [
                    {"provider": "DeepSeek", "model": "deepseek/deepseek-chat-v3", "selected": True}
                ]
            },
        },
    }
    identity = served_identity(body, kind="openrouter", configured_model="deepseek/deepseek-chat")
    assert identity.served_provider == "DeepSeek"
    assert identity.served_model == "deepseek/deepseek-chat-v3"
    assert identity.provider_unobserved is False
    missing = served_identity({"model": "deepseek/deepseek-chat"}, kind="openrouter", configured_model="deepseek/deepseek-chat")
    assert missing.served_provider is None
    assert missing.provider_unobserved is True
    models = load_models()
    pinned = models["openrouter_pinned"]
    assert pinned.base_url == "https://openrouter.ai/api/v1"
    assert pinned.api_key_env == "OPENROUTER_API_KEY"
    assert pinned.routing["order"] == ["deepseek"]
    assert pinned.routing["allow_fallbacks"] is False
    assert pinned.routing["require_parameters"] is True
    assert pinned.http_referer
    assert pinned.x_title == "evcarry"


def test_no_real_keys_in_repo(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    banned = []
    # Every file git would commit: tracked, plus untracked files that .gitignore does not
    # exclude. The ignored .env is where keys belong and is not scanned.
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for name in listed:
        path = root / name
        if name.startswith("third_party/") or not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".parquet", ".pyc"}:
            continue
        if path.stat().st_size > 1_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # DeepSeek/OpenAI keys are sk-<alnum>; OpenRouter keys are sk-or-v1-<hex>.
        if re.search(r"sk-(?:or-v1-)?[A-Za-z0-9]{20,}", text):
            banned.append(str(path.relative_to(root)))
    assert banned == []
    example = (root / ".env.example").read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=" in example
    assert "DEEPSEEK_API_KEY=" in example
    # monkeypatch restores the variable; a bare os.environ write leaked into later tests.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-real")
    assert os.environ["OPENROUTER_API_KEY"] == "test-not-real"


def test_an_empty_peak_window_list_means_no_peak() -> None:
    """[] was read as 'not set' and replaced by DeepSeek's windows, so the OpenRouter pilot refused weekday mornings."""
    from datetime import datetime, timezone

    from evcarry.runs.budget import BudgetGuard

    price = load_models()["openrouter_ds_1017"].price_table()
    assert price.peak_windows_utc == ()
    weekday_peak = datetime(2026, 10, 27, 7, 0, tzinfo=timezone.utc)  # a Tuesday, inside DeepSeek's peak
    assert not is_peak(weekday_peak, price)
    BudgetGuard(cap_usd=5, schedule="offpeak_only", price=price).check(billable=True, now=weekday_peak)
    assert is_peak(weekday_peak, load_models()["deepseek_flash"].price_table())


NOON = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def test_a_provider_with_two_endpoints_is_priced_at_the_larger_rate_of_each() -> None:
    table = PriceTable.from_mapping({
        "peak_windows_utc": [],
        "endpoints": {
            "acme/fp8": {"input_cache_miss": 0.1, "input_cache_hit": 0.01, "output": 0.5},
            "acme/us": {"input_cache_miss": 0.2, "input_cache_hit": 0.005, "output": 0.4},
            "cheap-co/fp8": {"input_cache_miss": 0.05, "input_cache_hit": 0.02, "output": 0.1},
        },
    })
    assert table.rates_for("Acme") == (Rates(0.2, 0.01, 0.5), "acme/fp8|acme/us")
    assert table.rates_for("CheapCo") == (Rates(0.05, 0.02, 0.1), "cheap-co/fp8")
    # Not in the table, or not observed: the larger rate of each component over the table.
    assert table.rates_for("Other") == table.rates_for(None) == (Rates(0.2, 0.02, 0.5), "upper_bound")


def test_the_counted_cost_is_the_larger_of_the_served_price_and_openrouters_cost() -> None:
    table = PriceTable.from_mapping({"peak_windows_utc": [], "per_million_tokens": {"input_cache_miss": 1, "input_cache_hit": 0.1, "output": 2}})
    under = call_cost(TokenUsage(1_000_000, 0, 1_000_000, 0, provider_reported_cost_usd=0.5), table, NOON, "Acme")
    assert under.usd == 1.0 and under.table_usd == 1.0 and under.reported_usd == 0.5
    over = call_cost(TokenUsage(1_000_000, 0, 1_000_000, 0, provider_reported_cost_usd=1.5), table, NOON, "Acme")
    assert over.usd == 1.5 and over.fields()["price_table_cost_usd"] == 1.0
    unreported = call_cost(TokenUsage(1_000_000, 0, 1_000_000, 0), table, NOON, "Acme")
    assert unreported.usd == 1.0 and unreported.reported_usd is None


def test_a_single_endpoint_table_prices_every_provider_alike() -> None:
    table = PriceTable.from_mapping({"peak_windows_utc": [], "per_million_tokens": {"input_cache_miss": 1, "input_cache_hit": 0.1, "output": 2}})
    tokens = TokenUsage(2_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert cost_usd(tokens, table, NOON) == cost_usd(tokens, table, NOON, "Anyone") == pytest.approx(3.1)
