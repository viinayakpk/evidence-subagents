"""The 128-tool probe: what it would send, and its recorded result. No network."""

import importlib.util
import json

from evcarry.paths import repo_root


def _probe_module():
    path = repo_root() / "scripts" / "probe_tool_limit.py"
    spec = importlib.util.spec_from_file_location("probe_tool_limit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_target_sits_past_the_128th_tool_and_cannot_be_guessed() -> None:
    probe = _probe_module()
    tools, target = probe.build_tools(probe.N_TOOLS, probe.TARGET_POSITION)
    names = [tool.name for tool in tools]
    assert len(names) == len(set(names)) == 150
    assert names == sorted(names)  # the harness sends tools sorted by name
    assert names.index(target) + 1 > 128
    # Only the target's description answers the prompt; its name is a hash.
    assert [tool.name for tool in tools if "zebra" in tool.description] == [target]
    assert "zebra" not in target


def test_probe_does_nothing_without_hit(capsys) -> None:
    assert _probe_module().main([]) == 0
    assert "not asked to hit" in capsys.readouterr().out


def test_recorded_probe_result_is_internally_consistent() -> None:
    report = json.loads((repo_root() / "docs/evidence/tool_limit_probe_2026-10-24.json").read_text(encoding="utf-8"))
    assert report["tools_sent"] == 150 and report["http_status"] == 200
    assert report["called_target"] is True
    assert report["routing"] == {"allow_fallbacks": False, "order": ["deepinfra/fp8"], "require_parameters": True}
    assert report["response_body"]["provider"] == "DeepInfra"
    assert report["cost_usd_price_table"] < 0.01
