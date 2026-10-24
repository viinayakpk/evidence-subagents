export PATH := $(HOME)/.local/bin:$(PATH)
UV ?= uv

.PHONY: test mock summarize pilot headline local fault-mock

test:
	$(UV) run pytest

mock:
	$(UV) run evcarry run configs/experiments/mock_e2e.yaml
	$(UV) run evcarry summarize runs/mock_e2e

summarize:
	$(UV) run evcarry summarize runs/mock_e2e

pilot:
	$(UV) run evcarry run configs/experiments/pilot_deepseek.yaml

headline:
	$(UV) run evcarry run configs/experiments/headline_deepseek.yaml

local:
	$(UV) run evcarry run configs/experiments/local_smoke.yaml

fault-mock:
	$(UV) run python -c "from evcarry.analysis.fault_injection import catch_table; import json; print(json.dumps(catch_table(), indent=2))"
