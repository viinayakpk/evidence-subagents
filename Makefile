export PATH := $(HOME)/.local/bin:$(PATH)
UV ?= uv

.PHONY: test mock summarize pilot headline local fault-mock fault-judge gym-images gym-check gym-mock gym-clean

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

fault-judge:
	$(UV) run python -c "from evcarry.analysis.fault_injection import catch_table; from evcarry.protocol.judge import build_judge; import json; print(json.dumps(catch_table(build_judge('mock_judge'), include_clean=True), indent=2))"

gym-images:
	bash scripts/build_gym_images.sh

gym-check:
	$(UV) run evcarry gym check

gym-mock:
	$(UV) run evcarry run configs/experiments/gym_mock.yaml
	$(UV) run evcarry summarize runs/gym_mock

gym-clean:
	$(UV) run evcarry gym clean
