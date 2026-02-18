PYTHON ?= python3

.PHONY: venv
venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -e .
	@echo ""
	@echo "Run: source .venv/bin/activate"
