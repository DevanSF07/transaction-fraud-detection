.PHONY: all install test train benchmark serve clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTEST = $(VENV)/bin/pytest
UVICORN = $(VENV)/bin/uvicorn

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

benchmark:
	$(PYTHON) -m scripts.run_benchmark

train:
	$(PYTHON) -m scripts.train_pipeline --n-trials 15

evaluate:
	$(PYTHON) -m scripts.evaluate_model

explain:
	$(PYTHON) -m scripts.explain_transaction

serve:
	$(UVICORN) fraud_detection.serving.api:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf __pycache__ .pytest_cache *.egg-info build dist .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
