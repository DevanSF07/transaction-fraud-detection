.PHONY: install test download verify train
install:
	python -m pip install -e '.[dev]'
test:
	python -m pytest -q
download:
	python -m fraudlab.cli download --output data/creditcard.csv
verify:
	python -m fraudlab.cli train --data data/creditcard.csv --config configs/verification.yaml --output runs/verification
train:
	python -m fraudlab.cli train --data data/creditcard.csv --config configs/default.yaml --output runs/full
