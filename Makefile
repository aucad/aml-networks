SHELL := /bin/bash

clean:
	@rm -fr output/
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

all:
	python -m src experiment -d data/CTU-1-1.csv -a hop --validator IOT23
	python -m src experiment -d data/CTU-1-1.csv -a hop --validator IOT23 --robust
	python -m src experiment -d data/CTU-1-1.csv -a zoo --validator IOT23
	python -m src experiment -d data/CTU-1-1.csv -a zoo --validator IOT23 --robust
	python -m src experiment -d data/nb15-10K.csv -a hop --validator NB15
	python -m src experiment -d data/nb15-10K.csv -a hop --validator NB15 --robust
	python -m src experiment -d data/nb15-10K.csv -a zoo --validator NB15
	python -m src experiment -d data/nb15-10K.csv -a zoo --validator NB15 --robust
