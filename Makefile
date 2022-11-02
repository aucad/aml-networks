SHELL := /bin/bash

clean:
	@rm -fr output/
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

all:
	 python -m src -d data/CTU-1-1.csv -a hop --validator IOT23 --save_log
	 python -m src -d data/CTU-1-1.csv -a hop --validator IOT23 --robust --save_log
	 python -m src -d data/CTU-1-1.csv -a zoo --validator IOT23 --save_log
	 python -m src -d data/CTU-1-1.csv -a zoo --validator IOT23 --robust --save_log
	 python -m src -d data/nb15-10K.csv -a hop --validator NB15 --save_log
	 python -m src -d data/nb15-10K.csv -a hop --validator NB15 --robust --save_log
	 python -m src -d data/nb15-10K.csv -a zoo --validator NB15 --save_log
	 python -m src -d data/nb15-10K.csv -a zoo --validator NB15 --robust --save_log
