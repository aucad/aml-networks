SHELL := /bin/bash

DATA_DIR = ./data

ATTACKS = hop zoo

IOT_DATASETS = CTU-1-1
IOT_OPTIONS = --validator IOT23

NB15_DATASETS = nb15-10K
NB_OPTIONS = --validator NB15

all: non_robust robust

non_robust:
	$(foreach attack, $(ATTACKS), $(foreach ds, $(IOT_DATASETS),  \
        python -m src experiment -a $(attack) -d ./data/$(ds).csv $(IOT_OPTIONS) ; ))
	$(foreach attack, $(ATTACKS), $(foreach ds, $(NB15_DATASETS),  \
        python -m src experiment -a $(attack) -d ./data/$(ds).csv $(NB_OPTIONS) ; ))

robust:
	$(foreach attack, $(ATTACKS), $(foreach ds, $(IOT_DATASETS),  \
        python -m src experiment -a $(attack) -d ./data/$(ds).csv $(IOT_OPTIONS) --robust ; ))
	$(foreach attack, $(ATTACKS), $(foreach ds, $(NB15_DATASETS),  \
        python -m src experiment -a $(attack) -d ./data/$(ds).csv $(NB_OPTIONS) --robust ; ))

valid:
	@$(foreach file, $(wildcard $(DATA_DIR)/CTU*),  \
		python -m src validate -d $(file) $(IOT_OPTIONS) --capture;)
	@$(foreach file, $(wildcard $(DATA_DIR)/nb15*), \
		python -m src validate -d $(file) $(NB_OPTIONS) --capture;)

code_stats:
	@cd src && find . -name '*.py' | xargs wc -l && cd ..

clean:
	@rm -fr output/
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +

lint:
	flake8 ./src --count --show-source --statistics