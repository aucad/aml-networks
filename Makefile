SHELL := /bin/bash

DATA_DIR = ./data

ifndef $ITERS
ITERS:=2 5
endif

ATTACKS = hop zoo
ROBUST = T_ROBUST F_ROBUST

T_ROBUST := --robust
F_ROBUST :=

IOT_OPTIONS = --validator IOT23
NB_OPTIONS = --validator NB15

DS_1 = -d ./data/CTU-1-1.csv $(IOT_OPTIONS)
DS_2 = -d ./data/nb15-10K.csv $(NB_OPTIONS)

DATASETS := DS_1 DS_2

TIME = $(shell date)

all:
	@$(foreach i, $(ITERS), $(foreach r, $(ROBUST), $(foreach attack, $(ATTACKS), $(foreach ds, $(DATASETS),  \
        python3 -m src experiment -a $(attack) $($(ds)) $($(r)) --iter $(i) ; )))) @echo "start $(TIME) end $(shell date)"

valid:
	@$(foreach file, $(wildcard $(DATA_DIR)/CTU*),  \
		python3 -m src validate -d $(file) $(IOT_OPTIONS) --capture;)
	@$(foreach file, $(wildcard $(DATA_DIR)/nb15*), \
		python3 -m src validate -d $(file) $(NB_OPTIONS) --capture;)

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