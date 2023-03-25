SHELL := /bin/bash

ifndef $ITERS
ITERS:=2 5 0
endif

ifndef $SAMPLE
SAMPLE:=0
endif

ifndef $TIMES
TIMES:=1
endif

ifndef $CLS
CLS:=xgb dnn
endif

ifndef $RESDIR
RESDIR:=output
endif

DATA_DIR := ./data

ATTACKS := hsj zoo

ROBUST = T_ROBUST F_ROBUST
T_ROBUST := --robust
F_ROBUST :=

ALWAYS := --resume
IOT_OPTIONS := --validator IOT23 --config config/iot.yaml
NB_OPTIONS := --validator NB15 --config config/unsw.yaml

DS_1 := -d ./data/CTU.csv $(IOT_OPTIONS)
DS_2 := -d ./data/nb15.csv $(NB_OPTIONS)

DATASETS := DS_1 DS_2

all:
	@$(foreach i, $(ITERS), $(foreach c, $(CLS), $(foreach r, $(ROBUST), \
	$(foreach attack, $(ATTACKS), $(foreach ds, $(DATASETS),  \
	python3 -m src experiment $(ALWAYS) -a $(attack) $($(ds)) $($(r)) \
		--iter $(i) -s $(SAMPLE) -t $(TIMES) -c $(c) ; )))))

sample:
	@$(foreach i, $(ITERS), $(foreach c, $(CLS), $(foreach r, $(ROBUST), \
	$(foreach attack, $(ATTACKS), \
	python3 -m src experiment $(ALWAYS) -a $(attack) $(DS_2) $($(r)) \
		--iter 0 -s 50 -t 3 -c $(c) ; ))))

fast:
	@$(foreach r, $(ROBUST), $(foreach c, $(CLS), $(foreach attack, $(ATTACKS), \
	$(foreach ds, $(DATASETS),  \
	python3 -m src experiment $(ALWAYS) -a $(attack) $($(ds)) $($(r)) \
		--iter 5 -s 500 -t 1 -c $(c) ; ))))

valid:
	@$(foreach file, $(wildcard $(DATA_DIR)/CTU*),  \
		python3 -m src validate -d $(file) $(IOT_OPTIONS) --capture;)
	@$(foreach file, $(wildcard $(DATA_DIR)/nb15*), \
		python3 -m src validate -d $(file) $(NB_OPTIONS) --capture;)

plot:
	@python3 -m src plot $(RESDIR)

stats:
	@cd src && find . -name '*.py' | xargs wc -l && cd ..

lint:
	flake8 ./src --count --show-source --statistics

clean:
	@rm -fr output/
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
