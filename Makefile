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
CLS:=tree
endif

DATA_DIR := ./data

ATTACKS := hsj zoo

ROBUST = T_ROBUST F_ROBUST
T_ROBUST := --robust
F_ROBUST :=

IOT_OPTIONS := --validator IOT23
NB_OPTIONS := --validator NB15

DS_1 := -d ./data/CTU-1-1.csv $(IOT_OPTIONS)
DS_2 := -d ./data/nb15-10K.csv $(NB_OPTIONS)

DATASETS := DS_1 DS_2

all:
	@$(foreach i, $(ITERS), $(foreach r, $(ROBUST), $(foreach attack, $(ATTACKS), $(foreach ds, $(DATASETS),  \
        python3 -m src experiment -a $(attack) $($(ds)) $($(r)) --iter $(i) -s $(SAMPLE) -t $(TIMES) -c $(CLS) ; ))))

sample:
	@$(foreach i, $(ITERS), $(foreach r, $(ROBUST), $(foreach attack, $(ATTACKS), \
        python3 -m src experiment -a $(attack) $(DS_2) $($(r)) --iter $(i) -s 50 -t 3  -c $(CLS) ; )))

valid:
	@$(foreach file, $(wildcard $(DATA_DIR)/CTU*),  \
		python3 -m src validate -d $(file) $(IOT_OPTIONS) --capture;)
	@$(foreach file, $(wildcard $(DATA_DIR)/nb15*), \
		python3 -m src validate -d $(file) $(NB_OPTIONS) --capture;)

fast:
	@$(foreach r, $(ROBUST), $(foreach attack, $(ATTACKS), $(foreach ds, $(DATASETS),  \
        python3 -m src experiment -a $(attack) $($(ds)) $($(r)) --iter 0 -s 0 -t 1 -c $(CLS) ; )))

plot:
	@python3 -m src plot output

code_stats:
	@cd src && find . -name '*.py' | xargs wc -l && cd ..

clean:
	@rm -fr output/
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -fr {} +
