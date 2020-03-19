.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
#BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = google-analytics-challenge
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Unpack Dataset
unpack_data: requirements
	$(PYTHON_INTERPRETER) -m src.data.unpack_dataset data/raw data/unpacked

## Recode Dataset
recode_data: requirements
	$(PYTHON_INTERPRETER) -m src.data.recode_dataset data/unpacked data/recoded

## Create X and y on the rolling window basis
create_X_y: requirements
	$(PYTHON_INTERPRETER) -m src.features.create_X_y_roll_windows data/recoded data/final

## Concatenate separate rolling window files for X and y 
concat_X_y: requirements
	$(PYTHON_INTERPRETER) -m src.features.concat_X_y data/final data/final_concat

## Train the model  
train: requirements
	$(PYTHON_INTERPRETER) -m src.models.train_model data/final_concat data/raw models/prediction models/trained

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete



## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help


.PHONY: help
help:
	@echo "$$(tput bold)Make help:$$(tput sgr0)"
	@echo
	@echo "This help repeats the workflow described in README on https://github.com/mzolotko/kaggle-google-analytics-challenge/."
	@echo
	@echo "Setup:"
	@echo "Run make create_environment to create a new virtual environment called google-analytics-challenge."
	@echo "Activate the new environment. If you use virtualenvwrapper, run workon google-analytics-challenge."
	@echo "Run make requirements to install the required packages."
	@echo "Download competition data from Kaggle and put the files train_v2.csv, test_v2.csv, sample_submission_v2.csv in the folder data/raw."
	@echo
	@echo "Analysis scripts:"
	@echo "Run make unpack_data to unpack json fields in the data (this will take several hours)."
	@echo "Run make recode_data to recode some features and to create some new ones." 
	@echo "Run make create_X_y to create time window based aggregated features and target variable values."
	@echo "Run make concat_X_y to concatenate time window based features and target variable values."
	@echo "Run make train to train the model, make predictions and save them in the format ready for submission."
	

