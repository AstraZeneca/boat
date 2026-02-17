.PHONY: clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

PYTHON_INTERPRETER=python
ifeq (,$(shell which poetry))
HAS_POETRY=False
else
HAS_POETRY=True
endif


#################################################################################
# COMMANDS                                                                      #
#################################################################################
init:

ifeq (True,$(HAS_POETRY))
	git config filter.strip-notebook-output.clean 'poetry run jupyter nbconvert --clear-output --to=notebook --stdin --stdout '
	poetry install
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg
else
	git config filter.strip-notebook-output.clean 'jupyter nbconvert --clear-output --to=notebook --stdin --stdout '
	@echo ">>>No poetry detected, not installing pre-commit and packages"
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
