.PHONY: venv install init

VENV_DIR := .venv

ifeq ($(OS),Windows_NT)
	PYTHON := py -3
	PIP := $(VENV_DIR)\Scripts\pip.exe
else
	PYTHON := python3
	PIP := $(VENV_DIR)/bin/pip
endif

venv:
	$(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(PIP) install -r requirements.txt

init: install
