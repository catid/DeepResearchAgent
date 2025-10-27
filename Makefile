SHELL=/usr/bin/env bash

UV=uv
PYTHON_VERSION=3.13
VENV_DIR=.venv
PYTHON=$(VENV_DIR)/bin/python
UV_PIP=$(UV) pip --python $(PYTHON)

# Default goal
.DEFAULT_GOAL := help

.PHONY: clean
clean:
	rm -rf $(VENV_DIR)

.PHONY: venv
venv:
	@echo "Creating virtual environment with uv"
	$(UV) venv --python $(PYTHON_VERSION) $(VENV_DIR)

.PHONY: install
install: venv
	@echo "Installing dependencies"
	$(UV_PIP) install poetry
	$(UV_PIP) install 'markitdown[all]'
	$(UV_PIP) install "browser-use[memory]"==0.1.48

	@echo install playwright
	$(UV_PIP) install playwright
	$(VENV_DIR)/bin/playwright install chromium --with-deps --no-shell

	@echo install dependencies
	$(VENV_DIR)/bin/poetry install

	@echo install xlrd
	$(UV_PIP) install xlrd==2.0.1

.PHONY: install-requirements
install-requirements: venv
	@echo "Installing dependencies"
	$(UV_PIP) install poetry
	$(UV_PIP) install 'markitdown[all]'
	$(UV_PIP) install "browser-use[memory]"==0.1.48

	@echo install playwright
	$(UV_PIP) install playwright
	$(VENV_DIR)/bin/playwright install chromium --with-deps --no-shell

	@echo install dependencies
	$(UV_PIP) install -r requirements.txt

	@echo install xlrd
	$(UV_PIP) install xlrd==2.0.1

.PHONY: update
update:
	$(VENV_DIR)/bin/poetry update

# 🛠️ Show available Makefile commands
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make venv        - Create uv virtual environment"
	@echo "  make clean       - Remove uv virtual environment"
	@echo "  make install     - Install dependencies using Poetry inside uv venv"
	@echo "  make update      - Update dependencies using Poetry"
