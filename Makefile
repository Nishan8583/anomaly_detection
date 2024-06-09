VENV_DIR=.test-venv

.PHONY: all
all: fixup-isort fixup-black

.PHONY: activate-venv
activate-venv:
	$(VENV_DIR)/bin/activate

.PHONY: create-venv
create-venv:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r test-requirements.txt

.PHONY: check-isort
check-isort: create-venv activate-venv
	$(VENV_DIR)/bin/isort *.py -c

.PHONY: check-black
check-black: create-venv activate-venv
	$(VENV_DIR)/bin/black *.py --check

.PHONY: fixup-black
fixup-black: create-venv activate-venv
	$(VENV_DIR)/bin/black *.py

.PHONY: check
check: check-isort check-black

.PHONY: fixup-isort
fixup-isort: create-venv activate-venv
	$(VENV_DIR)/bin/isort *.py

