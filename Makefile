PYTHON ?= python

.PHONY: test exact smoke check

test:
	$(PYTHON) -m unittest discover -s tests -v

exact:
	$(PYTHON) exact_solver.py

smoke:
	$(PYTHON) tests/smoke_multiprocessing.py
	$(PYTHON) tests/smoke_arena.py

check: test exact
