PYTHON ?= python

.PHONY: test exact capacity-smoke smoke check

test:
	$(PYTHON) -m unittest discover -s tests -v

exact:
	$(PYTHON) exact_solver.py

capacity-smoke:
	$(PYTHON) capacity_test.py --epochs 1 --batch-size 256 --device cpu \
		--checkpoint /tmp/alphaknot-capacity-smoke.pth.tar

smoke:
	$(PYTHON) tests/smoke_multiprocessing.py
	$(PYTHON) tests/smoke_arena.py

check: test exact
