.PHONY: env run-notebooks test clean-outputs

# Create Conda environment
env:
	conda env create -f environment.yml

# Run all notebooks with papermill
run-notebooks:
	mkdir -p outputs
	for NB in notebooks/*.ipynb; do \
	  papermill $$NB outputs/$$(basename $${NB%%.ipynb})_output.ipynb; \
	done

# Run tests (smoke tests, <5s)
test:
	pytest tests/ -v --tb=short -q

# Clean generated outputs
clean-outputs:
	rm -rf outputs/*.ipynb
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
