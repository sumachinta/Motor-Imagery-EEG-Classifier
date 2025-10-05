bootstrap:
	conda env create -f env/env.yml || true
	@echo "Env ready"