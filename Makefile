.PHONY: test
test:
	(echo "black pete"; \
	echo "isort pete"; \
	echo "ruff pete"; \
	echo "mypy pete"; \
	echo "pytest") | parallel -k
