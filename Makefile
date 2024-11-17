POETRY_OPTS ?=
POETRY ?= poetry $(POETRY_OPTS)

.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean:
	rm -rf ./dist
	rm -rf ./.mypy_cache
	rm -rf ./aanntwin/__pycache__/

wheel:
	$(POETRY) build --format wheel

source:
	$(POETRY) build --format sdist

build: wheel source;
