# Running the quality gate from the repository root; thresholds live in pyproject.toml.
.PHONY: check fast

ROOT := $(shell git rev-parse --show-toplevel)

fast:
	@cd "$(ROOT)" && uvx ruff check . && uvx ruff format --check . && uvx complexipy -q src

check: fast
	@cd "$(ROOT)" && uv run pyright && uv run pytest && uv run vulture \
	  && uv export --format requirements-txt --no-hashes --no-emit-project -q \
	  | uvx pip-audit --no-deps --disable-pip -r /dev/stdin
