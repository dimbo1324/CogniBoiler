.PHONY: install sync lint format format-check typecheck check test test-cov clean help

UV := uv
PYTHON := $(UV) run python
RUFF := $(UV) run ruff
MYPY := $(UV) run mypy

install:
	$(UV) sync --all-packages

sync:
	$(UV) sync

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

typecheck:
	$(UV) run mypy apps/ shared/

check: lint format-check typecheck

test:
	$(UV) run pytest tests/ -v

test-cov:
	$(UV) run pytest tests/ -v --cov=apps --cov-report=html --cov-report=term-missing

pre-commit-install:
	$(UV) run pre-commit install

pre-commit-run:
	$(UV) run pre-commit run --all-files

pre-commit-update:
	$(UV) run pre-commit autoupdate

clean:
	@if exist .ruff_cache rmdir /s /q .ruff_cache
	@if exist .mypy_cache rmdir /s /q .mypy_cache
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@for /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "Cleaned!"

help:
	@echo ""
	@echo "  install      Установить все зависимости"
	@echo "  sync         Синхронизировать с uv.lock"
	@echo "  lint         Проверить код (ruff check)"
	@echo "  format       Отформатировать код (ruff format)"
	@echo "  format-check Проверить форматирование без изменений"
	@echo "  typecheck    Проверить типы (mypy)"
	@echo "  check        Все проверки вместе"
	@echo "  test         Запустить тесты"
	@echo "  test-cov     Тесты с отчётом покрытия"
	@echo "  clean        Удалить кэши"
	@echo ""

.DEFAULT_GOAL := help
