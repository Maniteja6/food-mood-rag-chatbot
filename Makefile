# =============================================================================
# Food Mood RAG Chatbot — Makefile
# =============================================================================
# Usage:
#   make install      Install all dependencies
#   make ingest       Build the vector database from food data
#   make run          Start the Streamlit app locally
#   make test         Run the test suite
#   make lint         Run linter and formatter checks
#   make clean        Remove generated files and caches
# =============================================================================

.PHONY: help install install-dev ingest run test lint format clean logs dirs

# Default target
help:
	@echo ""
	@echo "  Food Mood RAG Chatbot — available commands"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install      Install Python dependencies"
	@echo "  make install-dev  Install dependencies + dev extras"
	@echo "  make dirs         Create required project directories"
	@echo "  make ingest       Run data ingestion (builds vector DB)"
	@echo "  make run          Launch the Streamlit app (localhost)"
	@echo "  make test         Run all tests with coverage report"
	@echo "  make lint         Check code with ruff"
	@echo "  make format       Auto-format code with black + ruff"
	@echo "  make logs         Tail the application log file"
	@echo "  make clean        Remove __pycache__, vector DB, logs"
	@echo "  make clean-all    Remove everything including .venv"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo ""

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Create all directories that are gitignored but required at runtime
dirs:
	mkdir -p data/raw data/processed data/vector_db logs assets/food_icons
	@echo "✓ Project directories created"

# Install from requirements.txt (production deps only)
install: dirs
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Dependencies installed"

# Install with dev extras (linting, testing, etc.)
install-dev: dirs
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "✓ Dev dependencies installed"

# Copy .env.example → .env if .env doesn't exist yet
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ .env created from .env.example — fill in your API keys"; \
	else \
		echo "⚠  .env already exists — skipping"; \
	fi

# -----------------------------------------------------------------------------
# Data ingestion
# -----------------------------------------------------------------------------

# Build (or rebuild) the vector database from the food dataset
ingest:
	@echo "Running data ingestion pipeline..."
	python -m ingestion.ingest
	@echo "✓ Vector DB built at data/vector_db/"

# Force re-ingest even if vector DB already exists
ingest-force:
	@echo "Force re-ingesting (existing vector DB will be overwritten)..."
	python -m ingestion.ingest --force
	@echo "✓ Done"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------

# Launch Streamlit app locally
run:
	streamlit run app/main.py \
		--server.port 8501 \
		--server.address localhost \
		--browser.gatherUsageStats false

# Run with hot-reload disabled (useful in production-like testing)
run-prod:
	streamlit run app/main.py \
		--server.port 8501 \
		--server.runOnSave false \
		--browser.gatherUsageStats false

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------

test:
	pytest tests/ \
		-v \
		--cov=. \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-omit="tests/*,setup.py"
	@echo "✓ Tests complete — HTML report at htmlcov/index.html"

# Run a single test file quickly
test-file:
	pytest $(FILE) -v

# -----------------------------------------------------------------------------
# Linting & formatting
# -----------------------------------------------------------------------------

lint:
	ruff check . --select=E,F,W,I
	@echo "✓ Lint passed"

format:
	black . --line-length 88
	ruff check . --fix --select=I
	@echo "✓ Formatting applied"

# Type checking
typecheck:
	mypy app/ rag/ llm/ vector_store/ ingestion/ config/ \
		--ignore-missing-imports \
		--no-strict-optional
	@echo "✓ Type check passed"

# -----------------------------------------------------------------------------
# Logs
# -----------------------------------------------------------------------------

logs:
	tail -f logs/app.log

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

# Remove caches and generated artifacts (keeps vector DB)
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage logs/*.log
	@echo "✓ Clean done (vector DB preserved)"

# Remove everything including the vector database (full reset)
clean-all: clean
	rm -rf data/vector_db data/processed
	@echo "✓ Full clean done — run 'make ingest' to rebuild the vector DB"