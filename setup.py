"""
setup.py — Food Mood RAG Chatbot
Allows the project to be installed as an editable package during development:
    pip install -e .
This makes all internal imports (e.g. `from rag.pipeline import ...`) work
without needing to manipulate sys.path or PYTHONPATH manually.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read pinned dependencies from requirements.txt (excludes dev/test extras)
def parse_requirements(filename: str) -> list[str]:
    lines = (Path(__file__).parent / filename).read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("#")
    ]

setup(
    # -------------------------------------------------------------------------
    # Package metadata
    # -------------------------------------------------------------------------
    name="food-mood-rag-chatbot",
    version="0.1.0",
    description="A mood-aware food recommendation chatbot powered by RAG and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/food-mood-rag-chatbot",
    license="MIT",

    # -------------------------------------------------------------------------
    # Package discovery
    # -------------------------------------------------------------------------
    packages=find_packages(
        exclude=["tests", "tests.*", "data", "assets", "logs"]
    ),
    python_requires=">=3.10",

    # -------------------------------------------------------------------------
    # Dependencies
    # -------------------------------------------------------------------------
    install_requires=parse_requirements("requirements.txt"),

    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "ruff>=0.4.0",
            "mypy>=1.10.0",
            "pre-commit>=3.7.0",
        ],
    },

    # -------------------------------------------------------------------------
    # CLI entry points (optional: run ingestion from terminal)
    # -------------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            "food-ingest=ingestion.ingest:main",
        ],
    },

    # -------------------------------------------------------------------------
    # Package data (include non-.py files)
    # -------------------------------------------------------------------------
    include_package_data=True,
    package_data={
        "": ["*.json", "*.csv", "*.yaml", "*.toml"],
    },

    # -------------------------------------------------------------------------
    # Classifiers (for PyPI, optional)
    # -------------------------------------------------------------------------
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["rag", "llm", "streamlit", "food", "recommendation", "mood", "chatbot"],
)