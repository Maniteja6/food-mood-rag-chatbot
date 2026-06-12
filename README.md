# food-mood-rag-chatbot


# 🍜 MoodBite — Food Recommendation RAG Chatbot

> Tell me how you're feeling. I'll tell you what to eat.

MoodBite is a mood-aware food recommendation chatbot built with **Retrieval-Augmented Generation (RAG)**. It combines a curated food knowledge base with a large language model to suggest meals that match your emotional state — whether you're happy, stressed, tired, romantic, or just hungry.

Built with **Streamlit**, deployable to **Streamlit Community Cloud** in minutes. No login required.

---

## ✨ Features

- 🧠 **Mood-aware retrieval** — maps emotions to food categories using semantic search
- 💬 **Conversational interface** — multi-turn chat with session memory
- 🗃️ **RAG pipeline** — answers grounded in a real food knowledge base, not hallucinations
- 🔄 **Swappable LLMs** — supports OpenAI, Anthropic Claude, and Google Gemini
- 🗄️ **Swappable vector stores** — ChromaDB (default) or FAISS
- 🌐 **No login** — open the app and start chatting
- 🚀 **One-command deploy** — push to GitHub, connect to Streamlit Cloud, done

---

## 🏗️ Architecture

```
User (mood input)
      │
      ▼
┌─────────────────────────────────┐
│        Streamlit Frontend        │  ← app/
│  Mood selector · Chat · Cards   │
└────────────────┬────────────────┘
                 │ query + mood
                 ▼
┌─────────────────────────────────┐
│          RAG Pipeline            │  ← rag/
│  Embed → Retrieve → Prompt      │
│  → LLM → Parse → Respond        │
└───────────┬─────────────────────┘
            │                  ▲
    vector  │                  │ top-K docs
    query   ▼                  │
┌─────────────────────────────────┐
│         Vector Store             │  ← vector_store/ + data/
│   ChromaDB / FAISS + Metadata   │
└─────────────────────────────────┘
                 │ augmented prompt
                 ▼
┌─────────────────────────────────┐
│              LLM                 │  ← llm/
│   GPT-4o / Claude / Gemini      │
└─────────────────────────────────┘
```

For the full detailed architecture, see [ARCHITECTURE.md](./docs/ARCHITECTURE.md).

---

## 📁 Project Structure

```
food-mood-rag-chatbot/
├── app/                    # Streamlit UI
│   ├── main.py             # App entrypoint
│   ├── ui_components.py    # Mood buttons, chat, food cards
│   ├── session_state.py    # st.session_state management
│   └── styles.py           # Custom CSS
├── rag/                    # RAG pipeline
│   ├── pipeline.py         # Orchestrator
│   ├── retriever.py        # Similarity search
│   ├── embeddings.py       # Embedding model wrapper
│   ├── prompt_builder.py   # Prompt construction
│   ├── memory.py           # Conversation history
│   └── response_parser.py  # Format LLM output
├── llm/                    # LLM provider abstraction
│   ├── base.py
│   ├── openai_llm.py
│   └── anthropic_llm.py
├── vector_store/           # Vector DB abstraction
│   ├── chroma_store.py
│   └── faiss_store.py
├── data/
│   ├── raw/                # Source food dataset (CSV/JSON)
│   ├── processed/          # Chunked text
│   └── vector_db/          # Persisted vector index (gitignored)
├── ingestion/              # One-time data ingestion pipeline
│   ├── ingest.py
│   ├── chunker.py
│   └── loader.py
├── config/
│   ├── settings.py         # Reads .env via pydantic-settings
│   └── moods.py            # Supported moods + display config
├── tests/
├── assets/
├── .env.example            # ← copy this to .env
├── .gitignore
├── requirements.txt
├── packages.txt            # System packages for Streamlit Cloud
├── setup.py
├── Makefile
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/Maniteja6/food-mood-rag-chatbot.git
cd food-mood-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
make install
# or manually:
pip install -r requirements.txt && pip install -e .
```

### 4. Set up environment variables

```bash
make env
# This copies .env.example → .env
# Then open .env and fill in your API key(s)
```

At minimum, set:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### 5. Build the vector database

Run this once to embed the food dataset and persist the vector index:

```bash
make ingest
```

### 6. Launch the app

```bash
make run
# Opens at http://localhost:8501
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. **Push your repo to GitHub** (make sure `data/vector_db/` is gitignored — it's rebuilt on deploy)
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**
3. Select your repo, branch, and set **Main file path** to `app/main.py`
4. Under **Advanced settings → Secrets**, add your environment variables (same as `.env`)
5. Click **Deploy** — Streamlit Cloud installs `requirements.txt` and `packages.txt` automatically

> **Note:** The vector DB is rebuilt during each cold start via `ingestion/ingest.py`. For faster cold starts, consider hosting the vector DB on Pinecone or a persistent volume.

---

## 🎛️ Configuration

All configuration is driven by environment variables defined in `.env`. Key options:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM backend: `openai`, `anthropic`, `google` |
| `LLM_MODEL` | `gpt-4o` | Model name for the selected provider |
| `EMBEDDING_PROVIDER` | `openai` | Embedding backend: `openai`, `huggingface` |
| `VECTOR_STORE_PROVIDER` | `chroma` | Vector DB: `chroma` or `faiss` |
| `RETRIEVER_TOP_K` | `5` | Number of food docs to retrieve per query |
| `CONVERSATION_MEMORY_LIMIT` | `10` | Max messages kept in session |
| `DEBUG_MODE` | `false` | Show retrieved chunks and scores in UI |

See `.env.example` for the full list.

---

## 🗃️ Food Dataset Format

Place your food data at `data/raw/food_dataset.csv` with these columns:

| Column | Type | Example |
|---|---|---|
| `id` | string | `dish_001` |
| `name` | string | `Ramen` |
| `description` | string | `A warming Japanese noodle soup...` |
| `cuisine` | string | `Japanese` |
| `moods` | comma-separated | `tired, cold, comforted` |
| `dietary_tags` | comma-separated | `gluten-free optional, dairy-free` |
| `prep_time_mins` | integer | `30` |
| `ingredients` | comma-separated | `noodles, broth, egg, nori` |

And `data/raw/mood_food_mapping.json`:

```json
{
  "happy": ["light", "fresh", "celebratory", "colourful"],
  "stressed": ["comfort food", "warm", "carbs", "indulgent"],
  "tired": ["energising", "protein-rich", "quick"],
  "romantic": ["elegant", "indulgent", "shareable"],
  "adventurous": ["exotic", "spicy", "street food"]
}
```

---

## 🧪 Testing

```bash
make test                        # Run all tests with coverage
make test-file FILE=tests/test_pipeline.py  # Run a single file
```

---

## 🛠️ Development Commands

```bash
make help         # Show all available commands
make install-dev  # Install with dev extras (linting, typing)
make lint         # Run ruff linter
make format       # Auto-format with black + ruff
make typecheck    # Run mypy type checks
make clean        # Remove caches (keeps vector DB)
make clean-all    # Full reset including vector DB
make logs         # Tail the app log file
```

---

## 🤝 Contributing

1. Fork the repo and create a feature branch: `git checkout -b feature/my-feature`
2. Install dev dependencies: `make install-dev`
3. Make your changes and add tests
4. Run `make lint` and `make test` before committing
5. Open a Pull Request

---

## 📄 License

MIT — see [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com/) — RAG orchestration
- [ChromaDB](https://www.trychroma.com/) — vector store
- [Streamlit](https://streamlit.io/) — app framework
- [OpenAI](https://openai.com/) — LLM and embeddings
