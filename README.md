# 🖊️ Blog Writing Agent

An AI-powered blog generation system built with **LangGraph**, **LangChain**, and **Streamlit**. Give it a topic — it researches, plans, writes, and illustrates a complete technical blog post autonomously.

🔗 **[Live Demo](https://blogwritingagent-8gdbker6qr2akkxwvn4ur4.streamlit.app/)**

---

## How It Works

The agent runs a multi-node LangGraph pipeline that mirrors how a real technical writer operates:

```
Topic Input
    │
    ▼
 Router ──── decides if web research is needed
    │
    ├── [needs research] ──► Research Node (Tavily)
    │                              │
    │                              ▼
    └──────────────────────► Orchestrator (generates blog plan)
                                   │
                                   ▼
                          Parallel Worker Nodes (one per section)
                                   │
                                   ▼
                             Reducer Subgraph
                          merge → image planning → DALL-E 3 generation
                                   │
                                   ▼
                            Final Markdown Blog
```

### Node Breakdown

| Node | Role |
|------|------|
| **Router** | Classifies topic as `closed_book`, `hybrid`, or `open_book` and decides whether live research is needed |
| **Research** | Runs parallel Tavily searches, deduplicates and filters results by recency |
| **Orchestrator** | Produces a structured `Plan` (5–9 sections) with goals, bullets, word targets, and citation flags |
| **Workers** | Parallel LLM calls — one per section — each grounded in the plan and evidence |
| **Reducer** | Merges sections, decides where images add value, generates them via DALL-E 3, and writes the final `.md` |

---

## Features

- **Adaptive research routing** — evergreen topics skip research; news/latest topics trigger live web search
- **Parallel section writing** — all sections generated concurrently via LangGraph's `Send` API
- **AI image generation** — DALL-E 3 generates contextual diagrams, placed inline automatically
- **Evidence grounding** — open-book mode enforces citation-only claims; no hallucinated sources
- **Streamlit UI** with live progress streaming, plan viewer, evidence table, and image gallery
- **Download options** — export as `.md` or bundled `.zip` with images

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful multi-agent orchestration
- **[LangChain](https://github.com/langchain-ai/langchain)** — LLM abstractions and Tavily tool integration
- **[OpenAI](https://platform.openai.com/)** — `gpt-4.1-mini` for text, `dall-e-3` for images
- **[Tavily](https://tavily.com/)** — real-time web search (optional)
- **[Streamlit](https://streamlit.io/)** — frontend UI
- **[LangSmith](https://smith.langchain.com/)** — tracing and observability

---

## Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key (required)
- Tavily API key (optional — enables web research mode)

### Installation

```bash
git clone https://github.com/Himeshxx04/blog_writing_agent.git
cd blog_writing_agent
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key        # optional
LANGSMITH_API_KEY=your_langsmith_key  # optional
```

### Run

```bash
streamlit run bwa_frontend.py
```

---

## Project Structure

```
blog_writing_agent/
├── bwa_backend.py      # LangGraph pipeline — all nodes, schemas, graph definition
├── bwa_frontend.py     # Streamlit UI — streaming, tabs, download buttons
├── requirements.txt
└── .gitignore
```

---

## Example Output

Input topic: *"Long-term memory in LLMs"*

The agent will:
1. Route as `hybrid` (evergreen + current tools)
2. Search for recent frameworks, papers, and benchmarks
3. Plan 6–8 sections with appropriate citation and code flags
4. Write all sections in parallel
5. Generate 2–3 inline diagrams via DALL-E 3
6. Output a complete, download-ready `.md` file

---

## License

MIT
