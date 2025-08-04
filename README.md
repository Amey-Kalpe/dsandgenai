# Multi-Agent System with LangGraph & SQLite3

This project implements a multi-agent system in Python using [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration and `sqlite3` for persistent storage.

## Features

- **Multi-Agent Coordination:** Agents communicate and collaborate to solve tasks.
- **LangGraph Integration:** Flexible agent workflows and graph-based orchestration.
- **SQLite3 Storage:** Lightweight, file-based database for agent state and logs.
- **Python uv Compatibility:** Designed for fast, async Python environments.

## Getting Started

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (optional, for fast installs)

### Installation

```bash
uv pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

## Project Structure

```
.
├── agents/         # Agent definitions and logic
├── db/             # SQLite3 database and helpers
├── main.py         # Entry point
├── README.md
└── requirements.txt
```

## Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph)
- Python `sqlite3` module
