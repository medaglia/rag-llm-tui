# README

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Authors](#authors)

## Overview

This project is a terminal-based AI assistant for helping with TTRPGs (Table Top Role-Playing Games). 

You can use it to
 - ask questions about your game rules, and get answers from the language model.
 - roll dice and get random numbers.

It's a terminal user interface (TUI) to a RAG Chatbot that will load your game rules into a vector database and 
provide a chat interface to the language model of your choice. 

This project is very much a work in progress.

### Built with

- [Textual](https://textual.textualize.io/)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [D7](https://github.com/NunoCastanho/d7) (for dice rolling)

## Setup

```
uv run --env-file .env main.py
```

### Prerequisites


- [uv](https://docs.astral.sh/uv/)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/rag-llm-tui.git
   cd rag-llm-tui
   ```

2. Copy the example environment file and fill in the values for the environment variables.
   ```bash
   cp env.example .env
   ```

3. Create a directory for PDFs, and add this path to the `PDF_DIR` environment variable.

4. Install and run with uv

   ```bash
   uv run --env-file .env main.py
   ```

## Usage

You can interact with the language model by typing your query in the terminal.

Once you have loaded your PDFs, you can ask questions about them.

## Authors

- [Michael Medaglia](https://github.com/medaglia)
