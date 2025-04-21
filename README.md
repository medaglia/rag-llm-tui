# README

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Project Overview](#project-overview)
- [Project Setup](#project-setup)
- [Project Structure](#project-structure)
- [Project Dependencies](#project-dependencies)
- [Project Configuration](#project-configuration)
- [Project Usage](#project-usage)
- [Project License](#project-license)
- [Project Authors](#project-authors)

## Project Overview

This project is a Python-based application that provides a terminal user interface (TUI) for interacting with a language model.

## Project Setup

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

## Project Authors

- [Michael Medaglia](https://github.com/medaglia)
