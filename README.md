# noesisnoema-pipeline

## Project Summary

**noesisnoema-pipeline** is an open-source pipeline for building Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) workflows. It provides a modular toolkit for tokenizing, chunking, and embedding textual data, optimized for use in Google Colab environments. The project is part of the **NoesisNoema** brand, emphasizing clarity, transparency, and extensibility in AI tooling.

The pipeline is designed for seamless cloud-based workflows, focusing on reproducibility, automation, and best practices using notebook cells.

---

## Features

- **RAG/LLM Preprocessing:** Tokenize, chunk, and embed text data for downstream use in RAG and LLM pipelines.
- **Google Colab Optimized:** All workflows are designed to run smoothly in Colab notebooks.
- **CoreML Exporters Included:** Export models for iOS app use via CoreML using the HuggingFace exporters submodule.
- **Extensible:** Modular codebase designed for easy customization and integration.

---

## Directory Structure

```
noesisnoema-pipeline/
├── notebooks/           # Colab notebooks demonstrating workflows
├── exporters/           # HuggingFace exporters submodule for CoreML model export
├── exported/            # Exported models and assets (empty folder included with .gitkeep)
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

- **notebooks/**: Example Colab notebooks for demonstration and prototyping.
- **exporters/**: HuggingFace exporters submodule for CoreML model export for iOS.
- **exported/**: Directory for storing exported models and assets; included as an empty folder in the repo (with a `.gitkeep` file).

---

## Setup

### Requirements

- **Google Colab**: The recommended environment for all workflows.

### Installation

In a Colab notebook cell, clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/NoesisNoema/noesisnoema-pipeline.git
# Or, if already cloned:
git submodule update --init --recursive
```

---

## Usage in Google Colab

### 1. Tokenizing Text Data

In a notebook cell, import and use the tokenizer as needed for your own data sources:

```python
from src.tokenizer import Tokenizer

tokenizer = Tokenizer("bert-base-uncased")
# Use tokenizer.tokenize_file() or tokenizer.tokenize_text() with your own input data
```

### 2. Chunking and Embedding

Embed tokenized data as appropriate for your workflow:

```python
from src.embedder import Embedder

embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
# Use embedder.embed_file() or embedder.embed_text() with your own tokenized data
```

### 3. Exporting Models

The `exporters` submodule is included specifically for exporting models to CoreML format for iOS app use. To export a model, use the CLI with a command like:

```
exporters export coreml --model meta-llama/Llama-3-8B-Instruct --task text-generation --output exported/llama3-8b-coreml
```

> **Note:** The `exported/` directory should exist prior to exporting and is included in this repository as an empty folder (with a `.gitkeep` file) to ensure proper version control.

You can update the exporters submodule to the latest version with:

```bash
cd exporters
git pull origin main
```

For advanced exporting options and details, refer to HuggingFace's [transformers CLI documentation](https://huggingface.co/docs/transformers/main/en/serialization).

---

## Best Practices for Colab

- Use notebook cells for all processing and exporting steps.
- Organize your own data and files outside of this repository structure.
- Avoid committing large or sensitive files; use `.gitignore` to exclude:
  
  ```
  __pycache__/
  .ipynb_checkpoints/
  *.pyc
  *.pyo
  .env
  .venv
  exported/
  *.npy
  *.jsonl
  .DS_Store
  *.log
  ```

- Regularly save your Colab notebooks and exported models.

---

## Contribution

Contributions are welcome! To contribute:

1. Open an issue to discuss your idea or bug report.
2. Fork the repository and create a new branch.
3. Submit a pull request with clear documentation.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- HuggingFace for their excellent Transformers and Datasets libraries.
- The open-source AI community for inspiration and support.
- All contributors to the NoesisNoema ecosystem.
