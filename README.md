# noesisnoema-pipeline

## Project Summary

**noesisnoema-pipeline** is an open-source, actively evolving pipeline for building Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) workflows. It provides modular tools primarily focused on two main Google Colab notebook workflows: creating "chunks & embeddings" from documents for RAGpack generation, and converting models to the GGUF format, which is now the primary standard for RAGfish/NoesisNoema-based applications. 

The legacy tokenizer archive workflow remains available for reference and reproducibility but is no longer the main focus. Additionally, a CoreML export workflow (via the exporters submodule) is retained as an experimental and optional feature for iOS/Apple ML use cases.

This repository is an active work-in-progress, intended to evolve alongside the latest LLM and RAG community best practices.

---

## Features

- **Primary: Chunks & Embeddings Notebook:** Create tokenized chunks and embeddings from documents to generate RAGpacks for retrieval-augmented workflows. Every RAGpack (.zip) now includes both `embeddings.npy` **and** `embeddings.csv` files. The CSV output improves cross-platform compatibility, especially for Apple ecosystem projects such as those targeting Swift, macOS, and iOS.
- **Primary: GGUF Conversion Notebook:** Convert LLM models to the GGUF format, the new standard for RAGfish and NoesisNoema pipeline integration.
- **Secondary: Legacy Tokenizer/Archive Workflow:** Tokenize and archive text data for compatibility and reproducibility; maintained for historical reference.
- **Experimental: CoreML Model Export:** Export models to CoreML format for iOS app use via the HuggingFace exporters submodule (optional and not required for mainline workflows).

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

- **notebooks/**: Example Colab notebooks for demonstration and prototyping of primary workflows (chunks & embeddings, GGUF conversion) and legacy tokenizer/archive usage.
- **exporters/**: HuggingFace exporters submodule for optional CoreML model export targeting iOS.
- **exported/**: Directory for storing exported models and assets; included as an empty folder in the repo (with a `.gitkeep` file).

---

## Setup

### Requirements

- **Google Colab**: The recommended environment for all workflows.
- Note: This repository is under active development and its structure and workflows may change as the LLM and RAG ecosystem evolves.

### Installation

In a Colab notebook cell, clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/NoesisNoema/noesisnoema-pipeline.git
# Or, if already cloned:
git submodule update --init --recursive
```

---

## Usage in Google Colab

This repository provides two primary Colab notebook workflows for common RAG and LLM pipeline needs:

### 1. Creating Chunks & Embeddings (RAGpack Generation)

Use the dedicated notebook to process your documents into tokenized chunks and generate embeddings suitable for retrieval-augmented generation workflows. This is the primary recommended approach for preparing textual data.

**RAGpack Output Format:** Every RAGpack `.zip` file now includes the following files:

- `chunks.json`: The tokenized text chunks.
- `embeddings.npy`: The embeddings in NumPy binary format.
- `embeddings.csv`: The embeddings in CSV format.

- `metadata.json`: Metadata associated with the chunks and embeddings.

The inclusion of `embeddings.csv` is intentional to provide easier integration with Swift, macOS, iOS, and other platforms where CSV is more broadly supported than `.npy` files.

### 2. Converting Models to GGUF Format (Model Preparation)

The GGUF conversion notebook handles converting LLM models into the GGUF format, which is now the main standard for RAGfish and NoesisNoema-based applications. This workflow is the current focus for model preparation and pipeline development.

---

### Legacy: Tokenizer Archive Workflow

The tokenizer archive workflow, which tokenizes and archives text data, remains available for reference and reproducibility but is considered legacy and secondary to the above workflows.

Example usage of the legacy tokenizer:

```python
from src.tokenizer import Tokenizer

tokenizer = Tokenizer("bert-base-uncased")
# Use tokenizer.tokenize_file() or tokenizer.tokenize_text() with your own input data
```

---

### Experimental & Optional: CoreML Model Export

The `exporters` submodule supports exporting models to CoreML format for iOS app use. This workflow is optional, experimental, and not required for the mainline pipeline.

Example export command:

```
exporters export coreml --model meta-llama/Llama-3-8B-Instruct --task text-generation --output exported/llama3-8b-coreml
```

> **Note:** The `exported/` directory should exist prior to exporting and is included in this repository as an empty folder (with a `.gitkeep` file) to ensure proper version control.

You can update the exporters submodule with:

```bash
cd exporters
git pull origin main
```

For advanced exporting options and details, refer to HuggingFace's [transformers CLI documentation](https://huggingface.co/docs/transformers/main/en/serialization).

---

## Best Practices for Colab

- Use notebook cells for all processing and exporting steps.
- Focus on the primary workflows: chunks & embeddings and GGUF conversion notebooks.
- Use the legacy tokenizer/archive workflow only if needed for compatibility or reproducibility.
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
