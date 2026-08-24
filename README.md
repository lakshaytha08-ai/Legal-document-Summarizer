# Legal Document Analyzer

A Streamlit-based legal document analysis application for PDF contract ingestion, clause extraction, summary generation, and evaluation. The app supports a fast mock mode for offline demos and optional deep-learning or external LLM providers for more advanced analysis.

## Features

- PDF upload and preprocessing
- Legal text extraction and chunking
- Clause detection using keyword heuristics and zero-shot classification
- Contract summary generation in offline or hybrid modes
- SQLite-backed document and clause logging
- Optional RAG-style ingestion and vector storage for Q&A/chat workflows
- CUAD-style evaluation utilities for summary and classification metrics

## Project Structure

- `app.py` – main Streamlit application
- `document_processor.py` – PDF extraction, preprocessing, and clause detection helpers
- `nlp_models.py` – summarization and classifier model loading logic
- `database.py` – SQLite database setup and storage functions
- `evaluation.py` – scoring and benchmark utilities
- `style.css` – custom UI styling
- `requirements.txt` – Python dependencies
- `cuad_samples.json` – sample evaluation data or benchmark fixtures

## Tech Stack

- Python
- Streamlit
- PyPDF2
- Pandas
- NumPy
- Scikit-learn
- Transformers
- PyTorch
- SQLite
- Optional RAG / LLM integrations

## Quick Start

1. Open a terminal in the project folder.
2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

5. Open the URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Usage

### Document Analyzer

- Upload a PDF contract
- Adjust chunk size and overlap if necessary
- Use the mock mode for quick testing without large model downloads
- Run document analysis to extract stats, clauses, and summaries

### Optional Model Settings

The sidebar allows you to configure:

- mock vs deep learning summarization
- LLM provider selection (`mock`, `gemini`, `openai`, `ollama`)
- embedding provider selection (`mock`, `gemini`, `local`)
- Gemini and OpenAI API keys for online model access

### Database

The app stores processed results in a local SQLite database named:

```text
legal_docs.db
```

This includes:

- document metadata
- summary text
- clause-level output
- confidence and uncertainty metrics

## Notes

- In mock mode, the app uses fast heuristic logic instead of large model downloads.
- Some model-backed features may require a compatible GPU or internet access to fetch model weights.
- If you are using a local LLM or Ollama setup, configure the relevant environment variables before running analysis.

## License

This project is intended for research, legal workflow experimentation, and local prototype development. Please confirm the appropriate licensing and compliance requirements before using it in production or for client-facing legal workflows.

## Troubleshooting

- If Streamlit cannot launch, verify Python and dependencies are installed correctly.
- If model download fails, switch to mock mode or check network access.
- If PDF parsing looks poor, inspect the source document quality and try preprocessing the file before uploading.
