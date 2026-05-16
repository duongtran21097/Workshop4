# Meeting AI Assistant

## Problem Statement

Organizations lose valuable decision-making time manually processing meeting notes into summaries, task tickets, and follow-up communications. Meeting AI Assistant automates this pipeline using RAG (Retrieval-Augmented Generation) — uploading meeting notes instantly produces an executive summary, Jira-formatted action items, a testing plan, and a follow-up email, while an AI chatbot provides on-demand answers grounded in the actual meeting content.

---

## Features

| Feature | Description |
|---|---|
| File Upload | `.txt`, `.pdf`, `.jpg`, `.png` supported |
| Sample Data | Built-in meeting note samples (EN & VI) for quick demo |
| Executive Summary | Overview, Key Results, Action Items |
| Jira Tickets | Structured tickets with priority and acceptance criteria via function calling |
| Testing Plan | Test cases derived from Jira tickets |
| Follow-up Email | Auto-generated professional email via function calling |
| AI Chatbot (RAG) | Langchain `ConversationalRetrievalChain` + FAISS semantic retrieval + conversation memory |
| Role Lens | Switch between Manager / Developer / QA perspective |
| Text-to-Speech | gTTS audio playback for summary (English & Vietnamese) |
| Knowledge Base | All chunks with metadata (source, index, char count) from the uploaded file |
| Embedding Visualization | 2D scatter plot of vector embeddings (PCA or t-SNE) |

---

## Project Structure

```
workshop-04/
├── main.py                         # Streamlit app — entry point
│
├── core/                           # RAG pipeline
│   ├── vector_store.py             # FAISS vector store: chunk, embed, index, retrieve with metadata
│   └── chain_builder.py            # Langchain: ConversationalRetrievalChain + VectorStoreRetriever
│
├── utils/                          # Helper modules
│   ├── ocr_utils.py                # OCR: pytesseract (primary) + pdfplumber + EasyOCR (optional fallback)
│   └── tts_utils.py                # Text-to-speech via gTTS
│
├── data/                           # Static data
│   ├── tools.py                    # OpenAI function calling schemas (Jira, Email)
│   └── mock_data.py                # Built-in sample meeting notes (EN & VI)
│
├── testcases/
│   ├── normal/                     # TC_01–TC_15: valid meeting notes (txt, pdf, image)
│   └── abnormal/                   # TC_16–TC_24: edge cases (empty, no chunks, duplicates...)
│
├── .streamlit/
│   └── config.toml                 # Streamlit server config (headless, CORS, light theme)
│
├── docs/                           # Project documentation
├── packages.txt                    # System apt packages for Streamlit Cloud (tesseract-ocr)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python 3.11 pin for Streamlit Cloud
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Azure OpenAI GPT-4o |
| RAG Chain | Langchain `ConversationalRetrievalChain` + `ConversationBufferMemory` |
| Vector Store | FAISS `IndexFlatIP` — in-memory cosine similarity search |
| Embeddings | SentenceTransformers `paraphrase-multilingual-MiniLM-L12-v2` (local, no API key needed) |
| Chunk Metadata | `{"source": filename, "index": i, "chars": n}` per chunk |
| OCR | pytesseract + pdfplumber (images & PDFs); EasyOCR optional local fallback |
| TTS | Google Text-to-Speech (gTTS) |
| Visualization | Plotly + scikit-learn (PCA / t-SNE) |

---

## Mock Data Schema

Built-in samples live in `data/mock_data.py`. Each sample is a plain-text meeting transcript. When stored, the pipeline attaches metadata to every chunk:

```python
{
    "page_content": "The team agreed to migrate the auth service to OAuth2 by end of sprint...",
    "metadata": {
        "source": "Sprint_Planning_Q2.txt",
        "index": 3,
        "chars": 284
    }
}
```

This metadata is passed through to Langchain `Document` objects so the RAG chain can reference the source in generated answers.

---

## Installation

### Prerequisites

**System:** Tesseract OCR (required for image and scanned-PDF upload):

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-vie

# macOS
brew install tesseract
```

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `openai` | Azure OpenAI API client |
| `langchain` | Chain orchestration |
| `langchain-openai` | ChatOpenAI wrapper for Azure |
| `langchain-core` | Base classes: BaseRetriever, Document, PromptTemplate |
| `langchain-community` | Community integrations |
| `langchain-classic` | ConversationalRetrievalChain, ConversationBufferMemory |
| `faiss-cpu` | FAISS `IndexFlatIP` — in-memory vector index |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`, ~80 MB) |
| `numpy` | Array operations |
| `scikit-learn` | PCA and t-SNE dimensionality reduction |
| `plotly` | Interactive 2D embedding visualization |
| `gtts` | Google Text-to-Speech |
| `pdfplumber` | Text extraction from text-based PDFs |
| `pytesseract` | Python wrapper for Tesseract OCR |
| `Pillow` | Image processing |
| `langdetect` | Language detection for TTS |

> **Note:** `easyocr` is **not** in `requirements.txt`. It is auto-detected at runtime as a secondary OCR fallback if `pytesseract` is unavailable. Install manually with `pip install easyocr` only if needed locally.

### Step 2 — Run the app

```bash
streamlit run main.py
```

App available at: `http://localhost:8501`

> First launch takes ~30–60 seconds while `sentence-transformers` downloads the embedding model (~80 MB).

---

## Configuration

Enter your Azure OpenAI credentials in the sidebar:

| Field | Example |
|---|---|
| Azure Endpoint | `https://<your-resource>.openai.azure.com/openai/deployments/GPT-4o/...` |
| API Key | Your Azure OpenAI key |

---

## How to Use

### Basic flow

1. **Load data** — choose a built-in sample from the sidebar dropdown, or upload your own file (`.txt`, `.pdf`, `.jpg`, `.png`).
2. Click **🚀 Run Analysis** — generates Summary, Jira tickets, and Testing plan.
3. Explore results in the tabs.

### Tab overview

```
📊 Reports
  ├── Summary          — Executive summary with TTS playback
  ├── Jira             — Structured tickets with priority & acceptance criteria
  ├── Testing          — Testing plan table from Jira tickets
  ├── 🧠 Knowledge Base — All chunks with source / index / char metadata
  └── 🔵 Embeddings    — 2D vector space visualization (PCA / t-SNE)

🤖 AI Chat            — Langchain RAG chatbot with role lens + conversation memory
⚡ Actions            — Generate follow-up email via function calling
📄 Context            — Raw JSON of all generated content
```

### AI Chat (RAG)

- Select a **lens** (Manager / Developer / QA) to focus responses on decisions, implementation, or quality.
- Each question is embedded with `all-MiniLM-L6-v2` and searched against the FAISS index (cosine similarity).
- Top-3 chunks — with their `source` metadata — are injected as context into the prompt.
- Conversation history is maintained via `ConversationBufferMemory`.
- Switching lens resets the conversation and rebuilds the chain.

### Knowledge Base tab

Displays every chunk stored in the FAISS index with its metadata:

```
source: TC_15_technical_architecture.txt | index: 3 | chars: 412
```

### Embedding Visualization

- Each dot = one text chunk.
- Dots close together share semantic meaning.
- **PCA** — fast, linear. **t-SNE** — slower, better cluster separation (requires ≥ 3 chunks).
- Hover over a dot to read the full chunk text.

> Chunks are created by splitting on blank lines (`\n\n`), minimum 30 characters each.

---

## Deployment (Streamlit Community Cloud)

**Live app:** [https://workshop4-group1.streamlit.app](https://workshop4-group1.streamlit.app)

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) from the `Group1` branch of the GitHub repository [`duongtran21097/Workshop4`](https://github.com/duongtran21097/Workshop4).

### How it works on the cloud

1. Streamlit Cloud clones the `Group1` branch on every deploy.
2. Apt packages in `packages.txt` (`tesseract-ocr`, `tesseract-ocr-vie`) are installed first.
3. Python 3.11 is used (pinned via `runtime.txt`).
4. All Python packages in `requirements.txt` are installed via `uv pip install`.
5. The app starts with `streamlit run main.py`.

All vector storage is **in-memory** (FAISS) — no database setup or persistent disk storage required.

### Cloud vs Local

| Component | Local | Streamlit Cloud |
|---|---|---|
| Vector store | FAISS `IndexFlatIP` (in-memory) | FAISS `IndexFlatIP` (in-memory) |
| OCR | pytesseract + pdfplumber | pytesseract + pdfplumber |
| `packages.txt` | N/A | `tesseract-ocr`, `tesseract-ocr-vie` (apt) |
| Python version | System default | 3.11 (pinned via `runtime.txt`) |
| EasyOCR | Optional fallback if installed | Not available (too large for cloud) |
| Data persistence | RAM only — resets on rerun | RAM only — resets on rerun |

### Limitations on cloud

- **No persistent storage** — uploaded files and analysis results are held in Streamlit session state only; data is lost when the session ends or the app restarts.
- **EasyOCR unavailable** — image OCR relies on pytesseract (Tesseract engine); quality may differ from local EasyOCR results.
- **Cold start** — first load after inactivity takes ~30–60 seconds while the embedding model (`all-MiniLM-L6-v2`, ~80 MB) downloads.

> ChromaDB was replaced by FAISS because ChromaDB's `opentelemetry-*` dependency chain includes a protobuf C extension incompatible with Python 3.14 (Streamlit Cloud default). FAISS carries no such transitive dependencies.

---

## Test Cases

### Normal (`testcases/normal/`) — TC_01 to TC_15

| File | Description |
|---|---|
| TC_01_plain_text.txt | Basic sprint planning notes (EN) |
| TC_02_text_pdf.pdf | Text-based PDF |
| TC_03_image_jpg.jpg | JPG image with text (OCR via pytesseract) |
| TC_04_image_png.png | PNG image with text (OCR via pytesseract) |
| TC_05_short_content.txt | Minimal content (few chunks) |
| TC_06_long_content.txt | Long multi-section meeting |
| TC_07_vietnamese_content.txt | Vietnamese meeting notes |
| TC_08_embedding_test.txt | 9 semantically diverse chunks — best for embedding viz |
| TC_09–TC_12 | Additional EN / VI samples |
| TC_13_many_chunks_EN.txt | ~18 chunks — full QBR in English |
| TC_14_many_chunks_VI.txt | ~18 chunks — full QBR in Vietnamese |
| TC_15_technical_architecture.txt | ~16 chunks — deep technical architecture review |

### Abnormal (`testcases/abnormal/`) — TC_16 to TC_24

| File | Expected behavior |
|---|---|
| TC_16_empty_content.txt | 0 chunks → warning shown in UI |
| TC_17_all_chunks_below_min_length.txt | All paragraphs < 30 chars → 0 chunks stored |
| TC_18_no_paragraph_breaks.txt | No `\n\n` → 1 giant chunk, embedding viz warns |
| TC_19_mixed_language.txt | EN + VI mixed → normal flow, TTS detects dominant language |
| TC_20_special_characters.txt | Emojis, URLs, code snippets, multilingual text |
| TC_21_duplicate_chunks.txt | Repeated content → duplicate vectors cluster together |
| TC_22_single_massive_chunk.txt | 1 chunk ~1500 chars → RAG has limited granularity |
| TC_23_whitespace_only.txt | Only whitespace → 0 chunks stored |
| TC_24_unstructured_freeform.txt | Unstructured notes → AI summary quality degrades |
