# 🩺 Physician Notetaker

**Physician Notetaker** is an advanced AI pipeline designed to automate the extraction of clinical insights from doctor-patient conversations. It combines efficient local NLP models with powerful cloud-level reasoning to deliver accurate medical summaries, sentiment analysis, and structured SOAP notes.

---

## 🚀 Key Features

*   **Clinical Entity Extraction (NER)**:
    *   Powered by **scispacy** (`en_ner_bc5cdr_md`).
    *   Extracts **Diseases** (e.g., "Whiplash extraction") and **Chemicals/Treatments** (e.g., "Painkillers") with high precision.
    *   Runs locally and fast on CPU.

*   **Automated SOAP Notes**:
    *   Powered by **OpenAI GPT-4o-mini**.
    *   Converts unstructured transcripts into professional standard **Subjective, Objective, Assessment, and Plan** formats.
    *   Leverages LLM reasoning to filter history vs. current symptoms.

*   **Medical Summarization**:
    *   Powered by **Falconsai/medical_summarization** (T5-base).
    *   Generates concise, abstractive summaries of the patient visit.

*   **Sentiment & Intent Analysis**:
    *   Powered by **DistilBERT** (Zero-Shot).
    *   Classifies patient anxiety levels ("Anxious", "Reassured") and intent ("Reporting symptoms", "Seeking reassurance").

---

## 🏗️ System Architecture

The system is designed for **"Fast Laptop Performance"**:

1.  **Local Processing (Speed)**: Heavy entity extraction and sentiment analysis use lightweight, optimized models (`spacy`, `distilbert`) that run instantly on your machine.
2.  **Cloud Intelligence (Quality)**: Complex reasoning for the SOAP note is offloaded to OpenAI, ensuring high-quality output without requiring massive local GPU resources.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   An [OpenAI API Key](https://platform.openai.com/)

### Step 1: Clone & Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd physician-notetaker-emittr
    ```

2.  **Install Dependencies**:

    **Option A: Using `uv` (Recommended for speed)**
    ```bash
    # This will sync the environment and install heavy models from lockfile
    uv sync
    ```

    **Option B: Standard Pip**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**:
    Create a `.env` file in the root directory:
    ```ini
    OPENAI_API_KEY=sk-your-openai-api-key-here
    ```

---

## 🏃‍♂️ Usage

To run the full pipeline on the sample transcript (`transcript.txt`):

```bash
# If using uv
uv run test.py

# If using standard python
python test.py
```

### Input
The system reads from `transcript.txt` in the root directory. You can edit this file with any doctor-patient dialogue.

### Console Output
The script will display a structured demo of the results directly in your terminal:
```text
🚀 Starting Physician Notetaker Test...

🔄 Processing Transcript...
   - Extracting Clinical Entities...
   - Generating Clinical Summary...
   - Generating SOAP Note...

============================================================
 📋 SOAP Note
============================================================
{
  "Subjective": { ... },
  "Objective": { ... },
  "Assessment": { ... },
  "Plan": { ... }
}
✅ Test Complete.
```

---

## 📂 Output Artifacts

The pipeline automatically saves the following JSON files for integration:

1.  **`soap_note.json`**: The structured clinical note.
2.  **`medical_report.json`**: Contains the extractive entities and the abstractive summary.
3.  **`sentiment_analysis.json`**: JSON object with `Sentiment` (e.g., "Anxious") and `Intent`.

4.  **`REPORT.md`**: Answers to the technical questions in the assignment.

---

## 📂 Project Structure

```
physician-notetaker-emittr/
├── src/
│   ├── ner.py            # Named Entity Recognition logic (Scispacy)
│   ├── sentiment.py      # Sentiment & Intent analysis (DistilBERT)
│   ├── soap.py           # SOAP generation (OpenAI Integration)
│   ├── summarizer.py     # Medical Summarization (T5)
│   ├── pipeline.py       # (Internal) Pipeline orchestrator
│   └── utils.py          # Text parsing helpers
├── test.py               # Main demo script
├── transcript.txt        # Input data file
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```
