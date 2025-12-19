
import json
import os
import sys
import logging
import warnings
import contextlib

# 1. Suppress Warnings & Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Context manager to suppress stdout/stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Ensure src modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import parse_transcript, extract_patient_text
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

def print_section(title, content):
    print(f"\n{'-'*60}")
    print(f" {title}")
    print(f"{'-'*60}")
    if isinstance(content, (dict, list)):
        print(json.dumps(content, indent=2))
    else:
        print(content.strip())

def main():
    print("\n🚀 Starting Physician Notetaker Test...")
    
    # 2. Load Data
    input_file = "transcript.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        raw_text = f.read()

    # 3. Initialize Models
    print("   (Loading models... this may take a moment)")
    
    # Delayed import to suppress loading noise
    try:
        with suppress_output():
            from ner import MedicalNER
            from summarizer import MedicalSummarizer
            from sentiment import SentimentAnalyzer
            from soap import SOAPGenerator
            
            ner_system = MedicalNER()
            summarizer_system = MedicalSummarizer()
            sentiment_system = SentimentAnalyzer()
            soap_system = SOAPGenerator()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. Preprocess
    conversation = parse_transcript(raw_text)
    patient_text = extract_patient_text(conversation)
    full_text = " ".join([turn['text'] for turn in conversation])

    # 5. Execute Pipeline
    
    print("\n🔄 Processing Transcript...")
    
    # A. NER
    print("   - Extracting Clinical Entities...")
    with suppress_output():
        entities = ner_system.extract_entities(full_text)
    
    # B. Summarization
    print("   - Generating Clinical Summary...")
    with suppress_output():
        summary_text = summarizer_system.summarize(full_text)
    
    # C. Sentiment
    print("   - Analyzing Sentiment & Intent...")
    with suppress_output():
        patient_sentiment = sentiment_system.analyze_sentiment(patient_text)
        patient_intent = sentiment_system.analyze_intent(patient_text)
    
    # D. SOAP Generation
    print("   - Generating SOAP Note (via OpenAI)...")
    # Don't suppress this one as it has a nice print, or keep suppressed if we want absolute silence
    with suppress_output():
        soap_note = soap_system.generate(full_text, entities, summary_text)

    # 6. Output Results
    
    print("\n" + "="*60)
    print("                  DEMO RESULTS                  ")
    print("="*60)

    print_section("📝 Extracted Entities", entities)
    
    sentiment_output = {
        "Sentiment": patient_sentiment,
        "Intent": patient_intent
    }
    print_section("❤️  Sentiment Analysis", sentiment_output)
    
    print_section("📄 Clinical Summary", summary_text)
    
    print_section("📋 SOAP Note", soap_note)

    # Match the "Expected Output" structure from requirements
    structured_report = {
        "Patient_Name": "Janet Jones", # In a real app, extract this.
        "Symptoms": entities.get("Diseases", []) + entities.get("Body_Parts", []),
        "Diagnosis": entities.get("Diseases", []),
        "Treatment": entities.get("Chemicals", []),
        "Prognosis": "See Summary",
        "Current_Status": "See Summary",
        "Summary": summary_text
    }

    # Save to files for reference
    with open("medical_report.json", "w") as f:
        json.dump(structured_report, f, indent=4)
    with open("sentiment_analysis.json", "w") as f:
        json.dump(sentiment_output, f, indent=4)
    with open("soap_note.json", "w") as f:
        json.dump(soap_note, f, indent=4)

    print(f"\n✅ Test Complete. Artifacts saved:")
    print("   - medical_report.json")
    print("   - sentiment_analysis.json")
    print("   - soap_note.json")

if __name__ == "__main__":
    main()
