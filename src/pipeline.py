
import os
from utils import clean_text, parse_transcript, extract_patient_text, save_json
from ner import MedicalNER
from summarizer import MedicalSummarizer
from sentiment import SentimentAnalyzer
from soap import SOAPGenerator
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Load Data
    print("--- Starting Physician Notetaker Pipeline ---")
    input_file = "transcript.txt"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        raw_text = f.read()

    # 2. Preprocess
    print("Preprocessing transcript...")
    conversation = parse_transcript(raw_text)
    patient_text = extract_patient_text(conversation)
    full_text = " ".join([turn['text'] for turn in conversation]) # Context for summarization/NER

    # 3. Initialize Models
    # Note: Models are loaded in their classes
    mer_system = MedicalNER()
    summarizer_system = MedicalSummarizer()
    sentiment_system = SentimentAnalyzer()
    soap_system = SOAPGenerator()

    # 4. Run NER
    print("Running NER extraction...")
    entities = mer_system.extract_entities(full_text)
    keywords = mer_system.get_keywords(full_text)

    # 5. Run Summarization
    print("Generating Summary...")
    summary_text = summarizer_system.summarize(full_text)

    # 6. Run Sentiment & Intent Analysis
    print("Analyzing Sentiment & Intent...")
    # Analyze the aggregated patient text or per-turn? 
    # Global sentiment for the visit is usually desired, but per-turn is more granular.
    # We'll do global for the report.
    patient_sentiment = sentiment_system.analyze_sentiment(patient_text)
    patient_intent = sentiment_system.analyze_intent(patient_text)

    # 7. Construct Outputs
    
    # Medical Report JSON
    medical_report = {
        "Patient_Name": "Janet Jones", # Hardcoded or extracted if possible (Spacy PERSON)
        "Symptoms": entities.get("Diseases", []), # Roughly mapping Diseases to Symptoms for this context
        "Diagnosis": entities.get("Diseases", []),
        "Treatment": entities.get("Chemicals", []),
        "Current_Status": "See summary",
        "Prognosis": "See summary",
        "Summary": summary_text,
        "Keywords": keywords
    }

    # Sentiment JSON
    sentiment_report = {
        "Sentiment": patient_sentiment,
        "Intent": patient_intent
    }

    # SOAP Note JSON
    soap_note = soap_system.generate(full_text, entities, summary_text)

    # 8. Save Outputs
    print("Saving outputs...")
    save_json(medical_report, "medical_report.json")
    save_json(sentiment_report, "sentiment_analysis.json")
    save_json(soap_note, "soap_note.json")

    print("\n--- Pipeline Completed Successfully ---")
    print("Generated: medical_report.json, sentiment_analysis.json, soap_note.json")

if __name__ == "__main__":
    main()
