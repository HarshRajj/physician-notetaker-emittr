
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        print("Loading Sentiment & Intent models...")
        # Using a dedicated sentiment model for doctor-patient interactions if available and working well,
        # otherwise Zero-Shot is a robust backup for custom labels.
        # The user approved AventIQ-AI/sentiment-analysis-for-doctor-patient-interactions
        try:
            self.sentiment_pipeline = pipeline("text-classification", model="AventIQ-AI/sentiment-analysis-for-doctor-patient-interactions")
        except Exception as e:
            print(f"Warning: Specific sentiment model failed to load ({e}). Falling back to Zero-Shot.")
            self.sentiment_pipeline = None

        # Zero-shot for Intent as it's flexible with custom labels
        # Switched to DistilBERT (~260MB) for much faster download
        self.zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

    def analyze_sentiment(self, text):
        """
        Input: Patient's text.
        Output: Anxious, Neutral, or Reassured.
        """
        # If the dedicated model works and outputs compatible labels, use it. 
        # However, checking the model card, labels might differ. 
        # For strict adherence to "Anxious, Neutral, Reassured", Zero-Shot is often safer unless we map the outputs.
        # Let's try Zero-Shot for the specific requested labels to be precise.
        
        labels = ["Anxious", "Neutral", "Reassured"]
        result = self.zero_shot(text, candidate_labels=labels)
        return result['labels'][0]

    def analyze_intent(self, text):
        """
        Input: Patient's text.
        Output: 'Seeking reassurance', 'Reporting symptoms', 'Expressing concern'
        """
        labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern"]
        result = self.zero_shot(text, candidate_labels=labels)
        return result['labels'][0]
