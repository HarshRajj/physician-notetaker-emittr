
from transformers import pipeline

class MedicalSummarizer:
    def __init__(self):
        print("Loading Summarization model...")
        # Falconsai/medical_summarization is a T5 model fine-tuned for medical text
        self.summarizer = pipeline("summarization", model="Falconsai/medical_summarization")

    def summarize(self, text):
        # T5 models have token limits (usually 512). 
        # We'll truncate the input text to ~512 tokens (approx 2000 chars) to avoid warnings.
        safe_text = text[:2500] 

        # Adjust max_length based on input length but keep reasonable bounds
        input_len = len(safe_text.split())
        max_len = min(500, max(50, int(input_len * 0.6)))
        min_len = min(20, int(input_len * 0.2))

        summary_output = self.summarizer(safe_text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary_output[0]['summary_text']
