
import os
import json
from openai import OpenAI

class SOAPGenerator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

    def generate(self, conversation_text, ner_entities=None, summary=None):
        """
        Generates a SOAP note using OpenAI LLM.
        ned_entities and summary are kept in signature for compatibility but might not be strictly needed
        if the LLM does the extraction itself.
        """
        if not self.client:
            return {"Error": "OpenAI API Key missing. Cannot generate SOAP note."}

        print("Generating SOAP note using OpenAI (this may take a moment)...")
        
        system_prompt = """
        You are an expert medical scribe. 
        Your task is to convert the provided Doctor-Patient transcript into a professional SOAP Note in JSON format.
        
        The JSON should strictly follow this structure:
        {
          "Subjective": {
            "Chief_Complaint": "...",
            "History_of_Present_Illness": "..."
          },
          "Objective": {
            "Physical_Exam": "...",
            "Observations": "..."
          },
          "Assessment": {
            "Diagnosis": "...",
            "Analysis": "..."
          },
          "Plan": {
            "Treatment": "...",
            "Follow_Up": "..."
          }
        }
        Do not include markdown formatting (like ```json). Return raw JSON only.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", # Fast and capable
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transcript:\n{conversation_text}"}
                ],
                temperature=0.2, # Low temperature for consistent formatting
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return {"Error": str(e)}
