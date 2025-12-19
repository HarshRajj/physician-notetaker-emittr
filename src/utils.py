
import re
import json

def clean_text(text):
    """
    Basic text cleaning.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_transcript(transcript):
    """
    Parses the transcript into speaker segments.
    Returns a list of dicts: [{'speaker': 'Doctor', 'text': '...'}, ...]
    """
    lines = transcript.strip().split('\n')
    conversation = []
    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect speaker (Doctor/Physician/Patient)
        match = re.match(r'^(Doctor|Physician|Patient):\s*(.*)', line, re.IGNORECASE)
        if match:
            # Save previous turn
            if current_speaker:
                conversation.append({
                    'speaker': current_speaker,
                    'text': ' '.join(current_text)
                })
            
            current_speaker = match.group(1).title()
            current_text = [match.group(2)]
        else:
            # Continuation of previous line
            if current_speaker:
                current_text.append(line)
    
    # Save last turn
    if current_speaker:
        conversation.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text)
        })
            
    return conversation

def extract_patient_text(conversation):
    """
    Joins all patient text for analysis.
    """
    return " ".join([turn['text'] for turn in conversation if 'Patient' in turn['speaker']])

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
