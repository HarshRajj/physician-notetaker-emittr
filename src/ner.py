import spacy
import warnings

# Suppress warnings if necessary
warnings.filterwarnings("ignore")

class MedicalNER:
    def __init__(self):
        print("Loading NER models... (this might take a few seconds)")
        try:
            self.nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md") # Disease, Chemical
            self.nlp_bionlp = spacy.load("en_ner_bionlp13cg_md") # Cancer, Organ, Tissue, etc.
        except OSError:
            print("Error: scispacy models not found. Please ensure they are installed.")
            raise

    def extract_entities(self, text):
        doc_bc5cdr = self.nlp_bc5cdr(text)
        doc_bionlp = self.nlp_bionlp(text)

        entities = {
            "Diseases": [],
            "Chemicals": [],
            "Physiological_Processes": [],
            "Body_Parts": []
        }

        # Extract from BC5CDR
        for ent in doc_bc5cdr.ents:
            if ent.label_ == "DISEASE":
                if ent.text not in entities["Diseases"]:
                    entities["Diseases"].append(ent.text)
            elif ent.label_ == "CHEMICAL":
                if ent.text not in entities["Chemicals"]:
                    entities["Chemicals"].append(ent.text)

        # Extract from BioNLP13CG (More granular)
        # It has labels like: CANCER, ORGAN, TISSUE, ORGANISM, CELL, AMINO_ACID, etc.
        # We'll filter for relevant ones for a general physician note
        relevant_labels = ["ORGAN", "TISSUE", "SYSTEM_ORGAN_PART"]
        for ent in doc_bionlp.ents:
            if ent.label_ in relevant_labels:
                 if ent.text not in entities["Body_Parts"]:
                    entities["Body_Parts"].append(ent.text)

        return entities

    def get_keywords(self, text):
        """
        Returns a list of all unique entity strings found.
        """
        res = self.extract_entities(text)
        keywords = set()
        for cat in res:
            keywords.update(res[cat])
        return list(keywords)
