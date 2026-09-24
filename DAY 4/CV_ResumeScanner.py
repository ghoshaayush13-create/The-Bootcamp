import spacy

# Load spaCy's lightweight English core model (~12MB)
nlp = spacy.load("en_core_web_sm")

# Sample unstructured text input
raw_resume_text = """
Rahul Sharma is a Software Developer living in New Delhi. 
He worked at Google for 3 years starting from March 2021. 
Rahul graduated from Delhi University and has expertise in Python, Machine Learning and AWS.
"""

print("Processing text with spaCy...\n")
doc = nlp(raw_resume_text)

print("--- Extracted Named Entities ---")
# Extract entities like Persons, Organizations, Locations, and Dates
for ent in doc.ents:
  print(f"Entity: {ent.text:<20} | Type: {ent.label_:<10} ({spacy.explain(ent.label_)})")

print("\n--- Identified Keywords/Nouns ---")
# Extract key nouns/skills
nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
print("Key Nouns Found:", list(set(nouns)))