import re
import string

# clean text and 
def clean_general_text(text):

   # separate text into words 
    text = str(text).strip()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Extra whitespace cleanup
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_english(text):
    """
    Preprocessing specifically for English Sentiment Analysis.
    """
    text = clean_general_text(text).lower()
    # Basic token-level clean
    tokens = text.split()
    return " ".join(tokens)

def preprocess_arabic(text):
    """
    Preprocessing specifically for Arabic Sentiment Analysis.
    Removes diacritics (Tashkeel), lengthenings (Tatweel), and normalizes letters.
    """
    text = clean_general_text(text)
    
    # Remove Tashkeel (Arabic diacritics)
    tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    
    # Remove Tatweel (kashida)
    text = re.sub(r'\u0640', '', text)
    
    # Normalize Alef variations
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text) # أ, إ, آ -> ا
    # Normalize Yaa and Taa Marbouta
    text = re.sub(r'\u0649', '\u064A', text) # ى -> ي
    text = re.sub(r'\u0629', '\u0647', text) # ة -> ه
    
    return text.strip()