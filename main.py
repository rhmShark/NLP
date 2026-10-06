import pickle
from Preprocessing_pipeline import clean_general_text
from English_model import EnglishSentimentModel
from Arabic_model import ArabicSentimentModel

class MultilingualNLPPipeline:
    def __init__(self):
        # Load language classifier
        with open('Language_classifier_weights.pkl', 'rb') as f:
            self.lang_classifier = pickle.load(f)
            
        # Initialize language-specific models
        self.english_model = EnglishSentimentModel()
        self.arabic_model = ArabicSentimentModel()

    def process_text(self, raw_text):
        # Step 1: Clean text for language identification
        cleaned_lang_text = clean_general_text(raw_text)
        
        # Step 2: Detect language
        detected_language = self.lang_classifier.predict([cleaned_lang_text])[0]
        
        # Step 3: Route to language-specific model
        if detected_language == 'Arabic':
            sentiment = self.arabic_model.predict(raw_text)
        elif detected_language == 'English':
            sentiment = self.english_model.predict(raw_text)
        else:
            sentiment = "Unsupported Language"

        # Step 4: Display exact required output
        print("-" * 40)
        print(f"User Text                : {raw_text}")
        print(f"Language                 : {detected_language}")
        print(f"Sentiment Classification : {sentiment}")
        print("-" * 40)

if __name__ == "__main__":
    pipeline = MultilingualNLPPipeline()
    
    # Test cases
    test_inputs = [
        "This movie was perfect and I enjoyed it.",
        "كان الاوردر متأخر جدا والسعر عالي"
    ]
    
    for text in test_inputs:
        pipeline.process_text(text)