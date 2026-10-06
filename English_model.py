import pickle
from Preprocessing_pipeline import preprocess_english

class EnglishSentimentModel:
    def __init__(self, model_path='English_model_weights.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, text):
        cleaned_text = preprocess_english(text)
        prediction = self.model.predict([cleaned_text])[0]
        return prediction