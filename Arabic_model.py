import pickle
from Preprocessing_pipeline import preprocess_arabic

class ArabicSentimentModel:
    def __init__(self, model_path='Arabic_model_weights.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, text):
        cleaned_text = preprocess_arabic(text)
        prediction = self.model.predict([cleaned_text])[0]
        return prediction