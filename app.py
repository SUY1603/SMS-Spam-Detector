import torch
from torch import nn
import gradio as gr
import pickle
from sklearn.feature_extraction.text import CountVectorizer


model = nn.Linear(5000, 1)
try:
    model.load_state_dict(torch.load('./model/model.pt', map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: './model/model.pt' not found. Make sure you have trained and saved your model.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

model.eval()

try:
    with open('./model/vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    print("Vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: './model/vectorizer.pkl' not found. Make sure you have saved your fitted CountVectorizer.")
except Exception as e:
    print(f"An error occurred while loading the vectorizer: {e}")

def detect_spam(message: str):
    """
    Analyzes a single message string to determine if it is spam.

    Args:
        message: The text message to analyze.

    Returns:
        A dictionary with labels and their confidence scores for Gradio.
    """
    if not message.strip():
        return "Not Spam: 1.0"
        
    # The message needs to be in a list or iterable to be transformed by the vectorizer
    message_vector = cv.transform([message]).toarray()
    message_tensor = torch.tensor(message_vector, dtype=torch.float32)

    with torch.no_grad():
        output_logits = model(message_tensor)
        spam_prob = torch.sigmoid(output_logits).item()

        if spam_prob > 0.5:
            return f"Spam (Probability: {spam_prob:.2%})"
        else:
            return f"Not Spam (Probability: {(1-spam_prob):.2%})"

if __name__ == "__main__":
    demo = gr.Interface(
        fn=detect_spam,
        inputs=gr.Textbox(lines=5, label="Message", placeholder="Enter a message to check for spam"),
        outputs=gr.Textbox(label="Prediction"),
        title="SMS Spam Detection Model",
        description="Enter a text message to see if our deep learning model classifies it as Spam or Not Spam. This app uses a PyTorch model and a Scikit-learn CountVectorizer.",
        examples=[
            ["WINNER!! As a valued network customer you have been selected to receive a Rs.900 prize reward!"],
            ["Hey, what time are you getting here?"],
            ["URGENT! You have 1 new voicemail from a secret admirer. Call 09066368311"]
        ]
    )

    demo.launch()
