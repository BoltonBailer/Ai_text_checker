import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load a pre-trained model for AI text detection
MODEL_NAME = "roberta-base-openai-detector"  # A common model for AI-generated text detection

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def check_text(text):
    """Checks if a given text is AI-generated or human-made."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_prob = probs[0][1].item()  # Probability of being AI-generated

    if ai_prob > 0.5:
        print(f"🖥️ This text is likely AI-generated with {ai_prob*100:.2f}% confidence.")
    else:
        print(f"🧑 This text is likely human-made with {(1-ai_prob)*100:.2f}% confidence.")

if __name__ == "__main__":
    user_text = input("Enter a text snippet: ")
    check_text(user_text)
