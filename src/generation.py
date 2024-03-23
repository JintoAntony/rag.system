# src/generation.py
from transformers import pipeline

class AnswerGenerator:
    def __init__(self):
        self.summarization_pipeline = pipeline("summarization")

    def generate_answer(self, document_path):
        with open(document_path, 'r') as f:
            document_text = f.read()
            summary = self.summarization_pipeline(document_text, max_length=50, min_length=10, do_sample=False)
            return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    generator = AnswerGenerator()
    document_path = "data/documents/document1.txt"  # Replace with relevant document path
    answer = generator.generate_answer(document_path)
    print("Generated Answer:", answer)
