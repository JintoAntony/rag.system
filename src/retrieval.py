# src/retrieval.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class DocumentRetriever:
    def __init__(self, documents_dir):
        self.documents_dir = documents_dir
        self.document_paths = [os.path.join(documents_dir, f) for f in os.listdir(documents_dir)]
        self.vectorizer = TfidfVectorizer()

    def retrieve_documents(self, query, num_results=1):
        query_vector = self.vectorizer.fit_transform([query])
        similarities = []
        for doc_path in self.document_paths:
            with open(doc_path, 'r') as f:
                document_text = f.read()
                document_vector = self.vectorizer.transform([document_text])
                similarity = cosine_similarity(query_vector, document_vector)
                similarities.append((doc_path, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_results]

# Example usage
if __name__ == "__main__":
    retriever = DocumentRetriever("data/documents/")
    query = "How to reset password?"
    relevant_documents = retriever.retrieve_documents(query, num_results=1)
    print("Relevant Document:", relevant_documents[0][0])
