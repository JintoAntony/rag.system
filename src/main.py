# src/main.py
from retrieval import DocumentRetriever
from generation import AnswerGenerator

def main():
    retriever = DocumentRetriever("data/documents/")
    generator = AnswerGenerator()

    query = input("Enter your question: ")
    relevant_documents = retriever.retrieve_documents(query, num_results=1)
    if relevant_documents:
        answer = generator.generate_answer(relevant_documents[0][0])
        print("Answer:", answer)
    else:
        print("No relevant documents found.")

if __name__ == "__main__":
    main()
