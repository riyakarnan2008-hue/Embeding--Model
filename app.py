from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# List of sentences
sentences = [
    "Vera loves learning Python.",
    "Vera enjoys programming.",
    "Vera likes building AI projects.",
    "Vera wants to become an AI engineer.",
    "Vera is interested in machine learning.",
    "Artificial intelligence is changing the world.",
    "Python is widely used in artificial intelligence.",
    "Vera enjoys listening to music."
]


# Convert sentences into embeddings
embeddings = model.encode(sentences)


# Display total sentences and embedding dimension
print("Total number of sentences:", len(sentences))
print("Embedding dimension:", len(embeddings[0]))


# Display embeddings
print("\n--- Embeddings ---")

for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    print("Embedding:", embeddings[i])


# Calculate cosine similarity
similarity = cosine_similarity(embeddings)


# Display semantic similarity
print("\n--- Semantic Similarity ---")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        if similarity[i][j] > 0.5:
            print(
                f"\nSentence 1: {sentences[i]}"
                f"\nSentence 2: {sentences[j]}"
                f"\nSimilarity: {similarity[i][j]:.4f}"
            )