from sentence_transformers import SentenceTransformer
import config
from pinecone import Pinecone

# Initialize Pinecone client and index
pc = Pinecone(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV)
index = pc.Index(config.PINECONE_INDEX)

# Load local sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embedding for a sample text
text = "Hanoi offers a mix of culture, food, heritage attractions..."
embedding = model.encode(text).tolist()
print(len(embedding))  # Should print 384

# Upsert the embedding into Pinecone
index.upsert(vectors=[("city_hanoi", embedding)])

print("Embedding upserted successfully for city_hanoi!")