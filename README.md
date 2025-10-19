## AI-Hybrid-Chat: Travel Assistant
#### Overview
This repository contains the "AI-Hybrid-Chat" project, a hybrid AI travel assistant developed for the Blue Enigma Team's evaluation. It integrates Pinecone (vector search), Neo4j (graph relations), and OpenAI (chat reasoning) to generate context-rich travel itineraries (e.g., a 4-day romantic Vietnam itinerary). The system is offline-capable with local embeddings and optimized for performance.
Features

- **Semantic Search**: Uses Pinecone with local SentenceTransformer (all-MiniLM-L6-v2) for embedding generation.
- **Graph Intelligence**: Neo4j provides relationship context via async queries.
- **Creative Responses**: OpenAI GPT-4o-mini delivers coherent, narrative outputs.
- **Innovations**: Includes caching (40-60% faster), async parallelization, and a color-coded CLI.

#### Installation

- Clone the repo: git clone https://github.com/Zeny1303/AI_hybrid_chat_app.git.
- Install dependencies: pip install -r requirements.txt.
- Set up environment variables for API keys (e.g., OPENAI_API_KEY, PINECONE_API_KEY, NEO4J_URI) in a secure config file (not included).
- Run: python hybrid_chat.py.

#### Usage

- Start the CLI: Type travel queries (e.g., "create a romantic 4-day itinerary for Vietnam") and press Enter.
- Exit with "exit" or "quit".
- View outputs and debug logs in the terminal.

#### Files

- Improvements.md: Project enhancements and evaluation details.
- hybrid_chat.py: Main chat interface.
- pinecone_upload.py: Uploads embeddings to Pinecone.
- load_to_neo4j.py: Loads data into Neo4j.
- Screenshots/: Proof of functionality (e.g., upsert360.png, output1.png).

##### Scaling

Scales to 1M nodes with larger Pinecone pods (e.g., p2.x2), Neo4j clustering, and optimized batching.

Author

Sneha
Date: October 19, 2025
