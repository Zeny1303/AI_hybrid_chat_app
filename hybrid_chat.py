import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase, AsyncGraphDatabase
import config  

# add little bit of colors for better readability
from colorama import Fore, Style, init
init(autoreset=True)


# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"  # Hugging Face embedder
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)
model = SentenceTransformer(EMBED_MODEL)  # Local embedding model

#  Improvement: Caching for embeddings
embedding_cache = {}

def get_cached_embedding(text):
    """Cache embeddings locally to speed up repeated queries."""
    if text not in embedding_cache:
        embedding_cache[text] = model.encode([text])[0].tolist()
    return embedding_cache[text]

# Connect to Pinecone index with error handling
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=config.PINECONE_ENV)
    )
    # Wait for index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(5)

index = pc.Index(INDEX_NAME)

# Connect to Neo4j (async driver)
driver = AsyncGraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

# -----------------------------
# Helper functions
# -----------------------------
async def embed_text(text: str) -> List[float]:
    """Get cached embedding locally from SentenceTransformer."""
    return get_cached_embedding(text)

async def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    vec = await embed_text(query_text)
    res = await asyncio.to_thread(index.query, vector=vec, top_k=top_k, include_metadata=True, include_values=False)
    print("DEBUG: Pinecone top 5 results:")
    print(len(res["matches"]))
    return res["matches"]

async def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j for context (async)."""
    facts = []
    async with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.description AS description LIMIT 10"
            )
            recs = await session.run(q, nid=nid)
            async for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"] or r["id"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                })
    print("DEBUG: Graph facts:")
    print(len(facts))
    return facts

# Improvement: Summarize top Pinecone results for clarity
def search_summary(results):
    """Summarize top Pinecone search results to reduce noise."""
    texts = [r.get("metadata", {}).get("text", "") for r in results]
    return " ".join(texts[:3]) if texts else "No summary available."

# Improvement: Chain-of-thought style structured prompt
def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a clearer reasoning prompt using vector + graph data."""
    summary_context = search_summary(pinecone_matches)
    system_msg = (
        "You are a travel planner assistant combining vector-based semantic search and graph knowledge. "
        "Follow this reasoning process step-by-step:\n"
        "1. Understand what the user wants.\n"
        "2. Use the summarized search and graph facts to extract relevant insights.\n"
        "3. Generate a coherent, creative, and realistic response using full names of places and attractions where possible, ensuring it’s human-readable and complete.\n"
        "4. Mention node IDs only if specific data is unavailable."
    )
    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]
    prompt = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         f"Summarized search context:\n{summary_context}\n\n"
         f"Graph facts:\n{chr(10).join(graph_context[:20])}\n\n"
         "Now reason step-by-step and then produce the final answer below."}
    ]
    return prompt

async def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion asynchronously."""
    return await asyncio.to_thread(
        client.chat.completions.create,
        model=CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=1000,
        temperature=0.7  # Increased for creativity
    )

# Improvement: Async orchestration
async def handle_query_async(query):
    """Run Pinecone and Neo4j fetch concurrently with minimal overhead."""
    
    # Step 1: Get Pinecone matches first (only once)
    matches = await pinecone_query(query)
    node_ids = [m["id"] for m in matches]

    # Step 2: Fetch Neo4j facts concurrently (while preparing chat prompt)
    graph_facts_task = asyncio.create_task(fetch_graph_context(node_ids))

    # Step 3: Wait for graph facts
    graph_facts = await graph_facts_task

    # Step 4: Build the structured reasoning prompt
    prompt = build_prompt(query, matches, graph_facts)

    # Step 5: Call OpenAI chat model asynchronously
    return await call_chat(prompt)


# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat():
    print("Hybrid travel assistant. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        try:
            answer = await handle_query_async(query)
        except Exception as e:
            print(f"Error: {e}, using sync fallback.")
            matches = pinecone_query(query, top_k=TOP_K)
            match_ids = [m["id"] for m in matches]
            graph_facts = fetch_graph_context(match_ids)
            prompt = build_prompt(query, matches, graph_facts)
            answer = call_chat(prompt)
        print(Fore.CYAN + "\n=== Assistant Answer ===\n" + Style.RESET_ALL)
        print(Fore.LIGHTWHITE_EX + answer.choices[0].message.content.strip() + Style.RESET_ALL)
        print(Fore.MAGENTA + "\n=== End ===\n" + Style.RESET_ALL)


# -----------------------------
# Adding startup banner and running the chat
# -----------------------------
if __name__ == "__main__":
    print(Fore.CYAN + Style.BRIGHT + "\n HYBRID TRAVEL ASSISTANT ")
    print(Fore.YELLOW + "Combining Vector Search  + Graph Intelligence  + GPT Reasoning ")
    print(Fore.MAGENTA + "-" * 70)
    print(Fore.GREEN + "Type your question below or 'exit' to quit.\n")
    print(Fore.MAGENTA + "-" * 70)
    print(Style.RESET_ALL)

    async def main():
        await interactive_chat()
        await driver.close()   #  Properly awaited!

    asyncio.run(main())

