import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j"
)

# ── Connect to existing vector index ────────────────────────────────
vector_index = Neo4jVector.from_existing_index(
    embeddings,
    url="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    index_name="movie_plot_vector",
    text_node_property="plot"
)

# ── Fulltext query builder ───────────────────────────────────────────
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = ""
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# ── Fulltext candidate search ────────────────────────────────────────
candidate_query = """
CALL db.index.fulltext.queryNodes($index, $fulltextQuery, {limit: $limit})
YIELD node
RETURN
    coalesce(node.title, node.name) AS candidate,
    labels(node)[0] AS type
"""

def get_candidates(input: str, limit: int = 5) -> List[Dict]:
    ft_query = generate_full_text_query(input)
    if not ft_query:
        return []
    results = graph.query(
        candidate_query,
        {"fulltextQuery": ft_query, "index": "entity", "limit": limit}
    )
    return results  # [{"candidate": "...", "type": "Movie|Person"}]

# ── Vector search with optional metadata filter ──────────────────────
def vector_search(query: str, genre: str = None, min_year: int = None, top_k: int = 3):
    """
    Vector similarity search with optional pre-filtering via Cypher.
    """
    where_clauses = []
    params = {"query_embedding": embeddings.embed_query(query), "top_k": top_k}

    if genre:
        where_clauses.append("m.genre = $genre")
        params["genre"] = genre
    if min_year:
        where_clauses.append("m.year >= $min_year")
        params["min_year"] = min_year

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
        CALL db.index.vector.queryNodes('movie_plot_vector', $top_k, $query_embedding)
        YIELD node AS m, score
        {where_sql}
        RETURN m.title AS title, m.genre AS genre, m.year AS year,
               m.rating AS rating, m.plot AS plot, score
        ORDER BY score DESC
    """
    return graph.query(cypher, params)

# ── Graph context retrieval ──────────────────────────────────────────
def get_movie_context(title: str) -> Dict:
    """Fetch full graph context for a movie (actors, director)."""
    result = graph.query("""
        MATCH (m:Movie {title: $title})
        OPTIONAL MATCH (a:Person)-[:ACTED_IN]->(m)
        OPTIONAL MATCH (d:Person)-[:DIRECTED]->(m)
        RETURN m.title AS title, m.year AS year, m.genre AS genre,
               m.rating AS rating, m.plot AS plot,
               collect(DISTINCT a.name) AS actors,
               collect(DISTINCT d.name) AS directors
    """, {"title": title})
    return result[0] if result else {}

# ── Hybrid retrieval pipeline ────────────────────────────────────────
def hybrid_search(query: str, genre: str = None, min_year: int = None):
    print(f"\n🔍 Query: '{query}'")
    if genre:
        print(f"   Filter: genre={genre}")
    if min_year:
        print(f"   Filter: year>={min_year}")

    # Step 1: Fulltext candidates
    candidates = get_candidates(query)
    print(f"\n📝 Fulltext candidates: {[c['candidate'] for c in candidates]}")

    # Step 2: Vector search (with optional metadata filter)
    vector_results = vector_search(query, genre=genre, min_year=min_year)
    print(f"\n🧠 Vector results:")
    for r in vector_results:
        print(f"   [{r['score']:.3f}] {r['title']} ({r['year']}) - {r['genre']}")

    # Step 3: Graph context for top result
    if vector_results:
        top_movie = vector_results[0]["title"]
        context = get_movie_context(top_movie)
        print(f"\n🕸️  Graph context for '{top_movie}':")
        print(f"   Actors: {context.get('actors', [])}")
        print(f"   Directors: {context.get('directors', [])}")

    return vector_results

# ── Test it ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: pure semantic search
    hybrid_search("space travel and wormholes")

    # Test 2: semantic + genre filter
    hybrid_search("man with extraordinary journey", genre="Drama")

    # Test 3: semantic + year filter
    hybrid_search("crime family power struggle", min_year=2000)

    # Test 4: fulltext candidate lookup
    print("\n--- Candidate lookup ---")
    print(get_candidates("nolan"))
    print(get_candidates("matrix"))