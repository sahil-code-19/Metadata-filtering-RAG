from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j"
)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# ── 1. Clear old data ───────────────────────────────────────────────
graph.query("MATCH (n) DETACH DELETE n")

# ── 2. Sample movies dataset ────────────────────────────────────────
movies = [
    {"title": "The Matrix", "year": 1999, "genre": "Sci-Fi", "rating": 8.7,
     "plot": "A hacker discovers reality is a simulation and joins a rebellion against the machines."},
    {"title": "Inception", "year": 2010, "genre": "Sci-Fi", "rating": 8.8,
     "plot": "A thief who enters dreams to steal secrets is given the task of planting an idea."},
    {"title": "Interstellar", "year": 2014, "genre": "Sci-Fi", "rating": 8.6,
     "plot": "Astronauts travel through a wormhole near Saturn to find a new home for humanity."},
    {"title": "The Godfather", "year": 1972, "genre": "Crime", "rating": 9.2,
     "plot": "The aging patriarch of a crime dynasty transfers control to his reluctant son."},
    {"title": "Pulp Fiction", "year": 1994, "genre": "Crime", "rating": 8.9,
     "plot": "The lives of two hitmen, a boxer, and a gangster's wife intertwine in Los Angeles."},
    {"title": "The Dark Knight", "year": 2008, "genre": "Action", "rating": 9.0,
     "plot": "Batman faces the Joker, a criminal mastermind who plunges Gotham into chaos."},
    {"title": "Forrest Gump", "year": 1994, "genre": "Drama", "rating": 8.8,
     "plot": "A man with a low IQ witnesses and influences major historical events in 20th-century America."},
    {"title": "Schindler's List", "year": 1993, "genre": "Drama", "rating": 9.0,
     "plot": "A businessman saves over a thousand Jewish refugees during the Holocaust."},
]

cast = [
    {"movie": "The Matrix",     "actor": "Keanu Reeves",    "director": "Lana Wachowski"},
    {"movie": "Inception",      "actor": "Leonardo DiCaprio","director": "Christopher Nolan"},
    {"movie": "Interstellar",   "actor": "Matthew McConaughey","director": "Christopher Nolan"},
    {"movie": "The Godfather",  "actor": "Marlon Brando",   "director": "Francis Ford Coppola"},
    {"movie": "Pulp Fiction",   "actor": "John Travolta",   "director": "Quentin Tarantino"},
    {"movie": "The Dark Knight","actor": "Christian Bale",  "director": "Christopher Nolan"},
    {"movie": "Forrest Gump",   "actor": "Tom Hanks",       "director": "Robert Zemeckis"},
    {"movie": "Schindler's List","actor": "Liam Neeson",    "director": "Steven Spielberg"},
]

# ── 3. Create Movie nodes + embeddings ──────────────────────────────
print("Generating embeddings for movie plots...")
for m in movies:
    embedding = embeddings.embed_query(m["plot"])
    graph.query("""
        CREATE (movie:Movie {
            title: $title,
            year: $year,
            genre: $genre,
            rating: $rating,
            plot: $plot,
            embedding: $embedding
        })
    """, {**m, "embedding": embedding})
    print(f"  ✓ {m['title']}")

# ── 4. Create Person nodes + relationships ──────────────────────────
print("\nCreating Person nodes and relationships...")
for c in cast:
    graph.query("""
        MATCH (movie:Movie {title: $movie})
        MERGE (actor:Person {name: $actor})
        MERGE (director:Person {name: $director})
        MERGE (actor)-[:ACTED_IN]->(movie)
        MERGE (director)-[:DIRECTED]->(movie)
    """, c)
    print(f"  ✓ {c['movie']} → {c['actor']}, {c['director']}")

# ── 5. Create FULLTEXT index ────────────────────────────────────────
print("\nCreating fulltext index...")
graph.query("DROP INDEX entity IF EXISTS")
graph.query("""
    CREATE FULLTEXT INDEX entity
    FOR (n:Movie|Person)
    ON EACH [n.title, n.name]
""")
print("  ✓ Fulltext index 'entity' created")

# ── 6. Create VECTOR index (3072 dims for Gemini) ───────────────────
print("\nCreating vector index...")
graph.query("DROP INDEX movie_plot_vector IF EXISTS")
graph.query("""
    CREATE VECTOR INDEX movie_plot_vector
    FOR (m:Movie)
    ON m.embedding
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 3072,
            `vector.similarity_function`: 'cosine'
        }
    }
""")
print("  ✓ Vector index 'movie_plot_vector' created (3072 dims)")

print("\n✅ Database seeded successfully!")
print(f"   {len(movies)} movies | {len(cast)} actors | {len(cast)} directors")