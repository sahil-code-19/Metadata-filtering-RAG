1. Set the .env file with following variables
GOOGLE_API_KEY="AIz...."
NEO4J_URI="neo4j://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"
NEO4J_DATABASE="neo4j"

2. Run the `docker-compose up -d` to run neo4j

3. Run the `uv sync` to installing the all dependencies.

4. run seed_data.py file with `uv run seed_data.py`

5. then finally run the main.py file 🥳
