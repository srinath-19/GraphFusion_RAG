"""
Graph RAG module for entity-based knowledge graph construction and retrieval.

Extracts entities and relationships from document chunks during ingestion,
builds a knowledge graph, and provides graph-based retrieval to complement
vector similarity search. This enables multi-hop reasoning across documents
(e.g., finding that an employee works on a product that a contract covers).
"""

import json
from pathlib import Path
from pydantic import BaseModel, Field
from litellm import completion
from tenacity import retry, wait_exponential
from tqdm import tqdm

MODEL = "openai/gpt-4.1-nano"

GRAPH_PATH = str(Path(__file__).parent.parent / "preprocessed_db" / "knowledge_graph.json")

wait = wait_exponential(multiplier=1, min=10, max=240)


class Entity(BaseModel):
    name: str = Field(
        description="The canonical name of the entity (e.g., 'Jessica Liu', 'Bizllm', 'Insurellm')"
    )
    entity_type: str = Field(
        description="The type of entity: person, product, company, role, or feature"
    )


class Relationship(BaseModel):
    source: str = Field(description="The name of the source entity")
    target: str = Field(description="The name of the target entity")
    relation: str = Field(
        description="A short description of the relationship (e.g., 'works_on', 'signed_by', 'offers')"
    )


class ChunkEntities(BaseModel):
    entities: list[Entity] = Field(description="All entities mentioned in the chunk")
    relationships: list[Relationship] = Field(
        description="All relationships between entities in the chunk"
    )


EXTRACTION_PROMPT = """You are an expert at extracting structured knowledge from text.
Given the following text chunk from a company knowledge base, extract all entities and relationships.

Entity types: person, product, company, role, feature
Relationship examples: works_on, develops, manages, signed_by, offers, priced_at, reports_to, uses

Extract ONLY entities and relationships explicitly mentioned in the text.
Be precise with entity names. Normalize entity names to their canonical form
(e.g., 'Jessica' -> 'Jessica Liu' if the full name is apparent).

Text chunk:
{text}

Extract all entities and relationships."""


@retry(wait=wait)
def extract_entities_from_chunk(chunk_text):
    """Extract entities and relationships from a single chunk using LLM."""
    messages = [{"role": "user", "content": EXTRACTION_PROMPT.format(text=chunk_text)}]
    response = completion(model=MODEL, messages=messages, response_format=ChunkEntities)
    return ChunkEntities.model_validate_json(response.choices[0].message.content)


def build_graph(chunks):
    """
    Build a knowledge graph from document chunks.

    Each entity becomes a node with associated chunk IDs.
    Each relationship becomes an edge in the graph.
    The graph is persisted as JSON for later retrieval.

    Args:
        chunks: List of Result objects with page_content and metadata

    Returns:
        dict: The knowledge graph structure
    """
    graph = {
        "nodes": {},
        "edges": [],
    }

    for idx, chunk in enumerate(tqdm(chunks, desc="Building knowledge graph")):
        try:
            extracted = extract_entities_from_chunk(chunk.page_content)
        except Exception as e:
            print(f"Warning: Failed to extract entities from chunk {idx}: {e}")
            continue

        chunk_id = str(idx)

        for entity in extracted.entities:
            name = entity.name.lower().strip()
            if name not in graph["nodes"]:
                graph["nodes"][name] = {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "chunk_ids": [],
                }
            if chunk_id not in graph["nodes"][name]["chunk_ids"]:
                graph["nodes"][name]["chunk_ids"].append(chunk_id)

        for rel in extracted.relationships:
            graph["edges"].append(
                {
                    "source": rel.source.lower().strip(),
                    "target": rel.target.lower().strip(),
                    "relation": rel.relation,
                    "chunk_id": chunk_id,
                }
            )

    Path(GRAPH_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    node_count = len(graph["nodes"])
    edge_count = len(graph["edges"])
    print(f"Knowledge graph built: {node_count} entities, {edge_count} relationships")

    return graph


def load_graph():
    """Load the persisted knowledge graph from disk."""
    if not Path(GRAPH_PATH).exists():
        return None
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_related_chunk_ids(query, graph, max_hops=2, max_chunks=10):
    """
    Given a query, find chunk IDs of related content via graph traversal.

    1. Match query terms to entity names in the graph
    2. Collect chunks directly associated with matched entities
    3. Traverse edges to find connected entities (up to max_hops)
    4. Return associated chunk IDs

    Args:
        query: The user's question
        graph: The knowledge graph dict
        max_hops: Maximum number of graph hops (default: 2)
        max_chunks: Maximum chunk IDs to return (default: 10)

    Returns:
        List of chunk ID strings
    """
    if not graph or not graph.get("nodes"):
        return []

    query_lower = query.lower()

    # Step 1: Find entities mentioned in the query
    matched_entities = set()
    for entity_key, entity_data in graph["nodes"].items():
        if entity_key in query_lower or entity_data["name"].lower() in query_lower:
            matched_entities.add(entity_key)
        else:
            query_words = [w for w in query_lower.split() if len(w) > 2]
            for word in query_words:
                if word in entity_key:
                    matched_entities.add(entity_key)
                    break

    if not matched_entities:
        return []

    # Step 2: BFS traversal to find connected entities
    visited = set(matched_entities)
    frontier = set(matched_entities)

    for _ in range(max_hops):
        next_frontier = set()
        for edge in graph["edges"]:
            if edge["source"] in frontier and edge["target"] not in visited:
                next_frontier.add(edge["target"])
            elif edge["target"] in frontier and edge["source"] not in visited:
                next_frontier.add(edge["source"])
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    # Step 3: Collect chunk IDs (prioritize direct matches)
    direct_chunk_ids = []
    connected_chunk_ids = []

    for entity_key in matched_entities:
        if entity_key in graph["nodes"]:
            direct_chunk_ids.extend(graph["nodes"][entity_key]["chunk_ids"])

    for entity_key in visited - matched_entities:
        if entity_key in graph["nodes"]:
            connected_chunk_ids.extend(graph["nodes"][entity_key]["chunk_ids"])

    for edge in graph["edges"]:
        if edge["source"] in matched_entities or edge["target"] in matched_entities:
            if edge["chunk_id"] not in direct_chunk_ids:
                direct_chunk_ids.append(edge["chunk_id"])

    # Deduplicate while preserving order (direct first, then connected)
    seen = set()
    ordered_ids = []
    for cid in direct_chunk_ids + connected_chunk_ids:
        if cid not in seen:
            seen.add(cid)
            ordered_ids.append(cid)

    return ordered_ids[:max_chunks]