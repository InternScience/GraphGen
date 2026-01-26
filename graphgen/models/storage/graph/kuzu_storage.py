import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set

try:
    import kuzu
except ImportError:
    kuzu = None

from graphgen.bases.base_storage import BaseGraphStorage


@dataclass
class KuzuStorage(BaseGraphStorage):
    """
    Graph storage implementation based on KuzuDB.
    Since KuzuDB is a structured graph database and GraphGen uses dynamic dictionaries for properties,
    we map the data to a generic schema:
    - Node Table 'Entity': {id: STRING, data: STRING (JSON)}
    - Rel Table 'Relation': {FROM Entity TO Entity, data: STRING (JSON)}
    """

    working_dir: str = None
    namespace: str = None
    _db: Any = None
    _conn: Any = None

    def __post_init__(self):
        if kuzu is None:
            raise ImportError(
                "KuzuDB is not installed. Please install it via `pip install kuzu`."
            )

        self.db_path = os.path.join(self.working_dir, f"{self.namespace}_kuzu")
        self._init_db()

    def _init_db(self):
        # KuzuDB automatically creates the directory
        self._db = kuzu.Database(self.db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()
        print(f"KuzuDB initialized at {self.db_path}")

    def _init_schema(self):
        """Initialize the generic Node and Edge tables if they don't exist."""
        # Check and create Node table
        try:
            # We use a generic table name "Entity" to store all nodes
            self._conn.execute(
                "CREATE NODE TABLE Entity(id STRING, data STRING, PRIMARY KEY(id))"
            )
            print("Created KuzuDB Node Table 'Entity'")
        except RuntimeError as e:
            # Usually throws if table exists, verify safely or ignore
            print("Node Table 'Entity' already exists or error:", e)

        # Check and create Edge table
        try:
            # We use a generic table name "Relation" to store all edges
            self._conn.execute(
                "CREATE REL TABLE Relation(FROM Entity TO Entity, data STRING)"
            )
            print("Created KuzuDB Rel Table 'Relation'")
        except RuntimeError as e:
            print("Rel Table 'Relation' already exists or error:", e)

    def index_done_callback(self):
        """KuzuDB is ACID, changes are immediate, but we can verify generic persistence here."""

    @staticmethod
    def _safe_json_loads(data_str: str) -> dict:
        if not isinstance(data_str, str) or not data_str.strip():
            return {}
        try:
            return json.loads(data_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {}

    def is_directed(self) -> bool:
        return True

    def get_all_node_degrees(self) -> Dict[str, int]:
        query = """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r]-()
            RETURN n.id, count(r) as degree
        """

        result = self._conn.execute(query)
        degree_map = {}
        while result.has_next():
            row = result.get_next()
            if row and len(row) >= 2:
                node_id, degree = row[0], row[1]
                degree_map[node_id] = int(degree)

        return degree_map

    def get_isolated_nodes(self) -> List[str]:
        query = """
            MATCH (n:Entity)
            WHERE NOT (n)--()
            RETURN n.id
        """

        result = self._conn.execute(query)
        return [row[0] for row in result if row]

    def get_node_count(self) -> int:
        result = self._conn.execute("MATCH (n:Entity) RETURN count(n)")
        return result.get_next()[0]

    def get_edge_count(self) -> int:
        result = self._conn.execute("MATCH ()-[e:Relation]->() RETURN count(e)")
        return result.get_next()[0]

    def get_connected_components(self, undirected: bool = True) -> List[Set[str]]:
        parent = {}
        rank = {}

        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str):
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1

        all_nodes = self.get_all_node_degrees().keys()
        for node_id in all_nodes:
            parent[node_id] = node_id
            rank[node_id] = 0

        query = (
            """
            MATCH (a:Entity)-[e:Relation]-(b:Entity)
            RETURN DISTINCT a.id, b.id
        """
            if undirected
            else """
            MATCH (a:Entity)-[e:Relation]->(b:Entity)
            RETURN DISTINCT a.id, b.id
        """
        )

        result = self._conn.execute(query)
        for row in result:
            if row and len(row) >= 2:
                union(row[0], row[1])

        components_dict = defaultdict(set)
        for node_id in all_nodes:
            root = find(node_id)
            components_dict[root].add(node_id)

        return list(components_dict.values())

    def has_node(self, node_id: str) -> bool:
        result = self._conn.execute(
            "MATCH (a:Entity {id: $id}) RETURN count(a)", {"id": node_id}
        )
        count = result.get_next()[0]
        return count > 0

    def has_edge(self, source_node_id: str, target_node_id: str):
        result = self._conn.execute(
            "MATCH (a:Entity {id: $src})-[e:Relation]->(b:Entity {id: $dst}) RETURN count(e)",
            {"src": source_node_id, "dst": target_node_id},
        )
        count = result.get_next()[0]
        return count > 0

    def node_degree(self, node_id: str) -> int:
        # Calculate total degree (incoming + outgoing)
        query = """
            MATCH (a:Entity {id: $id})-[e:Relation]-(b:Entity)
            RETURN count(e)
        """
        result = self._conn.execute(query, {"id": node_id})
        if result.has_next():
            return result.get_next()[0]
        return 0

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        # In this context, usually checks existence or multiplicity.
        # Kuzu supports multi-edges, so we count them.
        query = """
            MATCH (a:Entity {id: $src})-[e:Relation]->(b:Entity {id: $dst})
            RETURN count(e)
        """
        result = self._conn.execute(query, {"src": src_id, "dst": tgt_id})
        if result.has_next():
            return result.get_next()[0]
        return 0

    def get_node(self, node_id: str) -> Any:
        result = self._conn.execute(
            "MATCH (a:Entity {id: $id}) RETURN a.data", {"id": node_id}
        )
        if not result.has_next():
            return None

        data_str = result.get_next()[0]
        return self._safe_json_loads(data_str)

    def update_node(self, node_id: str, node_data: dict[str, any]):
        current_data = self.get_node(node_id)
        if current_data is None:
            print(f"Node {node_id} not found for update.")
            return

        # Merge existing data with new data
        current_data.update(node_data)
        try:
            json_data = json.dumps(current_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            print(f"Error serializing JSON for node {node_id}: {e}")
            return

        self._conn.execute(
            "MATCH (a:Entity {id: $id}) SET a.data = $data",
            {"id": node_id, "data": json_data},
        )

    def get_all_nodes(self) -> Any:
        """Returns List[Tuple[id, data_dict]]"""
        result = self._conn.execute("MATCH (a:Entity) RETURN a.id, a.data")
        nodes = []
        while result.has_next():
            row = result.get_next()
            if row is None or len(row) < 2:
                continue
            node_id, data_str = row[0], row[1]
            data = self._safe_json_loads(data_str)
            nodes.append((node_id, data))
        return nodes

    def get_edge(self, source_node_id: str, target_node_id: str):
        # Warning: If multiple edges exist, this returns the first one found
        query = """
            MATCH (a:Entity {id: $src})-[e:Relation]->(b:Entity {id: $dst})
            RETURN e.data
        """
        result = self._conn.execute(
            query, {"src": source_node_id, "dst": target_node_id}
        )
        if not result.has_next():
            return None

        data_str = result.get_next()[0]
        return self._safe_json_loads(data_str)

    def update_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, any]
    ):
        current_data = self.get_edge(source_node_id, target_node_id)
        if current_data is None:
            print(f"Edge {source_node_id}->{target_node_id} not found for update.")
            return

        current_data.update(edge_data)
        try:
            json_data = json.dumps(current_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            print(
                f"Error serializing JSON for edge {source_node_id}->{target_node_id}: {e}"
            )
            return

        self._conn.execute(
            """
            MATCH (a:Entity {id: $src})-[e:Relation]->(b:Entity {id: $dst})
            SET e.data = $data
            """,
            {"src": source_node_id, "dst": target_node_id, "data": json_data},
        )

    def get_all_edges(self) -> Any:
        """Returns List[Tuple[src, dst, data_dict]]"""
        query = "MATCH (a:Entity)-[e:Relation]->(b:Entity) RETURN a.id, b.id, e.data"
        result = self._conn.execute(query)
        edges = []
        while result.has_next():
            row = result.get_next()
            if row is None or len(row) < 3:
                continue
            src, dst, data_str = row[0], row[1], row[2]
            data = self._safe_json_loads(data_str)
            edges.append((src, dst, data))
        return edges

    def get_node_edges(self, source_node_id: str) -> Any:
        """Returns generic edges connected to this node (outgoing)"""
        query = """
            MATCH (a:Entity {id: $src})-[e:Relation]->(b:Entity)
            RETURN a.id, b.id, e.data
        """
        result = self._conn.execute(query, {"src": source_node_id})
        edges = []
        while result.has_next():
            row = result.get_next()
            if row is None or len(row) < 3:
                continue
            src, dst, data_str = row[0], row[1], row[2]
            data = self._safe_json_loads(data_str)
            edges.append((src, dst, data))
        return edges

    def upsert_node(self, node_id: str, node_data: dict[str, any]):
        """
        Insert or Update node.
        Kuzu supports MERGE clause (similar to Neo4j) to handle upserts.
        """
        try:
            json_data = json.dumps(node_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            print(f"Error serializing JSON for node {node_id}: {e}")
            return
        query = """
            MERGE (a:Entity {id: $id})
            ON MATCH SET a.data = $data
            ON CREATE SET a.data = $data
        """
        self._conn.execute(query, {"id": node_id, "data": json_data})

    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, any]
    ):
        """
        Insert or Update edge.
        Note: We explicitly ensure nodes exist before merging the edge to avoid errors,
        although GraphGen generally creates nodes before edges.
        """
        # Ensure source node exists and target node exists
        if not self.has_node(source_node_id) or not self.has_node(target_node_id):
            print(
                f"Cannot upsert edge {source_node_id}->{target_node_id} as one or both nodes do not exist."
            )
            return

        try:
            json_data = json.dumps(edge_data, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            print(
                f"Error serializing JSON for edge {source_node_id}->{target_node_id}: {e}"
            )
            return
        query = """
            MATCH (a:Entity {id: $src}), (b:Entity {id: $dst})
            MERGE (a)-[e:Relation]->(b)
            ON MATCH SET e.data = $data
            ON CREATE SET e.data = $data
        """
        self._conn.execute(
            query, {"src": source_node_id, "dst": target_node_id, "data": json_data}
        )

    def delete_node(self, node_id: str):
        # DETACH DELETE removes the node and all connected edges
        query = "MATCH (a:Entity {id: $id}) DETACH DELETE a"
        self._conn.execute(query, {"id": node_id})
        print(f"Node {node_id} deleted from KuzuDB.")

    def get_neighbors(self, node_id: str) -> List[str]:
        query = """
            MATCH (a:Entity {id: $id})-[:Relation]->(b:Entity)
            RETURN b.id
        """
        result = self._conn.execute(query, {"id": node_id})
        neighbors = []
        while result.has_next():
            row = result.get_next()
            if row and len(row) >= 1:
                neighbors.append(row[0])
        return neighbors

    def clear(self):
        """Clear all data but keep schema (or drop tables)."""
        self._conn.execute("MATCH (n) DETACH DELETE n")
        print(f"Graph {self.namespace} cleared.")

    def reload(self):
        """For databases that need reloading, KuzuDB auto-manages this."""
