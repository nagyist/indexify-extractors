from .base_extractor import EXTRACTORS_PATH, ExtractorMetadata
import sqlite3
import json
from typing import List
import os
from indexify.extractor_sdk import EmbeddingSchema

class ExtractorMetadataStore:
    def __init__(self):
        self._path = os.path.join(EXTRACTORS_PATH, "extractors.db")
        self.create_store()

    def create_store(self):
        with sqlite3.connect(self._path) as conn:
            cur = conn.cursor()
            cur.execute(f"""
                    CREATE TABLE 
                    IF NOT EXISTS extractors (
                    id TEXT NOT NULL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    input_params TEXT,
                    input_mime_types TEXT,
                    metadata_schemas TEXT,
                    embedding_schemas TEXT
                    )
                    """)
            conn.commit()

    def save_description(self, id: str, description: ExtractorMetadata):
        with sqlite3.connect(self._path) as conn:
            cur = conn.cursor()
            input_params: str = json.dumps(description.input_params) if description.input_params else ""

            # Convert the lists to JSON strings
            mime_types = json.dumps(description.input_mime_types)
            schemas = {}
            for name, embedding_schema in description.embedding_schemas.items():
                schemas[name] = embedding_schema.model_dump_json()
 
            embedding_schemas = json.dumps(schemas)
            metadata_schemas = json.dumps(description.metadata_schemas)

            # Insert the extractor info into the database
            cur.execute("""
                INSERT INTO extractors (id, name, description, input_params, input_mime_types, metadata_schemas, embedding_schemas)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) 
                DO UPDATE SET 
                    id = excluded.id,
                    name = excluded.name,
                    description = excluded.description,
                    input_params = excluded.input_params,
                    input_mime_types = excluded.input_mime_types,
                    metadata_schemas = excluded.metadata_schemas,
                    embedding_schemas = excluded.embedding_schemas;
            """, [id, description.name, description.description, input_params, mime_types, metadata_schemas, embedding_schemas])

            conn.commit()

    def extractor_module_class(self, name: str) -> str:
        with sqlite3.connect(self._path) as conn:
            cur = conn.cursor()
            record = cur.execute("SELECT id FROM extractors WHERE name = ?", (name,)).fetchone()
            if record is None:
                raise ValueError(f"Extractor {name} not found in the database.")
            return record[0] 
        return None
    
    def all_extractor_metadata(self) -> List[ExtractorMetadata]:
        with sqlite3.connect(self._path) as conn:
            cur = conn.cursor()
            records = cur.execute("SELECT * FROM extractors").fetchall()
            return [self._load_extractor_description(record) for record in records]
        return []

    def _load_extractor_description(self, record) -> ExtractorMetadata:
        """Load the description of an extractor from SQLite database record."""

        # Rebuild the embedding schemas.
        _embedding_schemas = json.loads(record[6])
        embedding_schemas = {}
        for name, schema in _embedding_schemas.items():
            schema = json.loads(schema)
            embedding_schemas[name] = EmbeddingSchema(
                dim=schema["dim"],
                distance=schema["distance"],
            )

        input_params = json.loads(record[3]) if record[3] else None

        description = ExtractorMetadata(
            name=record[1],
            version="",
            description=record[2],
            python_dependencies=[],
            system_dependencies=[],
            input_params=input_params,
            input_mime_types=json.loads(record[4]),
            metadata_schemas=json.loads(record[5]),
            embedding_schemas=embedding_schemas,
        )
        return description

