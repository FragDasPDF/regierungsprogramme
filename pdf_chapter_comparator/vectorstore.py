import numpy as np
import json
import os
import datetime
import hnswlib
from typing import List, Dict, Any, Optional
import logging
from nltk import sent_tokenize


class VectorStore:
    def __init__(
        self,
        embedding_model,
        index_path="data/embeddings/vectorstore.bin",
        meta_path="data/embeddings/metadata.json",
        embedding_dimension=768,
    ):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata = []
        self.dimension = 768  # Hardcode to 768
        self.current_size = 0
        self.max_elements = 100000  # Adjust based on expected data size

        # Create directories if needed
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        # Try loading existing data
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load_from_disk()
        else:
            self._init_index()

    def _init_index(self):
        """Initialize a new HNSW index"""
        self.index = hnswlib.Index(space="l2", dim=self.dimension)
        self.index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
        self.index.set_ef(50)  # Controls accuracy vs speed during search

    def _load_from_disk(self):
        self.index = hnswlib.Index(space="l2", dim=self.dimension)
        self.index.load_index(self.index_path)
        with open(self.meta_path, "r") as f:
            self.metadata = json.load(f)
        self.current_size = len(self.metadata)

    def save_to_disk(self):
        if self.index is not None:
            self.index.save_index(self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f)

    def store_embedding(self, id_str: str, embedding: np.ndarray, metadata: dict):
        """Store a single embedding with its metadata"""
        try:
            # Reshape embedding if needed
            embedding = embedding.reshape(1, -1)

            # Add to index
            idx = self.current_size
            self.index.add_items(embedding, np.array([idx]))

            # Store metadata
            self.metadata.append(
                {
                    **metadata,
                    "id": id_str,
                    "vector_id": idx,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

            self.current_size += 1
            self.save_to_disk()
        except Exception as e:
            logging.error(f"Error storing embedding for {id_str}: {e}")
            raise

    def get_embedding(self, id_str: str) -> Optional[np.ndarray]:
        """Retrieve a stored embedding by its ID string"""
        try:
            # Find metadata entry for this ID
            entry = next(
                (item for item in self.metadata if item.get("id") == id_str), None
            )

            if entry is None:
                return None

            # Get embedding from index
            vector_id = entry["vector_id"]
            embedding = self.index.get_items([vector_id])[0]

            return embedding
        except Exception as e:
            logging.error(f"Error retrieving embedding for {id_str}: {e}")
            return None

    def store_sections_for_pdf(self, pdf_path: str, content: List[Dict]) -> None:
        """Store content for a PDF"""
        try:
            pdf_id = os.path.basename(pdf_path)

            # Store document-level content
            for doc_content in content:
                # Store the document embedding
                doc_id = f"{pdf_path}:document"
                if "embeddings" in doc_content:
                    self.store_embedding(
                        doc_id,
                        doc_content["embeddings"],
                        {
                            "type": "document",
                            "pdf_id": pdf_id,
                            "text": doc_content["text"],
                            "pages": doc_content["pages"],
                            "pdf_path": doc_content["pdf_path"],
                        },
                    )

                    # Extract and store sentence embeddings if not already stored
                    sentences = sent_tokenize(doc_content["text"].strip())
                    for sentence in sentences:
                        if sentence.strip():
                            sent_id = f"{pdf_path}:{sentence}"
                            # Check if sentence is already stored
                            if not any(m["id"] == sent_id for m in self.metadata):
                                page = self._find_page_number(
                                    sentence, doc_content["text"], doc_content["pages"]
                                )
                                # Store sentence with document context
                                self.store_sentence_embedding(
                                    sentence,
                                    pdf_path,
                                    self.embedding_model.encode(sentence),
                                    page,
                                    pdf_id,
                                )

        except Exception as e:
            logging.error(f"Error storing content for {pdf_path}: {e}")
            raise

    def _find_page_number(self, sentence: str, content: str, pages: List[int]) -> int:
        """Helper function to find the page number for a sentence"""
        sentence_pos = content.find(sentence)
        if sentence_pos == -1:
            return pages[0]  # Default to first page if not found

        content_before = content[:sentence_pos]
        newline_count = content_before.count("\n")
        page_index = min(
            newline_count // 40, len(pages) - 1
        )  # Assuming ~40 lines per page
        return pages[page_index]

    def get_sections_for_pdf(self, pdf_path: str) -> Optional[Dict]:
        """Retrieve stored content for a PDF"""
        try:
            pdf_id = os.path.basename(pdf_path)
            doc_id = f"{pdf_path}:document"

            # Look for document in metadata
            doc_entry = next(
                (
                    entry
                    for entry in self.metadata
                    if entry["type"] == "document" and entry["pdf_path"] == pdf_path
                ),
                None,
            )

            if not doc_entry:
                return None

            # Get the document embedding
            embedding = self.get_embedding(doc_id)
            if embedding is None:
                return None

            # Reconstruct document content
            return {
                "text": doc_entry["text"],
                "pages": doc_entry["pages"],
                "pdf_path": doc_entry["pdf_path"],
                "embeddings": embedding,
            }

        except Exception as e:
            logging.error(f"Error retrieving content for {pdf_path}: {e}")
            return None

    def get_sentence_embedding(
        self, sentence: str, pdf_path: str
    ) -> Optional[Dict[str, Any]]:
        """Get embedding and metadata for a specific sentence from a PDF"""
        try:
            sent_id = f"{pdf_path}:{sentence}"
            # Find metadata entry for this sentence
            entry = next(
                (
                    item
                    for item in self.metadata
                    if item["type"] == "sentence" and item["id"] == sent_id
                ),
                None,
            )

            if entry is None:
                return None

            # Get embedding from index
            embedding = self.get_embedding(sent_id)
            if embedding is None:
                return None

            return {
                "embedding": embedding,
                "page": entry["page"],
                "document_name": entry["document_name"],
                "text": entry["text"],
            }

        except Exception as e:
            logging.error(f"Error retrieving sentence embedding: {e}")
            return None

    def store_sentence_embedding(
        self,
        sentence: str,
        pdf_path: str,
        embedding: np.ndarray,
        page: int,
        document_name: str,
    ) -> None:
        """Store embedding and metadata for a specific sentence"""
        sent_id = f"{pdf_path}:{sentence}"
        self.store_embedding(
            sent_id,
            embedding,
            {
                "type": "sentence",
                "pdf_path": pdf_path,
                "text": sentence,
                "page": page,
                "document_name": document_name,
            },
        )

    def store_comparison_results(
        self, chapter_pair: Dict[str, str], matches: List[Dict[str, float]]
    ):
        """Store the results of chapter comparisons"""
        comparison_data = {
            "doc1_chapter": chapter_pair["doc1_chapter"],
            "doc2_chapter": chapter_pair["doc2_chapter"],
            "matches": matches,
        }
        self.metadata.append(
            {
                "type": "comparison",
                "data": comparison_data,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        self.save_to_disk()

    def get_chapter_comparisons(
        self, min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve stored chapter comparisons"""
        return [
            entry["data"]
            for entry in self.metadata
            if entry["type"] == "comparison"
            and entry["data"]["matches"][0]["similarity"] >= min_similarity
        ]
