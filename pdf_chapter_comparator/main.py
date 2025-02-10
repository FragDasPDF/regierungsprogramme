#!/usr/bin/env python3
import argparse
import concurrent.futures
import logging
import os
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pdf_chapter_comparator.report_generator import generate_html_report
from pdf_chapter_comparator.vectorstore import VectorStore

# Download required NLTK data (if not already present)
nltk.download("punkt", quiet=True)

########################################
# Configure logging
########################################
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

########################################
# Threading Configuration
########################################
MAX_WORKERS = os.cpu_count() or 4  # Default to 4 if cpu_count returns None
CHUNK_SIZE = 5  # Number of pages to process in each chunk

########################################
# Similarity Configuration
########################################
SENTENCE_SIMILARITY_THRESHOLD = 0.5  # Changed from 0.85 to 0.5

# Initialize the embedding model
embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)

# Initialize vectorstore with the embedding model
vectorstore = VectorStore(embedding_model)


def extract_content_from_pdf(pdf_path, vectorstore=None):
    """Extract all text content from PDF file with page numbers"""
    try:
        # First check if we have stored content for this PDF
        if vectorstore:
            stored_content = vectorstore.get_sections_for_pdf(pdf_path)
            if stored_content:
                logging.info(f"Found stored content for {pdf_path}")
                # Transform stored content into expected format
                return {
                    "text": stored_content.get("text", ""),
                    "pages": stored_content.get("pages", [1]),
                    "pdf_path": stored_content.get("pdf_path", pdf_path),
                    "embeddings": stored_content.get("embeddings", None)
                }

        content = {"sections": [], "pdf_path": pdf_path}

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logging.info(f"Processing PDF with {total_pages} pages")

            for page_num, page in tqdm(
                enumerate(pdf.pages, 1), total=total_pages, desc="Extracting text"
            ):
                try:
                    text = page.extract_text()
                    if not text:
                        logging.warning(f"Empty text on page {page_num}")
                        continue

                    # Store each page as a separate section
                    content["sections"].append(
                        {"text": text, "page": page_num, "embedding": None}
                    )

                except Exception as e:
                    logging.error(f"Error processing page {page_num}: {str(e)}")
                    continue

        # Compute embeddings for each section
        for section in tqdm(content["sections"], desc="Processing sections"):
            if section["text"]:
                try:
                    # Process in chunks to avoid memory issues
                    chunk_size = 1000  # Characters per chunk
                    text_chunks = [
                        section["text"][i : i + chunk_size]
                        for i in range(0, len(section["text"]), chunk_size)
                    ]

                    # Compute embedding for the section
                    embeddings = [
                        embedding_model.encode(chunk) for chunk in text_chunks
                    ]
                    section["embedding"] = np.mean(embeddings, axis=0)

                    # Store in vectorstore if available
                    if vectorstore:
                        vectorstore.store_sections_for_pdf(
                            pdf_path, content["sections"]
                        )

                except Exception as e:
                    logging.error(f"Error processing section: {str(e)}")
                    continue

        return content

    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return None


def get_embedding(text):
    """Returns the embedding for a given text using the embedding model."""
    try:
        return embedding_model.encode(text)
    except Exception as e:
        logging.error(f"Error computing embedding for text: {text[:30]}...: {e}")
        return None


def compute_embeddings(sentences):
    """Computes embeddings for a list of sentences concurrently."""
    embeddings = [None] * len(sentences)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(get_embedding, sentence): idx
            for idx, sentence in enumerate(sentences)
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Computing embeddings",
            unit="sent",
        ):
            idx = futures[future]
            result = future.result()
            embeddings[idx] = result
    return embeddings


def find_page_number(sentence, sections):
    """Find the page number for a sentence from section data"""
    for section in sections:
        if sentence in section["text"]:
            return section["page"]
    return sections[0]["page"] if sections else 1


def compare_documents(doc1_content, doc2_content, threshold=0.85, vectorstore=None):
    """Compare sentences between two documents"""
    try:
        # Validate input structure based on VectorStore format
        if not isinstance(doc1_content, dict) or "text" not in doc1_content:
            logging.error("Invalid document 1 content structure")
            return []

        if not isinstance(doc2_content, dict) or "text" not in doc2_content:
            logging.error("Invalid document 2 content structure")
            return []

        # Extract document paths
        doc1_path = doc1_content.get("pdf_path", "")
        doc2_path = doc2_content.get("pdf_path", "")

        # Extract text content and tokenize into sentences
        sentences1 = [
            (
                sent,
                doc1_content.get("pages", [1])[0],
            )  # Default to page 1 if not specified
            for sent in sent_tokenize(doc1_content["text"].strip())
            if sent.strip()
        ]
        sentences2 = [
            (
                sent,
                doc2_content.get("pages", [1])[0],
            )  # Default to page 1 if not specified
            for sent in sent_tokenize(doc2_content["text"].strip())
            if sent.strip()
        ]

        if not sentences1 or not sentences2:
            logging.warning("No sentences found in one or both documents")
            return []

        # Process embeddings for each document
        def process_embeddings(sentences, pdf_path):
            embeddings = []
            for sent, page in tqdm(
                sentences,
                desc=f"Processing embeddings for {os.path.basename(pdf_path)}",
            ):
                try:
                    # Try to get stored embedding
                    if vectorstore:
                        stored_data = vectorstore.get_sentence_embedding(sent, pdf_path)
                        if (
                            stored_data is not None
                            and isinstance(stored_data, dict)
                            and "embedding" in stored_data
                        ):
                            embeddings.append(np.array(stored_data["embedding"]))
                            continue

                    # If no stored embedding found, compute new one
                    emb = embedding_model.encode(sent)
                    embeddings.append(emb)

                    # Store the new embedding if vectorstore is available
                    if vectorstore:
                        vectorstore.store_sentence_embedding(
                            sent,
                            pdf_path,
                            emb.tolist(),
                            page,
                            os.path.basename(pdf_path),
                        )
                except Exception as e:
                    logging.error(f"Error processing sentence embedding: {e}")
                    embeddings.append(np.zeros(384))
                    continue

            return np.array(embeddings)

        # Get embeddings as numpy arrays
        embeddings1 = process_embeddings(sentences1, doc1_path)
        embeddings2 = process_embeddings(sentences2, doc2_path)

        # Compute similarity matrix using vectorized operations
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        above_threshold = np.argwhere(similarity_matrix >= threshold)

        # Collect matches from matrix indices
        doc1_name = os.path.basename(doc1_path)
        doc2_name = os.path.basename(doc2_path)
        matches = []
        for i, j in above_threshold:
            sent1, page1 = sentences1[i]
            sent2, page2 = sentences2[j]
            similarity = float(
                similarity_matrix[i, j]
            )  # Convert to native Python float

            matches.append(
                {
                    "doc1_sentence": sent1,
                    "doc2_sentence": sent2,
                    "similarity": similarity,
                    "doc1_page": page1,
                    "doc2_page": page2,
                    "doc1_name": doc1_name,
                    "doc2_name": doc2_name,
                }
            )

        # Cache the results
        if vectorstore:
            vectorstore.store_document_matches(doc1_path, doc2_path, threshold, matches)

        return matches

    except Exception as e:
        logging.error(f"Error comparing documents: {e}")
        return []


def main():
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Compare PDF documents and generate similarity report"
    )
    parser.add_argument(
        "--output", default="comparison_report.html", help="Output file path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=SENTENCE_SIMILARITY_THRESHOLD,
        help=f"Minimum similarity threshold (default: {SENTENCE_SIMILARITY_THRESHOLD})",
    )
    args = parser.parse_args()

    # Configuration settings
    PDF1_PATH = "/Users/matthiasneumayer/Dev/expirements/regierungsprogramme/pdf-chapter-comparator/data/PDF/protokoll-oevp-spoe-neos.pdf"
    PDF2_PATH = "/Users/matthiasneumayer/Dev/expirements/regierungsprogramme/pdf-chapter-comparator/data/PDF/protokoll.pdf"
    OUTPUT_FILE = args.output

    # Process both documents
    logging.info("Extracting text from PDFs...")
    doc1_content = extract_content_from_pdf(PDF1_PATH, vectorstore)
    doc2_content = extract_content_from_pdf(PDF2_PATH, vectorstore)

    if not doc1_content or not doc2_content:
        logging.error("Failed to extract content from one or both documents")
        return

    # Compare sentences between documents
    logging.info("Comparing sentences between documents...")
    similar_sentences = compare_documents(
        doc1_content, doc2_content, args.threshold, vectorstore
    )

    if not similar_sentences:
        logging.info("No similar content found above threshold.")
        return

    logging.info("Generating HTML report...")
    generate_html_report(
        {
            "doc1": {"path": PDF1_PATH},
            "doc2": {"path": PDF2_PATH},
            "similar_sections": [],  # No sections anymore
            "similar_sentences": similar_sentences,
        },
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
