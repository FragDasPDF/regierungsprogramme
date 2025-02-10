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
SENTENCE_SIMILARITY_THRESHOLD = (
    0.85  # Default similarity threshold for sentence matching
)

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
                return stored_content

        content = {"text": "", "pages": [], "pdf_path": pdf_path}

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

                    content["text"] += text + "\n"
                    content["pages"].append(page_num)

                except Exception as e:
                    logging.error(f"Error processing page {page_num}: {str(e)}")
                    continue

        # Compute embedding for the entire content
        if content["text"]:
            embedding = embedding_model.encode(content["text"])
            content["embeddings"] = np.array(embedding).reshape(1, -1)[0]

            # Store in vectorstore if available
            if vectorstore:
                vectorstore.store_sections_for_pdf(pdf_path, [content])

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


def find_page_number(sentence, content, pages):
    """Helper function to find the page number for a sentence"""
    sentence_pos = content.find(sentence)
    if sentence_pos == -1:
        return pages[0]  # Default to first page if not found

    content_before = content[:sentence_pos]
    newline_count = content_before.count("\n")
    page_index = min(newline_count // 40, len(pages) - 1)  # Assuming ~40 lines per page
    return pages[page_index]


def compare_documents(doc1_content, doc2_content, threshold=0.85, vectorstore=None):
    """Compare sentences between two documents"""
    try:
        # Extract sentences with their page numbers
        sentences1 = [
            (sent, find_page_number(sent, doc1_content["text"], doc1_content["pages"]))
            for sent in sent_tokenize(doc1_content["text"].strip())
            if sent.strip()
        ]
        sentences2 = [
            (sent, find_page_number(sent, doc2_content["text"], doc2_content["pages"]))
            for sent in sent_tokenize(doc2_content["text"].strip())
            if sent.strip()
        ]

        if not sentences1 or not sentences2:
            return []

        # Get embeddings for all sentences
        embeddings1 = []
        embeddings2 = []
        doc1_name = os.path.basename(doc1_content["pdf_path"])
        doc2_name = os.path.basename(doc2_content["pdf_path"])

        # Process doc1 sentences
        for sent, page in sentences1:
            # Try to get stored embedding
            stored_data = (
                vectorstore.get_sentence_embedding(sent, doc1_content["pdf_path"])
                if vectorstore
                else None
            )

            if stored_data is not None:
                emb = np.array(stored_data["embedding"])
                embeddings1.append(emb)
            else:
                emb = embedding_model.encode(sent)
                embeddings1.append(emb)
                if vectorstore:
                    vectorstore.store_sentence_embedding(
                        sent, doc1_content["pdf_path"], emb, page, doc1_name
                    )

        # Process doc2 sentences
        for sent, page in sentences2:
            # Try to get stored embedding
            stored_data = (
                vectorstore.get_sentence_embedding(sent, doc2_content["pdf_path"])
                if vectorstore
                else None
            )

            if stored_data is not None:
                emb = np.array(stored_data["embedding"])
                embeddings2.append(emb)
            else:
                emb = embedding_model.encode(sent)
                embeddings2.append(emb)
                if vectorstore:
                    vectorstore.store_sentence_embedding(
                        sent, doc2_content["pdf_path"], emb, page, doc2_name
                    )

        # Compare sentences
        matches = []
        for i, ((sent1, page1), emb1) in enumerate(zip(sentences1, embeddings1)):
            for j, ((sent2, page2), emb2) in enumerate(zip(sentences2, embeddings2)):
                # Ensure embeddings are numpy arrays and properly shaped
                emb1_reshaped = np.array(emb1).reshape(1, -1)
                emb2_reshaped = np.array(emb2).reshape(1, -1)

                similarity = cosine_similarity(emb1_reshaped, emb2_reshaped)[0][0]

                if similarity >= threshold:
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
