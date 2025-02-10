#!/usr/bin/env python3
import argparse
import concurrent.futures
import html
import logging
import re
from collections import defaultdict
from dotenv import load_dotenv
import os

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pdf_chapter_comparator.report_generator import generate_html_report

# Download required NLTK data (if not already present)
nltk.download("punkt", quiet=True)

########################################
# Configure logging
########################################
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


########################################
# PDF Extraction and Chapter Splitting
########################################
def extract_chapters_with_pages(
    pdf_path, chapter_regex=r"(?:UG|Cluster)\s+\d+\s*(?:[–-]|[:.])\s*(.*)"
):
    """
    Extracts text from the given PDF and splits it into chapters based on a regex.
    Returns a dictionary mapping chapter titles to a list of tuples:
      (sentence, page_number)
    """
    chapters = {}
    current_chapter = "Introduction"  # Default chapter name
    chapters[current_chapter] = []  # Initialize with default chapter

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                text = page.extract_text()
                if not text:
                    continue

                # Look for a chapter heading on this page.
                # Match either UG or Cluster followed by number and title
                chapter_match = re.search(chapter_regex, text, re.IGNORECASE)
                if chapter_match:
                    # Use the title part after the prefix and number
                    current_chapter = chapter_match.group(1).strip()
                    if current_chapter not in chapters:
                        chapters[current_chapter] = []

                # Tokenize the page text into sentences.
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        chapters[current_chapter].append((sentence, page_number))
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
    return chapters


########################################
# Azure OpenAI Embedding with Caching and Concurrency
########################################
class EmbeddingCache:
    def __init__(self):
        self.cache = {}

    def get(self, text):
        return self.cache.get(text)

    def set(self, text, embedding):
        self.cache[text] = embedding


# Initialize the embedding model
embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1", trust_remote_code=True
)


def get_embedding(text, deployment, cache: EmbeddingCache):
    """
    Returns the embedding for a given text using nomic-embed-text.
    Uses caching to avoid duplicate computations.
    """
    cached = cache.get(text)
    if cached is not None:
        return cached

    try:
        # Use the global embedding_model directly
        embedding = embedding_model.encode(text)
        cache.set(text, embedding)
        return embedding
    except Exception as e:
        logging.error(f"Error computing embedding for text: {text[:30]}...: {e}")
        return None


def compute_embeddings(sentences, deployment, cache):
    """
    Computes embeddings for a list of sentences concurrently.
    Returns a list of embeddings corresponding to the sentences.
    """
    embeddings = [None] * len(sentences)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(get_embedding, sentence, deployment, cache): idx
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


########################################
# LLM Verification for Similar Sentences
########################################
def verify_similarity_with_llm(sent1, sent2, chat_deployment):
    """
    Uses Azure OpenAI Chat Completion to decide whether two sentences express the same idea.
    Returns True if the LLM confirms similarity, False otherwise.
    """
    prompt = (
        f"Please analyze the following two sentences and answer with only 'Yes' or 'No'.\n\n"
        f'Sentence 1: "{sent1}"\n'
        f'Sentence 2: "{sent2}"\n\n'
        f"Do these two sentences express the same idea?"
    )

    try:
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates semantic similarity.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        logging.error(
            f"LLM verification error for sentences: {sent1[:30]}... / {sent2[:30]}...: {e}"
        )
        return False


########################################
# Sentence Comparison per Chapter (with optional LLM verification)
########################################
def compare_chapter_sentences(
    chapter,
    sentences,
    embedding_deployment,  # This parameter is no longer needed but kept for compatibility
    threshold,
    cache,
    use_llm_verification=False,
    chat_deployment=None,
):
    """
    For a given chapter, computes embeddings for the sentences and compares each sentence
    with every other sentence using cosine similarity.
    If use_llm_verification is True, an additional LLM check confirms similarity.
    Returns a list of dictionaries with the matching details.
    """
    results = []
    if len(sentences) < 2:  # Need at least 2 sentences to compare
        return results

    # Unzip sentence text and page numbers
    texts, pages = zip(*sentences)

    # Compute embeddings concurrently
    embeddings = compute_embeddings(texts, embedding_deployment, cache)

    # Ensure no None embeddings
    if any(e is None for e in embeddings):
        logging.warning(
            f"Some embeddings for chapter '{chapter}' could not be computed."
        )
        return results

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(np.vstack(embeddings))

    # Loop through the similarity matrix (upper triangle only to avoid duplicates)
    for i in range(len(sentences)):
        for j in range(
            i + 1, len(sentences)
        ):  # Start from i+1 to avoid self-comparison
            sim_score = sim_matrix[i, j]
            if sim_score >= threshold:
                sent1, page1 = sentences[i]
                sent2, page2 = sentences[j]

                # Optionally verify with LLM before adding the pair
                if use_llm_verification:
                    chat_dep = (
                        chat_deployment
                        if chat_deployment is not None
                        else embedding_deployment
                    )
                    llm_result = verify_similarity_with_llm(sent1, sent2, chat_dep)
                    if not llm_result:
                        continue

                results.append(
                    {
                        "chapter": chapter,
                        "sentence1": sent1,
                        "page1": page1,
                        "sentence2": sent2,
                        "page2": page2,
                        "similarity": sim_score,
                    }
                )
    return results


########################################
# Main Function and Argument Parsing
########################################
def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration settings
    PDF1_PATH = "/Users/matthiasneumayer/Dev/expirements/regierungsprogramme/pdf-chapter-comparator/data/PDF/protokoll-oevp-spoe-neos.pdf"
    PDF2_PATH = "/Users/matthiasneumayer/Dev/expirements/regierungsprogramme/pdf-chapter-comparator/data/PDF/protokoll.pdf"
    SIMILARITY_THRESHOLD = 0.55
    OUTPUT_FILE = "comparison_report.html"
    CHAPTER_REGEX = r"(?:UG|Cluster)\s+\d+\s*(?:[–-]|[:.])\s*(.*)"
    USE_LLM_VERIFICATION = False  # Since we're not using OpenAI, this should be False

    logging.info("Extracting chapters from PDFs...")
    chapters1 = extract_chapters_with_pages(PDF1_PATH, CHAPTER_REGEX)
    chapters2 = extract_chapters_with_pages(PDF2_PATH, CHAPTER_REGEX)

    logging.info("Comparing sentences within each document's chapters...")
    cache = EmbeddingCache()
    results_by_chapter = defaultdict(list)

    # Compare sentences within PDF1's chapters
    for chapter, sentences in chapters1.items():
        matches = compare_chapter_sentences(
            f"{chapter} (Document 1)",
            sentences,
            embedding_model,
            SIMILARITY_THRESHOLD,
            cache,
            use_llm_verification=USE_LLM_VERIFICATION,
        )
        if matches:
            results_by_chapter[f"{chapter} (Document 1)"].extend(matches)

    # Compare sentences within PDF2's chapters
    for chapter, sentences in chapters2.items():
        matches = compare_chapter_sentences(
            f"{chapter} (Document 2)",
            sentences,
            embedding_model,
            SIMILARITY_THRESHOLD,
            cache,
            use_llm_verification=USE_LLM_VERIFICATION,
        )
        if matches:
            results_by_chapter[f"{chapter} (Document 2)"].extend(matches)

    if not results_by_chapter:
        logging.info("No matching sentence pairs found above the similarity threshold.")
    else:
        logging.info("Generating HTML report...")
        generate_html_report(results_by_chapter, OUTPUT_FILE)


if __name__ == "__main__":
    main()
