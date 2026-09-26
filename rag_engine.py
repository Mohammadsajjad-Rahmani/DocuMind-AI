import re
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DocuMindEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def create_chunks(
        self,
        text: str,
        chunk_size: int = 400,
        overlap: int = 50
    ) -> List[str]:
        """Split text into chunks while preserving a small overlap."""

        sentences = re.split(r"(?<=[.?!۔\n])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # Keep the last part of the previous chunk as overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def generate_summary(
        self,
        text: str,
        top_n: int = 3
    ) -> Tuple[str, int]:

        chunks = self.create_chunks(text)

        if len(chunks) <= top_n:
            return "\n\n".join(chunks), len(chunks)

        embeddings = self.model.encode(chunks)

        doc_embedding = embeddings.mean(axis=0).reshape(1, -1)

        scores = cosine_similarity(
            embeddings,
            doc_embedding
        ).flatten()

        top_indices = scores.argsort()[-top_n:]

        # Preserve original document order
        top_indices.sort()

        summary = "\n\n".join(
            chunks[i] for i in top_indices
        )

        return summary, len(chunks)

    def answer_question(
        self,
        question: str,
        text: str,
        top_k: int = 2
    ) -> Tuple[str, List[str]]:

        chunks = self.create_chunks(text)

        if not chunks:
            return "متنی برای بررسی یافت نشد.", []

        chunk_embeddings = self.model.encode(chunks)
        question_embedding = self.model.encode([question])

        similarities = cosine_similarity(
            question_embedding,
            chunk_embeddings
        )[0]

        # Sort chunks by relevance
        ranked_indices = similarities.argsort()[::-1]

        best_score = similarities[ranked_indices[0]]

        # Minimum similarity required to consider the question
        # relevant to the document.
        min_threshold = 0.40

        if best_score < min_threshold:
            return "پاسخ دقیقی برای این سوال در متن یافت نشد.", []

        relevant_chunks = []

        for idx in ranked_indices[:top_k]:
            score = similarities[idx]

            # Ignore weak matches
            if score < min_threshold:
                continue

            # Only keep secondary results that are reasonably
            # close to the best result.
            if score < best_score * 0.75:
                continue

            relevant_chunks.append(chunks[idx])

        if not relevant_chunks:
            return "پاسخ دقیقی برای این سوال در متن یافت نشد.", []

        answer = "\n\n".join(relevant_chunks)

        return answer, relevant_chunks


doc_engine = DocuMindEngine()