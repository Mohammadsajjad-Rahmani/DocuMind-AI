import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DocuMindEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.?!۔\n])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def create_chunks(self, text: str) -> List[str]:
        return self._split_into_sentences(text)

    def generate_summary(self, text: str, top_n: int = 3) -> Tuple[str, int]:
        sentences = self.create_chunks(text)
        if len(sentences) <= top_n:
            return "\n\n".join(sentences), len(sentences)
        
        embeddings = self.model.encode(sentences)
        doc_embedding = embeddings.mean(axis=0).reshape(1, -1)
        
        scores = cosine_similarity(embeddings, doc_embedding).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_indices.sort()
        
        summary = "\n\n".join([sentences[i] for i in top_indices])
        return summary, len(sentences)

    def answer_question(self, question: str, text: str, top_k: int = 1) -> Tuple[str, List[str]]:
        sentences = self.create_chunks(text)
        if not sentences:
            return "متنی برای بررسی یافت نشد.", []
        
        sentence_embeddings = self.model.encode(sentences)
        question_embedding = self.model.encode([question])
        
        similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
        
        best_indices = similarities.argsort()[::-1]
        
        relevant_sentences = []
        for idx in best_indices[:top_k]:
            if similarities[idx] > 0.25:
                relevant_sentences.append(sentences[idx])
        
        if not relevant_sentences:
            return "پاسخ دقیقی برای این سوال در متن یافت نشد.", []
        
        answer = "\n\n".join(relevant_sentences)
        return answer, relevant_sentences

doc_engine = DocuMindEngine()