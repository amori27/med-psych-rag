from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from src.config import Config


class LocalRetriever:
    def __init__(self, index_path: str = "local_index.pkl"):
        self.index_path = Path(index_path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: list[Document] = []
        self._matrix = None

    def add_documents(self, documents: list[Document]):
        self.documents.extend(documents)
        texts = [d.page_content for d in self.documents]
        self._matrix = self.vectorizer.fit_transform(texts)
        self._save()

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        if not self.documents:
            return []
        k = k or Config.top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        return [self.documents[i] for i in top_indices if scores[i] > 0]

    def _save(self):
        with open(self.index_path, "wb") as f:
            pickle.dump({"docs": self.documents, "matrix": self._matrix, "vec": self.vectorizer}, f)

    def load(self):
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data["docs"]
            self._matrix = data["matrix"]
            self.vectorizer = data["vec"]
            return True
        return False
