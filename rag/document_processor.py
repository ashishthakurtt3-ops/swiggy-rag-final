import re
import fitz
from dataclasses import dataclass, field
from typing import List


@dataclass
class TextChunk:
    text: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    def __init__(self, chunk_size=800, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, pdf_path):
        pages = self._load_pdf(pdf_path)
        chunks = self._chunk_pages(pages)
        print(f"Got {len(pages)} pages, created {len(chunks)} chunks")
        return chunks

    def _load_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            text = self._clean(text)
            if text.strip():
                pages.append({"page_number": i + 1, "text": text})
        doc.close()
        return pages

    def _clean(self, text):
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [l for l in text.split("\n") if not re.fullmatch(r"\s*\d{1,4}\s*", l)]
        text = "\n".join(lines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _chunk_pages(self, pages):
        chunks = []
        idx = 0
        for page in pages:
            text = page["text"]
            pnum = page["page_number"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                if end < len(text):
                    end = self._find_split(text, end)
                chunk_text = text[start:end].strip()
                if len(chunk_text) > 50:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        page_number=pnum,
                        chunk_index=idx,
                        metadata={"source": "Swiggy Annual Report FY 2023-24", "page": pnum}
                    ))
                    idx += 1
                start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def _find_split(self, text, pos):
        segment = text[max(0, pos - 200):pos]
        for sep in ["\n\n", ". ", ".\n", "\n"]:
            i = segment.rfind(sep)
            if i != -1:
                return max(0, pos - 200) + i + len(sep)
        return pos
