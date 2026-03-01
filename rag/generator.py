import os
from huggingface_hub import InferenceClient

try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")


SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the Swiggy Annual Report FY 2023-24.
Rules:
- Only use information from the provided context. Never use outside knowledge.
- If the answer is not in the context, say exactly: "I couldn't find this information in the Swiggy Annual Report."
- Always cite the page number at the end like (Page X).
- Keep answers factual and concise."""


class AnswerGenerator:
    def __init__(self):
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            token=HF_TOKEN
        )

    def generate(self, question, context):
        prompt = f"""<s>[INST] {SYSTEM_PROMPT}

Context from Swiggy Annual Report:
{context}

Question: {question}

Answer based only on the context above: [/INST]"""

        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.1,
                repetition_penalty=1.1,
                do_sample=False
            )
            return response.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}. Please try again."
