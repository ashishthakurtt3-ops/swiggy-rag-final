import os
from groq import Groq

try:
    import streamlit as st
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the Swiggy Annual Report FY 2023-24.
Rules:
- Only use information from the provided context. Never use outside knowledge.
- If the answer is not in the context, say: "I couldn't find this information in the Swiggy Annual Report."
- Always cite the page number at the end like (Page X).
- Keep answers factual and concise."""


class AnswerGenerator:
    def __init__(self):
        self.client = Groq(api_key=api_key)

    def generate(self, question, context):
        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context from Swiggy Annual Report:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above:"}
                ],
                temperature=0.1,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"