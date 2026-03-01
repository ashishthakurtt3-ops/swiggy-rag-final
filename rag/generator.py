import os
from huggingface_hub import InferenceClient

try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")


class AnswerGenerator:
    def __init__(self):
        self.client = InferenceClient(
            model="HuggingFaceH4/zephyr-7b-beta",
            token=HF_TOKEN
        )

    def generate(self, question, context):
        prompt = f"""<|system|>
You are a helpful assistant that answers questions strictly from the Swiggy Annual Report FY 2023-24.
Rules:
- Only use information from the provided context.
- If the answer is not in the context, say: "I couldn't find this information in the Swiggy Annual Report."
- Cite the page number at the end like (Page X).
- Be factual and concise.</s>
<|user|>
Context from Swiggy Annual Report:
{context}

Question: {question}</s>
<|assistant|>"""

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
            return f"Error: {str(e)}"