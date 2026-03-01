from transformers import pipeline

# runs 100% locally on the server, zero API key needed
_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
        )
    return _pipe


SYSTEM = """Answer the question using only the context below from the Swiggy Annual Report FY 2023-24.
If the answer is not in the context, say: I couldn't find this in the Swiggy Annual Report.

Context:
{context}

Question: {question}
Answer:"""


class AnswerGenerator:
    def __init__(self):
        pass

    def generate(self, question, context):
        try:
            pipe = get_pipeline()
            prompt = SYSTEM.format(
                context=context[:2000],
                question=question
            )
            result = pipe(prompt, max_new_tokens=200, do_sample=False)
            answer = result[0]["generated_text"].strip()
            if not answer:
                return "I couldn't find this information in the Swiggy Annual Report."
            return answer
        except Exception as e:
            return f"Error: {str(e)}"