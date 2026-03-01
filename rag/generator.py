from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# runs fully locally, no API key needed
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)


class AnswerGenerator:
    def __init__(self):
        pass

    def generate(self, question, context):
        try:
            # roberta-base-squad2 has a 512 token limit so trim context if needed
            trimmed_context = context[:3000]

            result = qa_pipeline(
                question=question,
                context=trimmed_context
            )

            score = result.get("score", 0)
            answer = result.get("answer", "").strip()

            if not answer or score < 0.05:
                return "I couldn't find a clear answer to this in the Swiggy Annual Report. Try rephrasing your question."

            return answer

        except Exception as e:
            return f"Error generating answer: {str(e)}"