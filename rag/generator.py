from transformers import pipeline

_pipe = None

def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
    return _pipe


class AnswerGenerator:
    def __init__(self):
        pass

    def generate(self, question, context):
        try:
            pipe = get_pipeline()

            # split context into chunks and find best answer across all of them
            chunks = context.split("---")
            best_answer = ""
            best_score = 0

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    result = pipe(question=question, context=chunk[:2000])
                    if result["score"] > best_score:
                        best_score = result["score"]
                        best_answer = result["answer"]
                except:
                    continue

            if not best_answer or best_score < 0.01:
                return "I couldn't find this information in the Swiggy Annual Report."

            return best_answer

        except Exception as e:
            return f"Error: {str(e)}"