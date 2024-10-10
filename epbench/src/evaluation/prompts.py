
def generate_episodic_memory_prompt(book_content, question):
    prompt = f"""# Episodic Memory Benchmark

You are participating in an episodic memory test. You will be presented with a text to read and internalize as if you had personally experienced the events described. After the text, you will find a question about the content. Please answer this question based solely on the information provided in the text.

## The Text to Memorize:

{book_content}

## Question:

{question}

Please answer the question to the best of your ability, based only on the information provided in the text above. If you are unsure about an answer, it's okay to say so. Do not invent or assume information that was not explicitly stated in the text.
"""
    return prompt