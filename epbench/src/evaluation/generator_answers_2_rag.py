from typing import Tuple, List
from epbench.src.models.settings_wrapper import SettingsWrapper 
import pandas as pd
from openai import OpenAI
from scipy import spatial  # for calculating vector similarities for search
import numpy as np

def embed_chunks(my_chunks: List[str], 
                 answering_parameters = {'kind': 'rag', 'split': 'chapter', 'embedding_model': "text-embedding-3-small", 'embedding_batch_size': 2048}, 
                 env_file = '/repo/to/git/main/.env'):
    '''
    Source: https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search
    BATCH_SIZE: you can submit up to 2048 embedding inputs per request
    '''
    config = SettingsWrapper(_env_file = env_file)
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embedding_batch_size = answering_parameters['embedding_batch_size']
    embedding_model = answering_parameters['embedding_model']
    embeddings = []
    for batch_start in range(0, len(my_chunks), embedding_batch_size):
        batch_end = batch_start + embedding_batch_size
        batch = my_chunks[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = client.embeddings.create(model=embedding_model, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in same order as input
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    df = pd.DataFrame({"text": my_chunks, "embedding": embeddings})
    return df

def strings_ranked_by_relatedness(
    question: str,
    df: pd.DataFrame, # the embedding
    answering_parameters = {'kind': 'rag', 'split': 'chapter', 'embedding_model': "text-embedding-3-small", 'embedding_batch_size': 2048}, 
    env_file = '/repo/to/git/main/.env',
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
) -> Tuple[List[str], List[float]]:
    """
    Returns a list of strings and relatednesses, sorted from most related to least.
    Source: https://cookbook.openai.com/examples/question_answering_using_embeddings
    """
    config = SettingsWrapper(_env_file = env_file)
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    query_embedding_response = client.embeddings.create(
        model=answering_parameters['embedding_model'],
        input=question,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    top_n = answering_parameters['top_n']
    return strings[:top_n], relatednesses[:top_n]

def generate_episodic_memory_rag_prompt(book_content, question):

    prompt = f"""# Episodic Memory Benchmark

You are participating in an episodic memory test, based on the data below, which was retrieved from a book. You need to read it and internalize as if you had personally experienced the events described. After the text, you will find a question about the content. Please answer this question based solely on the information provided in the retrieved data.

## Retrieved Relevant Chunks from the Book:

{book_content}

## Question:

{question}

Please answer the question to the best of your ability, based only on the information provided in the relevant chunks above. If you are unsure about an answer, it's okay to say so. Do not invent or assume information that was not explicitly stated in the text.
"""
    return prompt

def query_message(
    question: str,
    df: pd.DataFrame,
    answering_parameters, 
    env_file
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(question, df, answering_parameters, env_file)
    book_content = ""
    for (string, relatedscore) in zip(strings, relatednesses):
        next_article = f'\n\n"""\n{string}\n"""'
        #next_article = f'\n\nFollowing section with relatedness {np.round(relatedscore, decimals=2)}\n"""\n{string}\n"""'
        book_content += next_article
        
    return generate_episodic_memory_rag_prompt(book_content, question)

def get_top_n(embedding_chunk, my_benchmark):
    # Assess the maximum number of chunks needed
    if embedding_chunk == 'chapter':
        # maximum number of chapters needed 
        # (a single question may involve up to 3 chapters for the book generated with 20 events,
        # while it may involve up to 17 chapters for the book generated with 200 events)
        top_n = max(my_benchmark.df_qa['n_chapters_correct_answer'])
    else:
        # maximum number of paragraphs needed
        # (4 times the number of chapters needed, since only 4 paragraphs are necessary to get the full view of a single chapter)
        top_n = max(my_benchmark.df_qa['n_chapters_correct_answer'])*4
    return top_n
