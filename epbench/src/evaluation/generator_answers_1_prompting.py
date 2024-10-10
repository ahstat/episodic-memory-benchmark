from epbench.src.models.settings_wrapper import SettingsWrapper 
from epbench.src.models.models_wrapper import ModelsWrapper
from epbench.src.io.io import answer_filepath_func, evaluate_filepath_func, chronological_filepath_func, import_list, export_list
from epbench.src.evaluation.scoring_answers import evaluate_answer, evaluate_chronological
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.evaluation.prompts import generate_episodic_memory_prompt
from epbench.src.evaluation.generator_answers_2_rag import query_message
import os
import pandas as pd
import re
import time

def generate_answers_func(
    my_benchmark: BenchmarkGenerationWrapper,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env',
    my_embedding = None):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    # model parameters: using the model to evaluate
    model_name = answering_parameters['model_name'] 
    max_new_tokens = answering_parameters['max_new_tokens']
    system_prompt = "You are an expert in memory tests."
    sleeping_time = answering_parameters['sleeping_time']
    
    config = SettingsWrapper(_env_file = env_file)

    book = my_benchmark.get_book()
    df_qa = my_benchmark.get_df_qa()
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()

    # loop
    generated_answers = []
    for q in range(len(df_qa)):
        answer_filepath = answer_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
        
        if not answer_filepath.is_file():
            question = df_qa.iloc[q]['question']
            correct_answer = df_qa.iloc[q]['correct_answer']
            print(f"Generate {str(q)} / {str(len(df_qa)-1)} [{correct_answer}for question {question}]")
            # only initialize the model if needed, and only initialize it once 
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_name, config)
            # generate the content
            if answering_parameters['kind'] == 'prompting': # context, my_embedding is None
                user_prompt = generate_episodic_memory_prompt(book, question)
            elif answering_parameters['kind'] == 'rag': # rag, there is an embedding
                user_prompt = query_message(question, my_embedding, answering_parameters, env_file)
            elif answering_parameters['kind'] == 'ftuning':
                user_prompt = my_benchmark.get_user_prompt_for_finetuning(question)
            if q == 0:
                print("[begin example of a prompt]")
                print(user_prompt)
                print("[end example of a prompt]")
            out = my_model.generate(user_prompt = user_prompt, system_prompt = system_prompt, max_new_tokens = max_new_tokens)
            print(f"sleeping for {sleeping_time} seconds")
            time.sleep(sleeping_time)
            print("woke up")
            answer_filepath.parent.mkdir(parents=True, exist_ok=True)
            print(answer_filepath)
            export_list(out, answer_filepath)
        generated_answer = import_list(answer_filepath)
        generated_answers.append(generated_answer)

    df_generated_answers = pd.concat([df_qa, pd.DataFrame({'llm_answer':generated_answers})], axis = 1)

    return df_generated_answers

def is_valid_chapter_string(s):
    pattern = r'^Full chapter \d+$'
    return bool(re.match(pattern, s))

def extract_chapter_number(s):
    match = re.search(r"Full chapter (\d+)", s)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("String does not match the expected format")

def generate_evaluation_func(
    my_benchmark: BenchmarkGenerationWrapper,
    df_generated_answers,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env'):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    # model parameters
    model_name = model_parameters['model_name'] # using the model that built the benchmark, not the one answering the questions
    
    config = SettingsWrapper(_env_file = env_file)

    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    split_chapters = my_benchmark.split_chapters

    # question/true answer and additionally containing the generated answers
    df_qa2 = df_generated_answers
    generated_evaluations = []

    # loop
    for q in range(len(df_qa2)):
        evaluate_filepath = evaluate_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)

        if not evaluate_filepath.is_file():
            question = df_qa2.iloc[q]['question'] # just for the printing
            llm_answer = df_qa2.iloc[q]['llm_answer']
            correct_answer = df_qa2.iloc[q]['correct_answer']
            retrieval_type = df_qa2.iloc[q]['retrieval_type']
            get_style = df_qa2.iloc[q]['get']
            print(f"Evaluate {str(q)} / {str(len(df_qa2)-1)} [question {question}]")
            # only initialize the model if needed, and only initialize it once 
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_name, config)

            # update the answer for full events
            if len(correct_answer) == 1:
                #print(correct_answer[0])
                if is_valid_chapter_string(correct_answer[0]):
                    #print('need to change with actual chapter')
                    chapter_number = extract_chapter_number(correct_answer[0])
                    #print(chapter_number)
                    correct_answer_long = split_chapters[chapter_number] # does not need to be a list in this case
                    #print("[begin book chapter]")
                    #print(correct_answer_long)
                    #print("[end book chapter]")
                else:
                    correct_answer_long = None
            else:
                correct_answer_long = None

            # generate the content
            out = evaluate_answer(llm_answer, correct_answer, retrieval_type, my_model, correct_answer_long, get_style)
            evaluate_filepath.parent.mkdir(parents=True, exist_ok=True)
            #print(evaluate_filepath)
            export_list(out, evaluate_filepath)
        generated_evaluation = import_list(evaluate_filepath)
        generated_evaluations.append(generated_evaluation)

    #df_generated_answers = pd.concat([df_qa2, pd.DataFrame({'llm_answer':generated_answers})], axis = 1)
        
    df_generated_evaluations = pd.DataFrame(generated_evaluations)
    df_generated_evaluations = pd.concat([df_qa2, df_generated_evaluations], axis = 1)

    return df_generated_evaluations








def generate_chronological_func(
    my_benchmark: BenchmarkGenerationWrapper,
    df_generated_evaluations,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env'):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    # model parameters
    model_name = model_parameters['model_name'] # using the model that built the benchmark, not the one answering the questions
    
    config = SettingsWrapper(_env_file = env_file)

    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    split_chapters = my_benchmark.split_chapters

    df_qa3 = df_generated_evaluations

    generated_chronologicals = []

    # loop
    for q in range(len(df_qa3)):
        if df_qa3.iloc[q]['get'] == 'chronological': # only consider the chronological questions
            chronological_filepath = chronological_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)

            if not chronological_filepath.is_file():
                predicted_items = df_qa3.iloc[q]['predicted_items']
                groundtruth_items = df_qa3.iloc[q]['groundtruth_items']
                question = df_qa3.iloc[q]['question'] # just for the printing
                print(f"Evaluate {str(q)} / {str(len(df_qa3)-1)} [question {question}]")
                # only initialize the model if needed, and only initialize it once 
                try:
                    my_model
                except NameError:
                    my_model = ModelsWrapper(model_name, config)

                # generate the content
                out = evaluate_chronological(groundtruth_items, predicted_items, my_model)
                chronological_filepath.parent.mkdir(parents=True, exist_ok=True)
                #print(evaluate_filepath)
                export_list(out, chronological_filepath)
            generated_chronological = import_list(chronological_filepath)
            generated_chronologicals.append(generated_chronological)

    df_generated_chronological = pd.DataFrame(generated_chronologicals)

    return df_generated_chronological



