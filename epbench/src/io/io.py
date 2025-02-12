import json
from pathlib import Path
import jsonlines

def export_list(data_list, filename):
    '''
    Dumping a list into a json file
    '''
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)

def import_list(filename):
    '''
    Loading a json file to a list
    '''
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def export_jsonl(res, filename):
    '''
    Dumping into a jsonl file (for fine-tuning)
    '''
    with jsonlines.open(filename, 'w') as writer:
        writer.write_all(res)

def data_folder_experiment_func(prompt_parameters):
    '''
    Folder path of the experiment
    '''
    return f"U{prompt_parameters['name_universe']}_S{prompt_parameters['name_styles']}_seed{prompt_parameters['seed']}"

def paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters):
    '''
    File name of the generated paragraph
    '''
    prompt_str = f"e{event_index}_iter{iteration}"
    model_str = f"model_{model_parameters['model_name']}"
    return f"{model_str}_{prompt_str}.json"

def paragraph_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters):
    '''
    File path of the generated paragraph
    '''
    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "paragraphs" / paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters)
    return data_paragraphs_filepath

def paragraph_verif_direct_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters):
    '''
    File path of the direct verification log for the target paragraph
    '''
    verifdirect_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "paragraphs_verif_direct" / paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters)
    return verifdirect_paragraphs_filepath

def paragraph_verif_llm_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters):
    '''
    File path of the LLM verification log for the target paragraph
    '''
    verifllm_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "paragraphs_verif_llm" / paragraph_file_name_func(event_index, iteration, prompt_parameters, model_parameters)
    return verifllm_paragraphs_filepath

def book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters):
    '''
    File name of the generated book
    '''
    indexing_str = f"I{book_parameters['indexing']}"
    model_str = f"model_{model_parameters['model_name']}_itermax_{model_parameters['itermax']}"
    this_book_str = f"nbchapters_{nb_chapters}_nbtokens_{nb_tokens}"
    return f"{model_str}_{indexing_str}_{this_book_str}"

def book_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters):
    '''
    File path of the generated book
    '''
    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "books" / book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters)
    return data_paragraphs_filepath

def answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    Full directory path of the answers
    '''
    data_paragraphs_filepath = Path(data_folder) / data_folder_experiment_func(prompt_parameters) / "answers" / book_dir_name_func(nb_chapters, nb_tokens, book_parameters, prompt_parameters, model_parameters) / answer_dir_name_func(answering_parameters)
    return data_paragraphs_filepath

def answer_dir_name_func(answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096}):
    '''
    Final folder path of the answers
    '''
    if answering_parameters['kind'] == 'prompting':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        return f"answered_by_{kind_str}_{model_str}"
    elif answering_parameters['kind'] == 'rag':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        chunk_str = f"chunk_{answering_parameters['embedding_chunk']}_top{answering_parameters['top_n']}"
        return f"answered_by_{kind_str}_{model_str}_{chunk_str}"
    elif answering_parameters['kind'] == 'ftuning':
        kind_str = f"kind_{answering_parameters['kind']}"
        model_str = f"model_{answering_parameters['model_name']}"
        ftuning_str = f"fpolicy_{answering_parameters['ftuning_input_data_policy']}"
        return f"answered_by_{kind_str}_{model_str}_{ftuning_str}"

def answer_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json answer from the question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    filename = f"{str(q)}.json"
    answer_filepath = my_dirpath / 'raw_answers' / filename
    return answer_filepath

def answer_reasoning_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json reasoning from the question with index q (only for reasoning models, e.g. DeepSeek)
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    filename = f"{str(q)}.json"
    answer_filepath = my_dirpath / 'raw_reasoning' / filename
    return answer_filepath

def evaluate_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json evaluation of the answer from the question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    filename = f"{str(q)}.json"
    evaluate_filepath = my_dirpath / 'evaluated_answers' / filename
    return evaluate_filepath

def chronological_filepath_func(q: int, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters):
    '''
    File path of a single json chronological of the answer from the question with index q
    '''
    my_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
    filename = f"{str(q)}.json"
    chronological_filepath = my_dirpath / 'chronological_answers' / filename
    return chronological_filepath
