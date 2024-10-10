from epbench.src.models.models_wrapper import ModelsWrapper
from epbench.src.models.settings_wrapper import SettingsWrapper
from epbench.src.generation.prompts import verification_prompt_func, system_prompt_verification_func
from epbench.src.io.io import paragraph_filepath_func, paragraph_verif_llm_filepath_func, export_list, import_list
from epbench.src.generation.verification_direct import has_passed_direct_verification_func, export_direct_verif_vector
import json
import re

def extract_json_from_text(text):
    # Find the JSON content using a regular expression
    json_match = re.search(r'\{[^}]+\}', text)
    
    if json_match:
        json_string = json_match.group(0)
        try:
            # Parse the JSON string into a Python dictionary
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            #print("Error: Invalid JSON format")
            return None
    else:
        #print("Error: No JSON content found in the text")
        return None

def extract_dict_results(text):
    json_results = extract_json_from_text(text)
    if json_results:
        if len(json_results) == 4:
            dict_results = {'location': json_results['1'], 'date': json_results['2'], 'entity': json_results['3'], 'content': json_results['4']}
            if all(isinstance(value, bool) for value in dict_results.values()):
                return dict_results
    # in all other cases, return None
    return None

def has_passed_llm_verification_through_model_func(paragraphs, my_model, max_new_tokens):
    results = my_model.generate(user_prompt = verification_prompt_func(paragraphs), 
                                system_prompt = system_prompt_verification_func(),
                                max_new_tokens = max_new_tokens)
    results_dict = extract_dict_results(results)
    issues = [key for key, value in results_dict.items() if value is False]
    has_passed_llm_verification = False
    if len(issues) == 0:
        has_passed_llm_verification = True
    return has_passed_llm_verification, issues, results_dict, results

def generate_has_passed_llm_verification_func(
        event_indexes = [0],
        iterations = [0],
        prompt_parameters = {'nb_events': 10, 'name_universe': 'default', 'name_styles': 'default', 'seed': 0},
        model_parameters = {'model_name': 'gpt-4o-2024-05-13', 'max_new_tokens': 4096},
        model_verification_parameters = {'model_name': 'gpt-4o-2024-05-13', 'max_new_tokens': 4096},
        data_folder = '/repo/to/git/main/epbench/data',
        env_file = '/repo/to/git/main/.env'):
    
    config = SettingsWrapper(_env_file = env_file)
    
    verification_paragraphs = []
    for iteration, event_index in zip(iterations, event_indexes):
        verifllm_paragraphs_filepath = paragraph_verif_llm_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
        if not verifllm_paragraphs_filepath.is_file():
            data_paragraphs_filepath = paragraph_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
            if not data_paragraphs_filepath.is_file():
                print(data_paragraphs_filepath)
                raise ValueError('the paragraphs should be generated before being verified')
            paragraphs = import_list(data_paragraphs_filepath)
            print("Verification prompt " + str(event_index) + "/" + str(prompt_parameters['nb_events']-1))
            # only initialize the model if needed, and only initialize it once 
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_verification_parameters['model_name'], config)
            out = has_passed_llm_verification_through_model_func(paragraphs, my_model, max_new_tokens = model_verification_parameters['max_new_tokens'])
            verifllm_paragraphs_filepath.parent.mkdir(parents=True, exist_ok=True)
            export_list(out, verifllm_paragraphs_filepath)
        verification_paragraph = import_list(verifllm_paragraphs_filepath)
        verification_paragraphs.append(verification_paragraph)
    return verification_paragraphs

def replace_at_indices(l, indices, values):
    res = l.copy()
    for index, value in zip(indices, values):
        res[index] = value
    return res

def has_further_passed_llm_verification_func(has_direct_verif_vector, iterations, prompt_parameters, model_parameters, data_folder, env_file):
    # extract the paragraphs that pass the direct checking, and for which we further do the llm verification
    event_indexes_passing_direct = [index for index, value in enumerate(has_direct_verif_vector) if value]
    iterations_passing_direct = [iterations[index] for index, value in enumerate(has_direct_verif_vector) if value]

    # select the verification model: we choose to verify with the same model as for generation
    model_verification_parameters = model_parameters

    # extract the full output given by the prompts. The first element of each element of the list is the global summary either True (passing the 4 questions), or False otherwise
    verif_out = generate_has_passed_llm_verification_func(event_indexes_passing_direct, iterations_passing_direct, prompt_parameters, model_parameters, model_verification_parameters, data_folder, env_file)
    #print(res)

    # only extract the list e.g. [False, True, True] corresponding to passing all the 4 questions (True) or not (False) for the corresponding
    # event index `event_indexes_passing_direct` and iteration indexes `iterations_passing_direct`.
    verif_out_binary = [x[0] for x in verif_out]

    has_verif_vector = replace_at_indices(has_direct_verif_vector, event_indexes_passing_direct, verif_out_binary)
    return has_verif_vector

def has_passed_direct_and_llm_verifications_func(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder, env_file):
    has_direct_verif_vector = [has_passed_direct_verification_func(p, e, m)[0] for (p, e, m) in zip(generated_paragraphs, events, meta_events)]

    # export the direct verification outputs
    export_direct_verif_vector(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder)

    # further adding the llm verification for the indexes that already pass the direct verification
    has_verif_vector = has_further_passed_llm_verification_func(has_direct_verif_vector, iterations, prompt_parameters, model_parameters, data_folder, env_file)
    return has_verif_vector
