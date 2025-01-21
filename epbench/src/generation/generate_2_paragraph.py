from epbench.src.models.models_wrapper import ModelsWrapper
from epbench.src.models.settings_wrapper import SettingsWrapper
from epbench.src.generation.prompts import generate_prompts, system_prompt_func
from epbench.src.generation.generate_1_events_and_meta_events import generate_and_export_events_and_meta_events_func
from epbench.src.io.io import paragraph_filepath_func, export_list, import_list
from epbench.src.generation.verification_llm import has_passed_direct_and_llm_verifications_func
import logging

def generate_paragraphs_func(
    prompt_parameters = {'nb_events': 10, 'name_universe': 'default', 'name_styles': 'default', 'seed': 0},
    model_parameters = {'model_name': 'gpt-4o-2024-05-13', 'max_new_tokens': 4096},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env',
    iterations = None,
    rechecking = True):

    # model parameters
    model_name = model_parameters['model_name']
    max_new_tokens = model_parameters['max_new_tokens']
    system_prompt = system_prompt_func()

    config = SettingsWrapper(_env_file = env_file)

    events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
    prompts = generate_prompts(events, meta_events, prompt_parameters['name_styles'])

    # iterations
    if iterations is None:
        iterations = [0]*prompt_parameters['nb_events']

    generated_paragraphs = []
    for event_index in range(len(prompts)):
        user_prompt = prompts[event_index]
        iteration = iterations[event_index]
        data_paragraphs_filepath = paragraph_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
        if not data_paragraphs_filepath.is_file():
            print("Generate " + str(event_index) + "/" + str(len(prompts)-1))
            # only initialize the model if needed, and only initialize it once 
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_name, config)
            # generate the content
            out = my_model.generate(user_prompt = user_prompt, system_prompt = system_prompt, max_new_tokens = max_new_tokens)
            data_paragraphs_filepath.parent.mkdir(parents=True, exist_ok=True)
            export_list(out, data_paragraphs_filepath)
        generated_paragraph = import_list(data_paragraphs_filepath)
        generated_paragraphs.append(generated_paragraph)

    return generated_paragraphs

def iteration_verbose_func(i, has_direct_verif_vector, final = False):
    idx_issues = [index for index, value in enumerate(has_direct_verif_vector) if value == False]
    percentage = (len(idx_issues) / len(has_direct_verif_vector)) * 100
    percentage_with_issues = f"{percentage:.2f}%"
    ratio_with_issues = f"{str(len(idx_issues))}/{str(len(has_direct_verif_vector))}"
    if final:
        final_str = "final "
    else:
        final_str = ""
    str_output = f"At {final_str}iteration {str(i)}, {percentage_with_issues} remaining with issues ({ratio_with_issues}), for index: {idx_issues}."
    return str_output

def iterative_generate_paragraphs_func(
    prompt_parameters = {'nb_events': 10, 'name_universe': 'default', 'name_styles': 'default', 'seed': 0},
    model_parameters = {'model_name': 'gpt-4o-2024-05-13', 'max_new_tokens': 4096, 'itermax': 10},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env',
    verbose = True,
    rechecking = True):
    itermax = model_parameters['itermax']
    # The iterations parameters is automatically iterated
    iterations = [0]*prompt_parameters['nb_events']
    events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
    generated_paragraphs = generate_paragraphs_func(prompt_parameters, model_parameters, data_folder, env_file, iterations, rechecking)
    for i in range(itermax-1):
        has_verif_vector = has_passed_direct_and_llm_verifications_func(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder, env_file)
        if not all(has_verif_vector):
            if verbose:
                print(iteration_verbose_func(i, has_verif_vector))
            # erase the previous iteration vector
            iterations = [i+1 if not v else i for (i,v) in zip(iterations, has_verif_vector)]
            # load or regenerate
            generated_paragraphs = generate_paragraphs_func(prompt_parameters, model_parameters, data_folder, env_file, iterations, rechecking)
        else:
            if verbose:
                print(iteration_verbose_func(i, has_verif_vector)) # no issue remaining
            break
    # get the final `has_verif_vector` (e.g. for itermax=1, the loop is passed, and we still need to get this element)
    has_verif_vector = has_passed_direct_and_llm_verifications_func(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder, env_file)
    if verbose:
        print(iteration_verbose_func(itermax-1, has_verif_vector, final = True))
        if not all(has_verif_vector):
            print("itermax reached but some events still did not pass the verification")

    # further filter the ones that does not pass after itermax iterations
    # (the length of the list if still the original one, but the one that did not pass the verification after itermax
    # are set to None)
    generated_paragraphs_filtered = [p if v else None for (p,v) in zip(generated_paragraphs, has_verif_vector)]
    return generated_paragraphs_filtered, has_verif_vector
