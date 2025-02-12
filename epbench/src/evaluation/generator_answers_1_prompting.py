from epbench.src.models.settings_wrapper import SettingsWrapper 
from epbench.src.models.models_wrapper import ModelsWrapper
from epbench.src.io.io import answer_filepath_func, answer_reasoning_filepath_func, evaluate_filepath_func, chronological_filepath_func, import_list, export_list
from epbench.src.evaluation.scoring_answers import evaluate_answer, evaluate_chronological
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.evaluation.prompts import generate_episodic_memory_prompt
from epbench.src.evaluation.generator_answers_2_rag import query_message
from epbench.src.generation.printing import split_chapters_func
import os
import pandas as pd
import re
import time

def check_and_remove(book, substring, error_if_not_found = True):
    count = book.count(substring)
    # Handle different cases
    if count == 0:
        if error_if_not_found:
            raise ValueError(f"Substring '{substring}' not found in document")
        else:
            return book
    elif count > 1:
        raise ValueError(f"Substring '{substring}' found {count} times, expected exactly once")
    else:  # count == 1
        # Remove the single occurrence
        return book.replace(substring, "", 1)
    
def patch_for_ensuring_token_size_lower_130k_in_llama3(book):
    # "maximum context length is 131000 tokens. However, you requested about 131878 tokens"
    # We are looking for some unused paragraphs to remove, for which no useful information is added:
    # max_idx_plus_2 = [2+max(a,b,c,d) for a,b,c,d in zip(df_book_groundtruth['idx_t'], df_book_groundtruth['idx_s'], 
    #                                                     df_book_groundtruth['idx_e'], df_book_groundtruth['idx_c'])]
    # df_book_groundtruth[df_book_groundtruth['nb_paragraphs'] > max_idx_plus_2]
    # > Chapter 13: 9 paragraphs, information in paragraphs {1,2,4,1}
    # > Chapter 139: 10 paragraphs, information in paragraphs {1,2,1,6}
    # Last 2 paragraphs of 139 are removed (also not containing additional entities), etc. for some other chapters
    # 131878 --> (remove 139,13,133,152,167 tokens) --> 131216
    substring139 = """

The lights began to flicker more violently, plunging the hall into moments of complete darkness. In those brief seconds, Julian heard whispers and movements that shouldn't have been possible. His heart pounded in his chest as he frantically tried to save his work, the laptop screen now displaying impossibly twisted images.

As the situation descended into chaos, Julian found himself trapped in a nightmare of his own creation. The UI he had designed seemed to come alive, reaching out from the screen with tendrils of corrupted code. He stumbled back, watching in horror as the digital realm bled into reality, transforming the museum into a grotesque fusion of technology and primal fear."""
    substring13 = """

As the festival drew to a close, Scarlett collapsed into a nearby chair, her clipboard abandoned on the floor beside her. She watched as the last stragglers stumbled towards the exit, their pockets bulging with bottle openers and branded coasters. Despite the chaos, a small smile played on her lips. After all, in the world of craft beer, there was always another festival just around the corner – and hopefully, this time, she'd remember to wear shoes with better traction on beer-soaked floors."""
    substring133 = """

Suddenly, the lights flickered and went out, plunging the room into darkness. Gasps and confused murmurs filled the air. Scarlett felt a brush of movement beside her, followed by the sound of hurried footsteps. When the lights came back on moments later, she immediately sensed that something was amiss.

Her eyes darted around the room, taking in the startled faces of the other attendees. It was then that she noticed an empty space on the wall where one of the most valuable photographs had been hanging just moments before. The theft had been swift and silent, executed with precision in those few seconds of darkness.

As security personnel rushed to secure the exits, Scarlett's mind raced. She recalled Jonathan Rea's suspicious behavior, Stevie Paterson's heated discussion, and the conveniently timed blackout. The pieces of the puzzle were there, but how did they fit together? She took a deep breath, steeling herself for the investigation that was sure to follow. This photography exhibition had just become far more intriguing than she could have ever anticipated."""
    substring152 = """

As the day wore on, the air grew thick with the scent of coffee and the electric tang of overclocked processors. Levi's creation began to take shape, a shimmering construct of light and data that hovered above his workstation like a sentient cloud. Other hackers paused in their work to marvel at the beauty of his code, its elegant simplicity masking layers of complexity beneath.

With mere minutes left before the final bell, Levi put the finishing touches on his masterpiece. The integrated services pulsed with life, each one a testament to his skill and vision. As he stepped back to admire his work, the entire hall erupted in applause. In that moment, surrounded by the timeless beauty of art and the cutting edge of technology, Levi knew that he had woven a spell that would change the world."""
    substring167 = """

With a wave of his hand, he conjured a shimmering aurora that danced across the dome, its ethereal light casting a soft glow over the awestruck faces below. In that instant, the boundary between science and magic blurred, leaving only wonder in its wake.

As the final constellations faded and the lights slowly came up, a chorus of applause filled the air. He bowed slightly, feeling the weight of countless dreams and aspirations settling on his shoulders. Tonight had been more than just an astronomy show; it had been a glimpse into the infinite, a reminder of humanity's place in the vast cosmic dance. And as the crowd began to disperse, he knew that the magic of this night would linger in their hearts long after the last star had faded from view."""
    substring64 = """

Panic rising in his chest, Henry tried to push his way through the press of bodies, desperate to find an exit. But every time he thought he'd found a way out, he found himself back in the center of the carnival, surrounded by leering faces and discordant music. The walls seemed to be closing in, the ceiling lowering with each passing moment.

As the night wore on, the carnival grew more frenzied, the performances more twisted and bizarre. Henry's mind began to fragment, unable to process the horrors unfolding before him. He clung to the hope that dawn would bring an end to this nightmare, but a small voice in the back of his mind whispered that the sun might never rise again in this unholy place."""
    substring25 = """ She moved to intervene, her role as organizer momentarily forgotten in the face of genuine conflict. But as she approached, a blood-curdling scream pierced the air, freezing everyone in their tracks. This wasn't part of the script.

In that moment of shared horror, Mila realized the tragic irony of her carefully orchestrated event. The very atmosphere of suspicion and intrigue she had cultivated had given rise to something far more sinister than she could have ever imagined. As chaos erupted around her, guests panicking and security rushing in, she stood rooted to the spot, the weight of unintended consequences crushing down upon her. The murder mystery dinner she had dreamed would be her crowning achievement now threatened to become her downfall, the line between entertainment and tragedy irrevocably blurred in the echoing halls of the museum."""
    book = check_and_remove(book, substring139) # 182 tokens
    book = check_and_remove(book, substring13) # 131 tokens
    book = check_and_remove(book, substring133) # 284 tokens
    book = check_and_remove(book, substring152) # 211 tokens
    book = check_and_remove(book, substring167) # 199 tokens
    book = check_and_remove(book, substring64)
    book = check_and_remove(book, substring25)
    return book

def whether_do_this_q(q, q_max):
    if q_max is None:
        return True
    else:
        return (q < q_max)
            
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
    if answering_parameters['model_name'] == 'llama-3.1-405b-instruct':
        if my_benchmark.nb_tokens() == 102870: # 102870 for our count, but 131878 for llama3
            # "maximum context length is 131000 tokens. However, you requested about 131878 tokens"
            # We are looking for some unused paragraphs to remove, for which no useful information is added
            book = patch_for_ensuring_token_size_lower_130k_in_llama3(book)

    df_qa = my_benchmark.get_df_qa()
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()

    # loop
    generated_answers = []
    for q in range(len(df_qa)):
        answer_filepath = answer_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
        answer_reasoning_filepath = answer_reasoning_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
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
            out, reasoning = my_model.generate(user_prompt = user_prompt, system_prompt = system_prompt, max_new_tokens = max_new_tokens, keep_reasoning = True)
            print(f"sleeping for {sleeping_time} seconds")
            time.sleep(sleeping_time)
            print("woke up")
            answer_filepath.parent.mkdir(parents=True, exist_ok=True)
            print(answer_filepath)
            export_list(out, answer_filepath)
            if reasoning is not None:
                answer_reasoning_filepath.parent.mkdir(parents=True, exist_ok=True)
                print(answer_reasoning_filepath)
                export_list(reasoning, answer_reasoning_filepath)
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

    if answering_parameters['model_name'] == 'llama-3.1-405b-instruct':
        if my_benchmark.nb_tokens() == 102870: # 102870 for our count, but 131878 for llama3
            # "maximum context length is 131000 tokens. However, you requested about 131878 tokens"
            # We are looking for some unused paragraphs to remove, for which no useful information is added
            # We also put this information at the evaluation stage (needed for evaluating the full chapters)
            book = my_benchmark.book
            book = patch_for_ensuring_token_size_lower_130k_in_llama3(book)
            split_chapters = split_chapters_func(book)

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



