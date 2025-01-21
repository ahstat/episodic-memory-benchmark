from epbench.src.io.io import paragraph_filepath_func, paragraph_verif_direct_filepath_func, paragraph_verif_llm_filepath_func, import_list
from epbench.src.generation.generate_1_events_and_meta_events import generate_and_export_events_and_meta_events_func
import re
from math import floor

def get_auto_iteration(event_index, prompt_parameters, model_parameters, data_folder):
    i = 0
    while True:
        file_path = paragraph_filepath_func(i, event_index, data_folder, prompt_parameters, model_parameters)
        if not file_path.exists():
            if i == 0:
                return None # No files found
            return i-1
        i += 1

def get_single_ei(event_index, iteration, prompt_parameters, model_parameters, data_folder, rechecking = True):
    if iteration == "last":
        iteration = get_auto_iteration(event_index, prompt_parameters, model_parameters, data_folder)
        if iteration is None:
            print(f"No iteration found for this event index {event_index}")
            return None
    paragraph_filepath = paragraph_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
    verif_direct_filepath = paragraph_verif_direct_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
    verif_llm_filepath = paragraph_verif_llm_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
    if not paragraph_filepath.is_file():
        print("Paragraph has not been generated")
        return None
    if not verif_direct_filepath.is_file():
        print("Paragraph generated but not verified")    

    events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
    event = events[event_index]
    meta_event = meta_events[event_index]

    generated_paragraph = import_list(paragraph_filepath)

    verif_direct = import_list(verif_direct_filepath)
    has_passed_verif_direct = verif_direct[0]
    which_issues_verif_direct = verif_direct[1]

    if verif_llm_filepath.is_file():
        verif_llm = import_list(verif_llm_filepath)
        has_passed_verif_llm = verif_llm[0]
        which_issues_verif_llm = verif_llm[1]
        verbose_issues_verif_llm = verif_llm[3]
    else:
        verif_llm = None
        has_passed_verif_llm = True
        which_issues_verif_llm = None
        verbose_issues_verif_llm = None

    return event, meta_event, generated_paragraph, has_passed_verif_direct, has_passed_verif_llm, which_issues_verif_direct, which_issues_verif_llm, verbose_issues_verif_llm, iteration

def wrap_text(text, width=120):
    wrapped_lines = []
    for line in text.split('\n'):
        while len(line) > width:
            wrap_index = line.rfind(' ', 0, width)
            if wrap_index == -1:
                wrap_index = width
            wrapped_lines.append(line[:wrap_index])
            line = line[wrap_index:].lstrip()
        wrapped_lines.append(line)
    return '\n'.join(wrapped_lines)

def col_func(text, color = "green"):
    if color == "green":
        return f"\x1b[32m{text}\x1b[0m"
    if color == "red":
        return f"\x1b[31m{text}\x1b[0m"
    if color == "blue":
        return f"\x1b[34m{text}\x1b[0m"
    if color == "gray":
        return f"\x1b[90m{text}\x1b[0m"

def highlight_func(text, target, color = "green"):
    color_codes = {
        'green': ('\033[30;42m', '\033[0m'),  # Black text on green background
        'blue': ('\033[37;44m', '\033[0m'),   # White text on blue background
        'red': ('\033[37;41m', '\033[0m'),    # White text on red background
        'gray': ('\033[30;47m', '\033[0m'),    # Black text on gray background
        'yellow': ('\033[30;43m', '\033[0m')  # Black text on yellow background
    }
    
    if color not in color_codes:
        raise ValueError(f"Color '{color}' not supported.")
    
    start, end = color_codes[color]

    # Use re.sub with a case-insensitive flag
    def replace_func(match):
        return f"{start}{match.group()}{end}"
    
    return re.sub(re.escape(target), replace_func, text, flags=re.IGNORECASE)
    
def pretty_print_before_post_processing(event_index, iteration, prompt_parameters, model_parameters, data_folder, width=150):
    output = get_single_ei(event_index, iteration, prompt_parameters, model_parameters, data_folder)
    if output is not None:
        event, meta_event, generated_paragraph, has_passed_verif_direct, has_passed_verif_llm, which_issues_verif_direct, which_issues_verif_llm, verbose_issues_verif_llm, iteration2 = output
        iteration = iteration2
    else:
        return output

    has_passed = has_passed_verif_direct and has_passed_verif_llm
    if(has_passed):
        print(col_func(f"*Correct* sample (event={event_index}, iter={iteration})", "green"))
    else:
        print(col_func(f"*Incorrect* sample (event={event_index}, iter={iteration})", "red"))
        if not has_passed_verif_direct:
            print(col_func(f"Issue in *direct* verification: {which_issues_verif_direct}", "gray"))
        if not has_passed_verif_llm:
            print(col_func(f"Issue in *llm* verification: {which_issues_verif_llm}, as the answer is: {verbose_issues_verif_llm}", "gray"))
    print("")
    print(col_func(event, "blue"))
    print(col_func(meta_event, "blue"))
    print("")

    print(format_generated_paragraph(generated_paragraph, event, width))

def pretty_print(event_index, d, width=150):
    generated_paragraph = d[event_index]['paragraphs']
    is_valid = d[event_index]['is_valid']
    nb_tokens = d[event_index]['nb_tokens']
    event_index = d[event_index]['event_idx']
    iteration = d[event_index]['iter_idx']
    event = d[event_index]['event']
    meta_event = d[event_index]['meta_event']
    post_entities = d[event_index]['post_entities']
    if is_valid:
        print(col_func(f"*Correct* sample (event={event_index}, iter={iteration})", "green"))
        print("")
        print(col_func(event, "blue"))
        print(col_func(meta_event, "blue"))
        print(col_func(post_entities, "blue"))
        print(col_func(f"Generated chapter has {nb_tokens} tokens", "gray"))
        print("")
        print(format_generated_paragraph(generated_paragraph, event, width))
    else:
        print(col_func(f"*Incorrect* sample (event={event_index}, iter={iteration})", "red"))
        print("Please use the method `pretty_print_debug_event_iter_idx` for more details")

def format_generated_paragraph(generated_paragraph, event, width=150):
    str = generated_paragraph
    str = highlight_func(str, event[0], color = "green")
    str = highlight_func(str, event[1], color = "blue")
    str = highlight_func(str, event[2], color = "red")
    str = highlight_func(str, event[4], color = "gray")
    str = wrap_text(str, width) # to do after the coloring
    return str

def split_chapters_func(book):
    # ad hoc function to get back the chapters
    pattern = r'Chapter (\d+)\n\n(.*?)(?=Chapter \d+\n\n|$)'
    matches = re.findall(pattern, book, re.DOTALL)
    
    result = {}
    for chapter, content in matches:
        chapter_num = int(chapter)
        if chapter_num in result:
            raise ValueError(f"Duplicate chapter number found: Chapter {chapter_num}")
        result[chapter_num] = content.strip()
    
    return result

def find_absolute_position(pattern, text):
    match = re.search(pattern, text)

    if match:
        matched_text = match.group()
        start_index = match.start()
        end_index = match.end()
    else:
        raise ValueError('cannot find the element')

    return floor((start_index+end_index)/2) # average floored position within the text

def find_relative_position(pattern, text):
    return find_absolute_position(pattern, text)/len(text)
