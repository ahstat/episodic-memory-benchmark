import re
from epbench.src.io.io import paragraph_verif_direct_filepath_func, import_list, export_list

def cut_paragraph_func(paragraphs):
    paragraphs_list = re.split(r'\n+', paragraphs.strip())
    # Remove any empty strings from the list
    return [p for p in paragraphs_list if p]

def remove_initial_numbering(text):
    return re.sub(r'^\(\d+\)\s*', '', text)

def extract_initial_numbering(text):
    match = re.match(r'^\(\d+\)\s*', text)
    if not match:
        # there is no expected the initial numbering
        return ''
    return match.group()

def create_numbered_list(n):
    return [f'({i}) ' for i in range(1, n + 1)]

def string_contains(substring, main_string):
    return substring.lower() in main_string.lower()

def find_match_paragraph_indexes(element, cut_paragraph_wo_numbers):
    bool_list = [string_contains(element, main_string) for main_string in cut_paragraph_wo_numbers]
    return [i+1 for i, value in enumerate(bool_list) if value]

def verification_unique_presence(idx_result, idx_groundtruth):
    """
    idx_result: a list, e.g. [1], [], [1,2]
    idx_groundtruth: a numeric, e.g. 2
    """
    has_X = -1
    single_X = -1
    correct_X = -1

    # check existence
    if len(idx_result) > 0:
        has_X = 1
    else:
        has_X = 0
        return has_X, single_X, correct_X
    
    # check unicity
    if len(idx_result) == 1:
        single_X = 1
    else:
        single_X = 0
        return has_X, single_X, correct_X
    
    # check correctness
    if idx_result[0] == idx_groundtruth:
        correct_X = 1
    else:
        correct_X = 0

    return has_X, single_X, correct_X

def verification_entity_shape_func(text, verbose):
    '''
    Check that the entity description is always on the form ' $entity_' (if any)
    Put False for instance in the case where the entity is written as: 'Entity_1', 'Entity1', etc.,
    which is incorrect (sometimes gpt4o outputs like that)
    '''

    # Convert text to lowercase for case-insensitive matching
    lower_text = text.lower()
    
    # Find all occurrences of 'entity' in the lowercase text
    matches = re.finditer(r'entity', lower_text)
    
    # Extract the original shape with surrounding characters
    results = []
    for match in matches:
        start = max(0, match.start() - 1)  # 1 character before, but not before start of string
        end = min(len(text), match.end() + 1)  # 1 character after, but not past end of string
        results.append(text[start:end])

    has_correct_entity_occurrences = all(item == '$entity_' for item in results)

    if verbose:
        if not has_correct_entity_occurrences:
            print(f' - Expected always "$entity_", but observed {','.join(results)}')

    return has_correct_entity_occurrences

def verification_entity_shape_integer_func(text, verbose):
    '''
    following the other entity shape check, here specifically check that each time it is followed by an integer
    '''
    # Find all occurrences of $entity_ followed by digits
    pattern = r'\$entity_(\d+)'
    matches = re.findall(pattern, text)
    
    # Find all occurrences of $entity_ (even if not followed by digits)
    all_entities = re.findall(r'\$entity_\w*', text)
    
    # Check if all $entity_ are followed by integers
    all_valid = len(matches) == len(all_entities)

    if verbose:
        if not all_valid:
            print(f' - Expected to see only {','.join(matches)}, but saw all {','.join(all_entities)}')
    
    return all_valid

def verification_paragraphs_func(paragraphs, event, meta_event, verbose = True):
    # -1: not tested
    #  0: fail the test
    #  1: pass the test
    failing = 0
    passing = 1
    verification_dict = {'paragraphs_nb': -1, 
                         'paragraphs_initials_shape': -1, 
                         'paragraphs_initials_increment': -1,
                         'other_entities_shape': -1,
                         'other_entities_shape_integer': -1,
                         'has_location': -1,
                         'has_date': -1,
                         'has_entity': -1,
                         'has_content': -1,
                         'single_location': -1,
                         'single_date': -1,
                         'single_entity': -1,
                         'single_content': -1,
                         'correct_location': -1,
                         'correct_date': -1,
                         'correct_entity': -1,
                         'correct_content': -1
                         }

    # check number of paragraphs
    cut_paragraphs = cut_paragraph_func(paragraphs)
    if meta_event['nb_paragraphs'] != len(cut_paragraphs):
        verification_dict['paragraphs_nb'] = failing
        if verbose:
            print(f' - Expected {meta_event['nb_paragraphs']} paragraphs, but observed {str(len(cut_paragraphs))}')
        return verification_dict
    else:
        verification_dict['paragraphs_nb'] = passing

    # check presence of "(X) " with X a number a the beginning of each paragraph
    candidate_numbering = [extract_initial_numbering(p) for p in cut_paragraphs]
    if any(x == '' for x in candidate_numbering):
        verification_dict['paragraphs_initials_shape'] = failing
        if verbose:
            print(f' - Expected always (X) at the begin of paragraphs, but observed {','.join(candidate_numbering)}')
        return verification_dict
    else:
        verification_dict['paragraphs_initials_shape'] = passing

    # check that this (X) numbering is incrementing: (1), (2), etc. 
    expected_numbering = create_numbered_list(len(cut_paragraphs))
    if not all([c==e for (c,e) in zip(candidate_numbering, expected_numbering)]):
        verification_dict['paragraphs_initials_increment'] = failing
        if verbose:
            print(f' - Expected {','.join(expected_numbering)}, but observed {','.join(candidate_numbering)}')
        return verification_dict
    else:
        verification_dict['paragraphs_initials_increment'] = passing
    cut_paragraph_wo_numbers = [remove_initial_numbering(p) for p in cut_paragraphs]

    # check that the other entities are always in lower case as $entity_
    if not verification_entity_shape_func(paragraphs, verbose):
        verification_dict['other_entities_shape'] = failing
        return verification_dict
    else:
        verification_dict['other_entities_shape'] = passing

    # check that the other entities $entity_ are always followed by an integer
    if not verification_entity_shape_integer_func(paragraphs, verbose):
        verification_dict['other_entities_shape_integer'] = failing
        return verification_dict
    else:
        verification_dict['other_entities_shape_integer'] = passing

    # check the presence verbatim of the date, the location, the entity, and the content detail
    # in the specified paragraph, while checking the absence in the other paragraphs
    date = event[0]
    location = event[1]
    entity = event[2]
    detail = event[4]

    idx_location_result = find_match_paragraph_indexes(location, cut_paragraph_wo_numbers)
    idx_date_result = find_match_paragraph_indexes(date, cut_paragraph_wo_numbers)
    idx_entity_result = find_match_paragraph_indexes(entity, cut_paragraph_wo_numbers)
    idx_content_result = find_match_paragraph_indexes(detail, cut_paragraph_wo_numbers)

    idx_location_groundtruth = meta_event['idx_paragraph']['location']
    idx_date_groundtruth = meta_event['idx_paragraph']['date']
    idx_entity_groundtruth = meta_event['idx_paragraph']['entity']
    idx_content_groundtruth = meta_event['idx_paragraph']['content']

    has_location, single_location, correct_location = verification_unique_presence(idx_location_result, idx_location_groundtruth)
    has_date, single_date, correct_date = verification_unique_presence(idx_date_result, idx_date_groundtruth)
    has_entity, single_entity, correct_entity = verification_unique_presence(idx_entity_result, idx_entity_groundtruth)
    has_content, single_content, correct_content = verification_unique_presence(idx_content_result, idx_content_groundtruth)

    verification_dict['has_location'] = has_location
    verification_dict['single_location'] = single_location
    verification_dict['correct_location'] = correct_location
    verification_dict['has_date'] = has_date
    verification_dict['single_date'] = single_date
    verification_dict['correct_date'] = correct_date
    verification_dict['has_entity'] = has_entity
    verification_dict['single_entity'] = single_entity
    verification_dict['correct_entity'] = correct_entity
    verification_dict['has_content'] = has_content
    verification_dict['single_content'] = single_content
    verification_dict['correct_content'] = correct_content

    return verification_dict

def has_passed_direct_verification_func(paragraphs, event, meta_event, verbose = False):
    # print(verification_paragraphs_func(paragraphs, event, meta_event))
    verification_paragraphs = verification_paragraphs_func(paragraphs, event, meta_event, verbose)
    count_one = sum(1 for value in verification_paragraphs.values() if value == 1)
    total_count = len(verification_paragraphs)
    has_passed_direct_verification = False
    issues = []
    if count_one == total_count:
        has_passed_direct_verification = True
    else:
        issues = [key for key, value in verification_paragraphs.items() if value == 0]
    return has_passed_direct_verification, issues

def export_direct_verif_vector(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder):
    result = [has_passed_direct_verification_func(p, e, m) for (p, e, m) in zip(generated_paragraphs, events, meta_events)]
    for iteration, event_index in zip(iterations, range(len(iterations))):
        verifdirect_paragraphs_filepath = paragraph_verif_direct_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
        if not verifdirect_paragraphs_filepath.is_file():
            verifdirect_paragraphs_filepath.parent.mkdir(parents=True, exist_ok=True)
            export_list(result[event_index], verifdirect_paragraphs_filepath)
        else:
            result_previous = import_list(verifdirect_paragraphs_filepath)
            if not list(result[event_index]) == result_previous:
                print(result[event_index])
                print(result_previous)
                raise ValueError('direct verification gave a different result before, for the same paragraph')
    return 0
