import pandas as pd
from itertools import chain, product
import random
import numpy as np
from epbench.src.generation.qa_templates import EpisodicMemoryTemplates
from epbench.src.generation.generate_1_events_and_meta_events import unused_universe_func

def get_tsec_given_question(chapter_idx, df_groundtruth, unused_universe, changed = set(), existing_change = True):
    # if changed=set(), it is simply the (t,s,e,c) for the current question (do not use {}, which is a dict and not a set)
    # changed can be e.g. changed = {'date', 'location'}, only used for generating questions with empty answers
    df_gt_cur = df_groundtruth.loc[chapter_idx]

    t=df_gt_cur['date']
    s=df_gt_cur['location']
    e=df_gt_cur['entity']
    c=df_gt_cur['content']

    if (len(changed) > 0) and (not existing_change): # when it is an existing change, the unused universe is not used
        (unused_t, unused_s, unused_e, unused_c, unused_d) = unused_universe

    random.seed(chapter_idx) # define a seed for reproducibility

    if 'date' in changed:
        if existing_change:
            t_other = sorted(list(set(df_groundtruth['date'].unique())))
            t_other.remove(t)
            t_other = random.sample(t_other, 1)[0]
            t = t_other
        else:
            t_unused = random.sample(unused_t, 1)[0]
            t = t_unused

    if 'location' in changed:
        if existing_change:
            s_other = sorted(list(set(df_groundtruth['location'].unique())))
            s_other.remove(s)
            s_other = random.sample(s_other, 1)[0]
            s = s_other
        else:
            s_unused = random.sample(unused_s, 1)[0]
            s = s_unused

    if 'entity' in changed:
        if existing_change:
            e_other = sorted(list(set(df_groundtruth['entity'].unique())))
            e_other.remove(e)
            e_other = random.sample(e_other, 1)[0]
            e = e_other
        else:
            e_unused = random.sample(unused_e, 1)[0]
            e = e_unused

    if 'content' in changed:
        if existing_change:
            c_other = sorted(list(set(df_groundtruth['content'].unique())))
            c_other.remove(c)
            c_other = random.sample(c_other, 1)[0]
            c = c_other
        else:
            c_unused = random.sample(unused_c, 1)[0]
            c = c_unused

    return (t,s,e,c)

def all_cue_str():
    '''
    List all the 16 cue str combinations '(*, *, *, *)', '(*, *, *, c)', ... '(t, s, ent, *)', '(t, s, ent, c)'
    '''
    time_str = "t"
    location_str = "s"
    entity_str = "ent"
    content_str = "c"
    l=list(product(['*', time_str], ['*', location_str], ['*', entity_str], ['*', content_str]))
    return [f"({t}, {s}, {e}, {c})" for t,s,e,c in l]

def identify_empty_cues(t,s,e,c, df_groundtruth):
    '''
    Identify the cues for which the answer is empty, given a (t,s,e,c)
    Note: if the tsec is the event of an existing chapter, the answer will be [] since the answer is non empty for any cue
    '''
    return [cue_str for cue_str in all_cue_str() if len(cue_subsetting_func(cue_str, t, s, e, c, df_groundtruth)) == 0]

def return_n_count(df_groundtruth, elem, type = 'location'):
    # elem is any of the t,s,e,c, e.g., s
    if type == 'date':
        found = df_groundtruth[df_groundtruth['date'] == elem]['n_date']
    elif type == 'location':
        found = df_groundtruth[df_groundtruth['location'] == elem]['n_location']
    elif type == 'entity':
        found = df_groundtruth[df_groundtruth['entity'] == elem]['n_entity']
    elif type == 'content':
        found = df_groundtruth[df_groundtruth['content'] == elem]['n_content']
    else:
        raise ValueError('unknown type')
    if len(found) > 0:
        return found.iloc[0]
    else:
        return int(0)
    
def retrieve_dataframe_for_single_question(chapter_idx, df_groundtruth, unused_universe, changed = set(), existing_change = True):
    # chapter_idx  = chapter question index

    (t,s,e,c) = get_tsec_given_question(chapter_idx, df_groundtruth, unused_universe, changed, existing_change)

    # Current selected chapter
    df_gt_cur = df_groundtruth.loc[chapter_idx].copy()
    if len(changed) > 0:
        # correct the row
        df_gt_cur['chapter'] = -1
        if t != df_gt_cur['date']:
            df_gt_cur['date'] = t
            df_gt_cur['n_date'] = return_n_count(df_groundtruth, t, 'date')
        if s != df_gt_cur['location']:
            df_gt_cur['location'] = s
            df_gt_cur['n_location'] = return_n_count(df_groundtruth, s, 'location')
        if e != df_gt_cur['entity']:
            df_gt_cur['entity'] = e
            df_gt_cur['n_entity'] = return_n_count(df_groundtruth, e, 'entity')
        if c != df_gt_cur['content']:
            df_gt_cur['content'] = c
            df_gt_cur['n_content'] = return_n_count(df_groundtruth, c, 'content')

    # Generate all questions for the selected chapter
    em_templates = EpisodicMemoryTemplates()
    questions = em_templates.generate_all_questions(t=t, s=s, ent=e, c=c)
    nb_questions = len(questions)
    cues = [x['cue'] for x in questions]
    cue_completed = [x['cue_completed'] for x in questions]
    retrieval_types = [x['retrieval_type'] for x in questions]
    gets = [x['get'] for x in questions]
    questions = [x['question'] for x in questions] # /!\ erase `questions` variable
    q_idx = [x for x in range(nb_questions)]

    df_cur = pd.DataFrame({'chapter': df_gt_cur['chapter'],
                    'date': df_gt_cur['date'],
                    'location': df_gt_cur['location'],
                    'entity': df_gt_cur['entity'],
                    'content': df_gt_cur['content'],
                    'n_date': df_gt_cur['n_date'],
                    'n_location': df_gt_cur['n_location'],
                    'n_entity': df_gt_cur['n_entity'],
                    'n_content': df_gt_cur['n_content'],
                    'debug_changed': [changed]*nb_questions,
                    'debug_existing_change': [existing_change]*nb_questions,
                    'q_idx': q_idx,
                    'cue': cues,
                    'cue_completed': cue_completed,
                    'retrieval_type': retrieval_types,
                    'get': gets,
                    'question': questions
                    }, index = q_idx)
    
    # if changed, select only the questions with 0 answer
    if len(changed) > 0:
        identified_cues = identify_empty_cues(t,s,e,c, df_groundtruth) # empty_cues
        df_cur = df_cur[df_cur['cue'].isin(identified_cues)]

    return df_cur

## Get the correct answer
def get_tuple_from_cue(cue_str):
    items = cue_str.strip('()').split(', ')    
    items = tuple([item.strip() for item in items])
    return tuple(items)

def cue_subsetting_func(cue_str, date, location, entity, content, df_groundtruth):
    cue = get_tuple_from_cue(cue_str)
    df_sliced = df_groundtruth.copy()
    # (t, s, ent, c)
    if cue[0] == "t":
        df_sliced = df_sliced[df_sliced['date'] == date]
    if cue[1] == "s":
        df_sliced = df_sliced[df_sliced['location'] == location]
    if cue[2] == "ent":
        df_sliced = df_sliced[df_sliced['entity'] == entity]
    if cue[3] == "c":
        df_sliced = df_sliced[df_sliced['content'] == content]
    return df_sliced

def cue_subsetting_from_chapter_idx(cue_str, chapter_idx, df_groundtruth):
    # cue_str = "(t, s, ent, *)"
    q = chapter_idx # chapter question index, e.g. 175
    df_gt_cur = df_groundtruth.loc[q]
    date=df_gt_cur['date']
    location=df_gt_cur['location']
    entity=df_gt_cur['entity']
    content=df_gt_cur['content']
    df_sliced = cue_subsetting_func(cue_str, date, location, entity, content, df_groundtruth)
    return df_sliced

def retrieval_type_subsetting_from_sliced(retrieval_type, df_sliced):
    if retrieval_type == "Times":
        res_list = df_sliced['date']
    elif retrieval_type == "Spaces":
        res_list = df_sliced['location']
    elif retrieval_type == "Entities":
        res_list = df_sliced['entity']
    elif retrieval_type == "Event contents":
        res_list = df_sliced['content']
    elif retrieval_type == "Other entities":
        # more complex since it is a single row of the data frame containing a list of entities
        extract_array = df_sliced[['post_entities']].values # array([[list(['Poet Angell', 'Tessa Nicholson'])]], dtype=object)
        post_entity_list = list(chain(*chain(*extract_array))) # ['Poet Angell', 'Tessa Nicholson']
        res_list = post_entity_list
    elif retrieval_type == "Full event details":
        res_list = [f"Full chapter {list(df_sliced['chapter'])[0]}"]
    else:
        raise ValueError('unknown retrieval_type')

    if retrieval_type != "Full event details" and retrieval_type != "Other entities":
        res = set(res_list.unique())
        corresponding_chapters = {chapter: r for (r, chapter) in zip(res_list, df_sliced['chapter'])}
    else:
        res = set(res_list)
        if len(df_sliced) > 1: # not possible, since all (t,s,e,c) is known
            raise ValueError('for the questions involving a single event, there should be only one chapter remaining')
        corresponding_chapters = {chapter: res_list for chapter in df_sliced['chapter']}

    # also retrieve only the chapter values
    chapter_list = set([x for x in corresponding_chapters.keys()])

    return res, corresponding_chapters, chapter_list # remember the corresponding_chapters for debugging, and the chapter set

def extract_chronological_order_func(df_groundtruth, chapter_list = {1, 8, 12, 55, 87}):
    df_order = df_groundtruth[['chapter', 'date']].copy()
    df_order['date_ts'] = pd.to_datetime(df_order['date'], format='%B %d, %Y')
    df_order_subset = df_order.loc[list(chapter_list)].sort_values('date_ts')
    chronological_chapters = df_order_subset.index.tolist() # loc is working since the `chapter` is the index already
    has_duplicates = bool(df_order_subset['date'].duplicated().any()) # True when 2 chapters with the same date, and in this case chronological_chapters is ambiguous 
    return chronological_chapters, has_duplicates

def get_groundtruth_answer(df_groundtruth, cue_str="(*, *, *, c)", retrieval_type="Times", get = "all", chapter_idx=175):
    if chapter_idx == -1:
        # valid for all values of `get` ("all", "latest", "chronological")
        res=set()
        corresponding_chapters=dict()
        chapter_list = set()
        return res, corresponding_chapters, chapter_list, 0
    df_sliced = cue_subsetting_from_chapter_idx(cue_str, chapter_idx, df_groundtruth)
    res, corresponding_chapters, chapter_list = retrieval_type_subsetting_from_sliced(retrieval_type, df_sliced)

    n_chapters_correct_answer = len(corresponding_chapters) # always this, even for latest

    if get == "latest" or get == "chronological":
        # need to identify the correct order
        chronological_chapters, has_duplicates = extract_chronological_order_func(df_groundtruth, chapter_list)
        if has_duplicates:
            raise ValueError('Chronological chapters is ambiguous since two selected chapters have the same date')
        chapter_list = chronological_chapters # change from a set to a list
        # `corresponding_chapters` is kept as a dictionary
        res = [corresponding_chapters[i] for i in chronological_chapters] # change from a set to a list

        if get == "latest": # further only take the last element
            chapter_list = [chapter_list[-1]] # keep a list, as before
            corresponding_chapters = {k: v for (k,v) in corresponding_chapters.items() if k in chapter_list}
            res = [res[-1]] # keep a list, as before

    return res, corresponding_chapters, chapter_list, n_chapters_correct_answer

def retrieve_dataframe_for_all_questions(idx_chapter_questions, df_groundtruth, events, prompt_parameters=None, changed_list=None, existing_change_list=False):
    if changed_list is None: # no change in building the questions
        changed_list = [set() for x in idx_chapter_questions]
        existing_change_list = [None for x in idx_chapter_questions] # no impact
        unused_universe = None
    elif all(existing_change_list): # only from the existing changed list, so the unused universe is not needed
        unused_universe = None
    else:
        unused_universe = unused_universe_func(prompt_parameters, events)

    res = [retrieve_dataframe_for_single_question(chapter_idx, df_groundtruth, unused_universe, changed, existing_change) for (chapter_idx, changed, existing_change) in zip(idx_chapter_questions, changed_list, existing_change_list)]
    df_qa = pd.concat(res, ignore_index=True)

    # add the ground truth answer
    new_columns = [get_groundtruth_answer(df_groundtruth, cue, retrieval_type, get, chapter) for (cue, retrieval_type, get, chapter) in zip(df_qa['cue'], df_qa['retrieval_type'], df_qa['get'], df_qa['chapter'])]
    df_qa['correct_answer'] = [x[0] for x in new_columns] # groundtruth_answer_set, answer as a set
    df_qa['correct_answer_chapters'] = [x[2] for x in new_columns] # groundtruth_answer_chapters, chapter listing only
    df_qa['correct_answer_detailed'] = [x[1] for x in new_columns] # groundtruth_answer_dict, indicating chapter before each element, as dict

    # adding n_items and n_chapters
    df_qa['n_items_correct_answer'] = df_qa['correct_answer'].apply(len) # df_qa['item_count'] = df_qa['correct_answer'].apply(len)
    # df_qa['n_chapters_correct_answer'] = df_qa['correct_answer_detailed'].apply(len) # df_qa['event_count'] = df_qa['correct_answer_detailed'].apply(len)
    # replace with the following, that cue to many chapters even for "latest"
    df_qa['n_chapters_correct_answer'] = [x[3] for x in new_columns]
    # It is possible to get n_chapters_correct_answer == n_items_correct_answer
    # For most questions, we have n_items == n_chapters, since the number of items to be retrieved in a single chapter is generally exactly 1
    #
    # It is possible to get n_chapters_correct_answer > n_items_correct_answer
    # e.g., an entity (cue) is in the same location s (question) in two different chapters (i and j), so that
    # the item is {s} but the detailed answer is {i: s, j: s}, and n_items is 1 while n_chapters is 2.
    # Example: In some case, 'One World Trade Center' appears two times, so nb_chapters > nb_items
    #
    # It is possible to get n_chapters_correct_answer < n_items_correct_answer 
    # e.g., when using the post entities as a proxy of the full event, there is usually > 1 post entities for each involved chapter
    # Example: In some case, there are many post entities for the single chapter 70, so nb_items > nb_chapters
    #
    # Additionally, n_chapters_correct_answer == n_events_correct_answer all the time since we built the data such that one chapter = one major event

    # There may be duplicated questions (every time n_chapters_correct_answer > 1), so we group to discard thoses
    # Remove elements related to the chapters, not to the questions:
    df_qa = df_qa.drop(columns=['date', 'location', 'entity', 'content', 'n_date', 'n_location', 'n_entity', 'n_content'])
    
    # group duplicated questions
    df_qa = df_qa.groupby('question').agg({
        'chapter': lambda x: sorted(list(x)), # this is dependent on the index of questions made in `idx_chapter_questions`
        # all the following give the same answer since it is the exact same question
        'q_idx': 'first',
        'cue': 'first',
        'cue_completed': 'first',
        'retrieval_type': 'first',
        'get': 'first',
        'correct_answer': 'first',
        'correct_answer_chapters': 'first',
        'correct_answer_detailed': 'first',
        'n_items_correct_answer': 'first',
        'n_chapters_correct_answer': 'first',
        # take any of the change set. Different change can give different set and answer but we can any with size 0
        'debug_changed': 'first',
        'debug_existing_change': 'first',
    }).rename(columns={'chapter': 'debug_chapter'}).reset_index()

    return df_qa

def sample_chapters(nb_selected, nb_chapters, seed):
    random.seed(seed)
    if nb_selected > nb_chapters:
        nb_selected = nb_chapters
    return random.sample(range(1, nb_chapters+1), nb_selected) # chapters begin from 1

def build_nonempty_qa_func(df_book_groundtruth, events, nb_chapters_max = 200, seed = 0):
    # nb_chapters_max = float('inf')
    random.seed(seed)
    nb_chapters = len(df_book_groundtruth) # all chapters
    idx_chapter_questions = sample_chapters(nb_chapters_max, nb_chapters, seed) # selected chapters on which we build all the questions
    df_qa = retrieve_dataframe_for_all_questions(idx_chapter_questions, df_book_groundtruth, events) # get all the questions for the selected, and remove duplicates
    # check that, when all the possible questions are computed, we retrieve the same correct answers
    if nb_chapters_max >= nb_chapters:
        assert all([len(x) == len(y) for x, y, z in zip(df_qa['debug_chapter'], df_qa['correct_answer_chapters'], df_qa['get']) if z == "all"])
        assert all([len(x) == y for x, y, z in zip(df_qa['debug_chapter'], df_qa['n_chapters_correct_answer'], df_qa['get']) if z == "all"])
    return df_qa

def sample_one_of_more_changed_elements(N, seed):
    random.seed(seed)
    elements = ['date', 'location', 'entity', 'content']
    results = []
    
    for _ in range(N):
        sample_size = random.randint(1, 4)
        sample = random.sample(elements, sample_size)
        results.append(set(sample))
    
    return results

def build_empty_qa_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max = 200, seed = 1):
    # nb_chapters_max = float('inf')
    random.seed(seed)
    nb_chapters = len(df_book_groundtruth) # all chapters
    idx_chapter_questions = sample_chapters(nb_chapters_max, nb_chapters, seed) # selected chapters on which we build all the questions
    changed_list = sample_one_of_more_changed_elements(len(idx_chapter_questions), seed) # [{'date', 'location'} for x in idx_chapter_questions]
    random.seed(seed)
    if nb_chapters_max < 1000:
        existing_change_list = random.choices([True, False], k=len(idx_chapter_questions))
    else: # not enough elements in the unused universe, only take from the existing one
        existing_change_list = random.choices([True], k=len(idx_chapter_questions))

    # verify that each element of changed_list is of length at least one (otherwise it is not an empty qa that is produced)
    assert all([len(x) >= 1 for x in changed_list]), "there is changed element that is empty, i.e. producing valid answer, whereas it should be the empty qa"

    df_qa = retrieve_dataframe_for_all_questions(idx_chapter_questions, df_book_groundtruth, events, prompt_parameters, changed_list, existing_change_list) # get all the questions for the selected, and remove duplicates

    return df_qa

def build_qa_all_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max = 200, seeds = (1,2)):
    seed1, seed2 = seeds
    df1 = build_nonempty_qa_func(df_book_groundtruth, events, nb_chapters_max, seed1)
    df2 = build_empty_qa_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max, seed2)
    df_qa_all = pd.concat([df1,df2], axis=0).reset_index(drop = True)
    df_qa_all = df_qa_all.drop(columns=['debug_chapter'])
    return df_qa_all

def filtering_built_qa_func(df_qa, target_number_of_questions_per_bin_per_q_idx = 5, verbose = False, bins_count = [0, 1, 2, 3, 6, np.inf], labels_count = ['0', '1', '2', '{3,4,5}', '6+'], seed = 3):
    # futher filter the question list randomly with the minimum losing in the 2 dimensions q_idx and bins
    # The bins are driven by bins_count and labels_count, currently five bins: {0} {1} {2} {3,4,5} {6,...} w.r.t. number of items
    # The q_idx can either have any number of items (e.g. for (t,*,*,*)) but can be limited to {0} or {1} (e.g. for (t,*,ent,*))
    if len(labels_count) != len(bins_count)-1:
        raise ValueError('the labels should correspond to all the [bin_i, bin_{i+1}) of the bins_count, hence of length nb bins_count - 1')
    
    debug_df_qa_value_counts = df_qa[['n_chapters_correct_answer']].value_counts() # only based w.r.t. chapters, not items
    if verbose:
        # it's normal to get a lot of 0 and 1, also since some questions only have 0 or 1 answer
        # then it decreases quickly, but there is a reasonable amount of >1 answers
        print("df_qa value count per nb of items (without looking at the q_idx):")
        print(debug_df_qa_value_counts)

    # binning
    df_qa['bins_items_correct_answer'] = pd.cut(df_qa['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)

    # filtering at random (`target_number_of_questions_per_bin_per_q_idx` if possible, or the maximum number per (q_idx, bin) group otherwise)
    df_qa_filtered = (df_qa
                    .sort_values(by='n_chapters_correct_answer', ascending=False) # favorise larger sets
                    .sort_values(by='cue_completed')
                    #.groupby(['cue', 'bins_items_correct_answer'], observed=True, sort=False)
                    .groupby(['q_idx', 'bins_items_correct_answer'], observed=True, sort=False)
                    #.apply(lambda x: x.sample(min(len(x), target_number_of_questions_per_bin_per_q_idx), replace=False, random_state=seed), include_groups=False)
                    .apply(lambda x: x.head(min(len(x), target_number_of_questions_per_bin_per_q_idx)), include_groups=False)
                    .reset_index())

    # widspreadness of filtered questions
    debug_df_qa_filtered_widespreadness = checking_widespreadness_of_questions(df_qa_filtered)
    if verbose:
        print('widspreadness after filtering the questions')
        print(debug_df_qa_filtered_widespreadness)

    return df_qa_filtered, debug_df_qa_value_counts, debug_df_qa_filtered_widespreadness

def count_nb_questions_per_cue(cue = '(t, *, *, *)'):
    return [x['cue'] for x in EpisodicMemoryTemplates().templates].count(cue)

def checking_widespreadness_of_questions(df_qa):
    df_value_counts = df_qa[['cue', 'bins_items_correct_answer']].value_counts().reset_index()
    res = df_value_counts.groupby('cue').agg({
        'cue': 'count',
        'count': ['min', 'max']
    }).rename(columns={'cue': 'nb_of_bins_with_at_least_one_question', 'count': 'nb_of_questions_for_the_bin_with_the_least_and_most_questions'}, level = 0)#.reset_index()
    
    # divide by the number of questions per cue
    res = res.copy()
    res[('nb_of_questions_for_the_bin_with_the_least_and_most_questions','min')] = res[('nb_of_questions_for_the_bin_with_the_least_and_most_questions','min')]/[count_nb_questions_per_cue(x) for x in res.index.values]
    res[('nb_of_questions_for_the_bin_with_the_least_and_most_questions','max')] = res[('nb_of_questions_for_the_bin_with_the_least_and_most_questions','max')]/[count_nb_questions_per_cue(x) for x in res.index.values]

    return res

def build_qa_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max=200, target_number_of_questions_per_bin_per_q_idx=5, verbose=False,
                  bins_count = [0, 1, 2, 3, 6, np.inf], labels_count = ['0', '1', '2', '{3,4,5}', '6+'], seeds=(1,2,3)):
    df_qa_all = build_qa_all_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max, seeds = (seeds[0],seeds[1]))
    df_qa, _, _ = filtering_built_qa_func(df_qa_all, target_number_of_questions_per_bin_per_q_idx, verbose, bins_count, labels_count, seed = seeds[2])
    return df_qa.rename(columns={'level_2': 'debug_level_2'})
