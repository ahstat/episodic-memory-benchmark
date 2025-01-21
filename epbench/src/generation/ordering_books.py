from datetime import datetime
import copy
from warnings import catch_warnings, filterwarnings

def get_date_ordering(date_strings):
    """
    Creates an ordering mapping for a list of date strings.
    Returns a dictionary mapping original positions to sorted positions.
    
    Parameters:
    date_strings (list): List of date strings in format 'Month DD, YYYY'
    
    Returns:
    dict: Mapping of original position (1-based) to sorted position (1-based)
    """
    # Create a list of tuples with (position, date)
    dates = []
    for i, date_str in enumerate(date_strings, 1):
        date_obj = datetime.strptime(date_str.strip(), '%B %d, %Y')
        dates.append((i, date_obj))
    
    # Sort by date
    sorted_dates = sorted(dates, key=lambda x: x[1])
    
    # Create mapping from original position to sorted position
    position_mapping = {orig_pos: new_pos for new_pos, (orig_pos, _) in enumerate(sorted_dates, 1)}
    
    return position_mapping

def reorder_dict_by_mapping(original_dict, ordering_dict):
    """
    Reorders a dictionary based on a mapping dictionary
    
    Parameters:
    original_dict (dict): Original dictionary to reorder
    ordering_dict (dict): Dictionary containing the new position mapping
    
    Returns:
    dict: Reordered dictionary
    """
    # Create new dictionary with positions from ordering_dict
    reordered = {}
    for old_pos, value in original_dict.items():
        new_pos = ordering_dict.get(old_pos, old_pos)  # If no mapping found, keep original position
        reordered[new_pos] = value
    
    return reordered

def give_natural_ordering(d):
    if len(d) == 0:
        return [], []

    # Sort the dictionary by keys
    sorted_items = sorted(d.items())
    # Unzip the sorted items into two lists using zip
    l1, l2 = zip(*sorted_items)
    # Convert to lists (as zip returns tuples)
    return list(l1), list(l2)

def reorder_questions(my_df_of_questions, dict_new_ordering):
    # remain to reorder the questions. The strategy is to remap `correct_answer_detailed`, from which 'correct_answer'
    # and 'correct_answer_chapters' are deduced

    my_df_of_questions['correct_answer_detailed'] = [reorder_dict_by_mapping(d_current, dict_new_ordering) for d_current in my_df_of_questions['correct_answer_detailed']]
    # For index 29 (Full chapter), need to rename the right chapter
    subset = my_df_of_questions[my_df_of_questions['q_idx'] == 29]['correct_answer_detailed']
    my_df_of_questions.loc[my_df_of_questions.q_idx==29, 'correct_answer_detailed'] = [{k: f'Full chapter {k}' for k in e.keys()} for e in subset]
    # print(my_df_of_questions[my_df_of_questions['q_idx'] == 29]['correct_answer_detailed'])

    # natural ordering gives the correct order of the chapters now (since the chapters are ordered)
    my_df_of_questions['correct_answer_chapters'] = [give_natural_ordering(d)[0] for d in my_df_of_questions['correct_answer_detailed']] 
    my_df_of_questions['correct_answer'] = [give_natural_ordering(d)[1] for d in my_df_of_questions['correct_answer_detailed']]

    # For index 28, need to convert list of list to list (for 29 it is now OK)
    subset2 = my_df_of_questions[my_df_of_questions['q_idx'] == 28]['correct_answer']
    for i in range(len(subset2)):
        if len(subset2.iloc[i]) > 0:
            subset2.iloc[i] = subset2.iloc[i][0]
    my_df_of_questions.loc[my_df_of_questions.q_idx==28, 'correct_answer'] = subset2
    #print("OK")

    return my_df_of_questions

def reorder_existing_book(my_benchmark):
    with catch_warnings():
        filterwarnings('ignore')
        # we reuse the original whole object for ensuring that the questions are exactly the same
        out =  copy.deepcopy(my_benchmark) # out = my_ordered_benchmark update object
        out.book_parameters['indexing'] = 'ordered'
        out.prompt_parameters['name_universe'] = out.prompt_parameters['name_universe'] + 'Ordered'

        # new ordering
        dict_new_ordering = get_date_ordering(out.df_book_groundtruth['date']) # we need to map the input chapter to the output one

        # Step 1: reordering for `df_book_groundtruth`
        out.df_book_groundtruth['new_chapter'] = [dict_new_ordering[i] for i in out.df_book_groundtruth['chapter']]
        out.df_book_groundtruth = out.df_book_groundtruth.drop('chapter', axis=1)
        out.df_book_groundtruth = out.df_book_groundtruth.rename(columns={'new_chapter': 'chapter'}) # Rename 'new_chapter' to 'chapter'
        out.df_book_groundtruth = out.df_book_groundtruth.reset_index(drop=True) # Reset the index to get 'chapter' as a column if it's in the index
        out.df_book_groundtruth['new_chapter_2'] = out.df_book_groundtruth['chapter'] # for keeping the column along with the index (that are identical)
        out.df_book_groundtruth = out.df_book_groundtruth.set_index('chapter').sort_index() # Set the new 'chapter' column as the index
        out.df_book_groundtruth = out.df_book_groundtruth.rename(columns={'new_chapter_2': 'chapter'}) 
        #self.df_book_groundtruth

        # Step 2: reordering for the book itself
        new_split_chapters = reorder_dict_by_mapping(out.split_chapters, dict_new_ordering)
        def build_book_split_chapters(split_chapters):
            # The book itself
            res = [None]*len(split_chapters)
            for k,v in split_chapters.items():
                res[k-1] = v
            chapters = [f'Chapter {1+chapter}' for chapter in range(len(res))]
            res2 = [f"{chapter}\n\n{paragraphs}" for (paragraphs, chapter) in zip(res, chapters)]
            book = '\n\n\n'.join(res2)
            return book
        new_book = build_book_split_chapters(new_split_chapters)
        from epbench.src.generation.printing import split_chapters_func
        new_split_chapters2 = split_chapters_func(new_book) # we checked it is identical to new_split_chapters
        out.book = new_book
        out.split_chapters = new_split_chapters2

        # Step 3: reordering for the questions
        out.df_qa = reorder_questions(out.df_qa, dict_new_ordering)
        out.finetuning_questions = reorder_questions(out.finetuning_questions, dict_new_ordering)
        out.finetuning_questions_one_chapter = reorder_questions(out.finetuning_questions_one_chapter, dict_new_ordering)

        # also update `debug_mapping_chapter_idx_to_event_idx`
        out.debug_mapping_chapter_idx_to_event_idx = reorder_dict_by_mapping(out.debug_mapping_chapter_idx_to_event_idx, dict_new_ordering)

    return out