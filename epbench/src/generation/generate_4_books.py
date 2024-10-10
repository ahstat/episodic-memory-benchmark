from epbench.src.generation.generate_3_secondary_entities import get_final_samples
import pandas as pd

def build_chapter(idx_chapters, d):
    # The book itself
    res = [d[e]['paragraphs'] for e in idx_chapters]
    if any([x is None for x in res]):
        raise ValueError('at least one chapter is invalid and should be removed from the list of idx_chapters')
    chapters = [f'Chapter {1+chapter}' for chapter in range(len(res))]
    res2 = [f"{chapter}\n\n{paragraphs}" for (paragraphs, chapter) in zip(res, chapters)]
    book = '\n\n\n'.join(res2)

    # The ground truth
    events = [d[e]['event'] for e in idx_chapters]
    meta_events = [d[e]['meta_event'] for e in idx_chapters]
    post_entities = [d[e]['post_entities'] for e in idx_chapters]
    details = [event[4] for event in events]

    # extract all events as a data frame

    # single tuple
    dates = [event[0] for event in events]
    locations = [event[1] for event in events]
    entities = [event[2] for event in events]
    contents = [event[3] for event in events]
    chapters_idx = [1+chapter for chapter in range(len(res))]
    df_groundtruth = pd.DataFrame({'chapter': chapters_idx, 'date': dates, 'location': locations, 'entity': entities, 'content': contents, 'post_entities': post_entities}).reset_index(drop=True)
    df_groundtruth = df_groundtruth.set_index('chapter', drop = False)

    # Count occurrences of each date
    date_counts = df_groundtruth['date'].value_counts()
    df_groundtruth['n_date'] = df_groundtruth['date'].map(date_counts)

    location_counts = df_groundtruth['location'].value_counts()
    df_groundtruth['n_location'] = df_groundtruth['location'].map(location_counts)

    entity_counts = df_groundtruth['entity'].value_counts()
    df_groundtruth['n_entity'] = df_groundtruth['entity'].map(entity_counts)

    content_counts = df_groundtruth['content'].value_counts()
    df_groundtruth['n_content'] = df_groundtruth['content'].map(content_counts)

    # for debug only: keep track of the original index (from 0 to prompt_parameters['nb_events']-1)
    # the missing indexes have failed the verifications
    df_groundtruth['raw_generated_paragraph_idx'] = idx_chapters

    return book, df_groundtruth

def default_idx_chapters_func(d):
    possible_events = [e for e in range(len(d))]
    valid_idxs = [e for e in possible_events if d[e]['is_valid']]
    return valid_idxs

def book_indexing_func(d, book_parameters):
    indexing = book_parameters['indexing']
    if indexing == 'default':
        idx_chapters = default_idx_chapters_func(d)
    else:
        raise ValueError('unknown indexing')
    return idx_chapters
