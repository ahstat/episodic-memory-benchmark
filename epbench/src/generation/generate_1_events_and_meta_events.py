from datetime import timedelta
import random
from pathlib import Path
import numpy as np
import pandas as pd
from epbench.src.generation.raw_materials import parameters_universe_func, parameters_styles_func
from epbench.src.io.io import import_list, export_list, data_folder_experiment_func

def generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking = True):
    if rechecking:
        # reproduce all the events each time, since it is quick
        events, meta_events = generate_events_and_meta_events_func(prompt_parameters)
        # save and check compared to the previous one
        # this is a useful check if the code for producing such event/meta events has changed
        export_events_func(events, data_folder, prompt_parameters, 'events.json')
        export_events_func(meta_events, data_folder, prompt_parameters, 'meta_events.json')
    else:
        # for 2000 events, it is slower, so directly load the events
        data_folder_experiment = data_folder_experiment_func(prompt_parameters)
        events_filepath = Path(data_folder) / data_folder_experiment / 'events.json'
        events = import_list(events_filepath)
        meta_events_filepath = Path(data_folder) / data_folder_experiment / 'meta_events.json'
        meta_events = import_list(meta_events_filepath)

    # testing: ensure that there is no (t,s) or (t,e) with count > 1
    # meaning of (t,s) with count > 1: two events at the same time in the same space,
    # meaning of (t,e) with count > 1: two events at the same time for the same entity.
    r=pd.DataFrame(events, columns=list('tsecd'))
    counts = r[['t','e']].value_counts()
    assert len(counts[counts > 1]) == 0, 'there is at least one (t,e) with count > 1'
    counts = r[['t','s']].value_counts()
    assert len(counts[counts > 1]) == 0, 'there is at least one (t,s) with count > 1'

    return events, meta_events

def export_events_func(events, data_folder, prompt_parameters, outfile = 'events.json'):
    """
    Work for both events and meta_events
    """
    data_folder_experiment = data_folder_experiment_func(prompt_parameters)
    events_filepath = Path(data_folder) / data_folder_experiment / outfile
    events_filepath.parent.mkdir(parents=True, exist_ok=True)
    # if the events have been saved before, check the equality and update them only if more events have been created
    if events_filepath.is_file():
        events_saved = import_list(events_filepath)
        Nmax = min(len(events_saved), len(events))
        if not events_saved[:Nmax] == events[:Nmax]:
            raise ValueError(f'saved {outfile} is not the same compared to the produced elements')
        if len(events) > len(events_saved):
            export_list(events, events_filepath)
    else:
        export_list(events, events_filepath)
    return 0

def generate_events_and_meta_events_func(prompt_parameters = {'nb_events': 100, 'seed': 0, 'distribution_events': {'name': 'geometric', 'param': 0.1}, 'name_universe': 'default', 'name_styles': 'default'}):
    # prompt generation parameters
    nb_events = prompt_parameters['nb_events']
    seed_events = prompt_parameters['seed']
    name_universe = prompt_parameters['name_universe']
    name_styles = prompt_parameters['name_styles']
    events = generate_events(nb_events, seed_events, prompt_parameters['distribution_events'], name_universe)
    meta_events = generate_meta_events(nb_events, seed_events, name_styles)
    return events, meta_events

def generate_events(nb_events = 2000, seed_events = 0, distribution_events = {'name': 'geometric', 'param': 0.1}, name_universe = 'default', N_universe = 100, seed_universe = 0):
    """Generate all the events, each event being described concisely as a list.

        Args:
            nb_events: number of event to create, each event being created by picking a random element within the
            universe for each category (date, location, entity, content). Increasing the number of events allows 
            more frequent conflicts
            seed_events: seed for generating events given the universe
            name_universe: name of the universe, associated to dates, locations, entities and contents
            N_universe: size of the number of options in each category for building an event, i.e. the universe 
            consists of N_universe dates, locations, entities, and contents (the additional content details is fixed to 30)
            Note that lowering the universe size allows more frequent conflicts. N_universe should be in 1..100
            seed_universe: seed for generating the universe

        Returns: a list of tuples, each element of the list corresponding to a single event. Each event has a date,
        a place, an entity, a content, one specific content detail, and the personality of the entity (this last one as 
        a list of length 2, with summary and details). Example of a single element of the list: 
        ('October 17, 2025', 'Empire State Building', 'Julian Stewart', 'Tech Hackathon', 'Discussed data privacy', 
        ['The Creative Landscape Architect', "Combining artistic vision with environmental knowledge, this person excels 
        at designing outdoor spaces..."])
    """

    parameters_universe = parameters_universe_func(name_universe)
    events = generate_events_given_parameters_universe(nb_events, seed_events, N_universe, seed_universe, parameters_universe, distribution_events)
    return events

def generate_events_given_parameters_universe(nb_events, seed_events, N_universe, seed_universe, parameters_universe, distribution_events):
    # universe
    temporal, entities, spatial, content, details = generate_universe(N_universe, seed_universe, parameters_universe)

    # events took in this universe
    # take three times more because duplicates of (temporal, entities) may happen
    if nb_events <= 200:
        events = [generate_event(temporal, entities, spatial, content, details, 10000000*seed_events+seed, distribution_events) for seed in range(3*nb_events)]
    elif nb_events <= 800:
        events = [generate_event(temporal, entities, spatial, content, details, 10000000*seed_events+seed, distribution_events) for seed in range(5*nb_events)]
    else:
        # generate more events, that will trimmed to only keep the valid one (e.g., no duplicated entity at the same time in two different places)
        events = [generate_event(temporal, entities, spatial, content, details, 10000000*seed_events+seed, distribution_events) for seed in range(3000*nb_events)]

    # We assess below whether two different events can have the same subset of elements:
    # Same (t), different s or e or c: OK [meaning, it is possible that 2 events happen at the same time, with different space]
    # Same (s), different t or e or c: OK
    # Same (e), different t or s or c: OK
    # Same (c), different t or s or e: OK
    # 
    # Same (t,s), different e or c: possible but excluded (given the mode of generation of the events) [it is not allowed to get different main entities, or different main contents, in the same (time, space)]
    # Same (t,e), different s or c: possible and excluded (given the mode of generation of the events) [it is not allowed to get a main entity at fixed time in different space or doing different main contents]
    # [this exclude all the triplets containing (t,s) or (t,e), and the quadruplet]
    # 
    # Same (t,c), different s or e: OK [meaning, it is possible that 2 events happen at the same time with the same content, with different space]
    # Same (s,e), different t or c: OK [meaning, it is possible that 2 events happen at the same space for the same entity, at different time]
    # Same (s,c), different t or e: OK
    # Same (e,c), different t or s: OK
    # 
    # Triplet remaining that have neither (t,s) or (t,e), so only one remaining:
    # Same (s,e,c), different t: OK
    #
    # Globally there are 2 conditions to exclude: (t,e) and (t,s)

    # filter events by keeping only one among identical (temporal, entities)
    unique_events1 = []
    nb_duplicates1 = 0
    seen1 = set()
    for event in events:
        t, _, e, _, _ = event
        key = (t, e)
        if key not in seen1:
            unique_events1.append(event)
            seen1.add(key)
        else:
            nb_duplicates1 = nb_duplicates1+1
    if len(unique_events1) < nb_events:
        print(len(unique_events1))
        raise ValueError("Not enough unique events (at step (t,e)).")
    
    # filter events by keeping only one among identical (temporal, spatial)
    unique_events2 = []
    nb_duplicates2 = 0
    seen2 = set()
    for event in unique_events1:
        t, s, _, _, _ = event
        key = (t, s)
        if key not in seen2:
            unique_events2.append(event)
            seen2.add(key)
        else:
            nb_duplicates2 = nb_duplicates2+1
    if len(unique_events2) < nb_events:
        print(len(unique_events2))
        raise ValueError("Not enough unique events (at step (t,s)).")

    return unique_events2[:nb_events]

def generate_universe(N_universe, seed_universe, parameters_universe):
    check_for_duplicates(parameters_universe)
    check_for_amount(N_universe, parameters_universe)
    temporal = generate_temporal(N_universe, parameters_universe['start_date'], parameters_universe['end_date'], seed_universe)
    entities = generate_entities(N_universe, parameters_universe['first_names'], parameters_universe['last_names'], seed_universe)
    spatial = generate_spatial(N_universe, parameters_universe['locations'], seed_universe)
    content = generate_content(N_universe, parameters_universe['contents'], seed_universe)
    details = generate_details(content, parameters_universe['content_details'])
    return temporal, entities, spatial, content, details

def check_for_duplicates(parameters_universe):
    duplicated_first_names = find_duplicates(parameters_universe['first_names'])
    duplicated_last_names = find_duplicates(parameters_universe['last_names'])
    duplicated_locations = find_duplicates(parameters_universe['locations'])
    duplicated_contents = find_duplicates(parameters_universe['contents'])
    if len(duplicated_first_names) > 0:
        print(duplicated_first_names)
        raise ValueError('Duplicated first names')
    if len(duplicated_last_names) > 0:
        print(duplicated_last_names)
        raise ValueError('Duplicated last names')
    if len(duplicated_locations) > 0:
        print(duplicated_locations)
        raise ValueError('Duplicated locations')
    if len(duplicated_contents) > 0:
        print(duplicated_contents)
        raise ValueError('Duplicated content')
    ## We are not allowing duplicated in the content details for a single content
    if (sum([len(find_duplicates(v)) for (_,v) in parameters_universe['content_details'].items()])) != 0:
        print([find_duplicates(v) for (_,v) in parameters_universe['content_details'].items()])
        raise ValueError('Duplicated content details for at least one content')
    ## We allow duplicate in the content details for different contents, e.g. 'Answered audience questions' is a content detail for
    ## 'Art Exhibition Opening', 'Film Premiere', and 'Political Rally', for the default `name_universe`
    # concatenated_content_details = [item for sublist in parameters_universe['content_details'].values() for item in sublist]
    # print(find_duplicates(concatenated_content_details))
    return 0

def find_duplicates(lst):
    seen = {}
    duplicates = []
    
    for item in lst:
        if item in seen:
            if seen[item] == 1:
                duplicates.append(item)
            seen[item] += 1
        else:
            seen[item] = 1
    return duplicates

def check_for_amount(N_universe, parameters_universe):
    if len(parameters_universe['first_names']) < N_universe:
        print(len(parameters_universe['first_names']))
        raise ValueError('Too few first names')
    if len(parameters_universe['last_names']) < N_universe:
        print(len(parameters_universe['last_names']))
        raise ValueError('Too few last names')
    if len(parameters_universe['locations']) < N_universe:
        print(len(parameters_universe['locations']))
        raise ValueError('Too few locations')
    if len(parameters_universe['contents']) < N_universe:
        print(len(parameters_universe['contents']))
        raise ValueError('Too few content')
    if not set(list(parameters_universe['content_details'].keys())) == set(parameters_universe['contents']):
        raise ValueError('content_details has not the keys equal to contents values')
    len_content_details_for_each_content = [len(v) for v in parameters_universe['content_details'].values()]
    if len(set(len_content_details_for_each_content)) > 1:
        raise ValueError('there are some content with a different number of options regarding content_details')
    return 0

def generate_temporal(N, start_date, end_date, seed):
    temporal = generate_dates(start_date, end_date, N, seed)
    # Convert datetime objects to string format
    temporal = [date.strftime('%B %d, %Y') for date in temporal]
    return temporal

def generate_dates(start_date, end_date, num_dates, seed = 1):
    """Generate temporal dates (shape is: temporal = ['January 1, 2024', 'March 15, 2024', ...])"""
    random.seed(seed)
    date_range = (end_date - start_date).days
    random_days = random.sample(range(date_range), num_dates)
    dates = [start_date + timedelta(days=day) for day in random_days]
    return dates # sorted(dates)

def generate_entities(N, first_names, last_names, seed = 1):
    """Generate entities (shape is: entities = ['Emma Thompson', 'Liam Rodriguez', ...])"""
    # code not optimize but works for N < 30000
    random.seed(seed+1)
    # Generate N unique full names
    entities = []
    while len(entities) < N:
        first = random.choice(first_names)
        last = random.choice(last_names)
        full_name = f"{first} {last}"
        if full_name not in entities:
            entities.append(full_name)
    return entities

def generate_spatial(N, locations, seed = 1):
    """Generate spatial (shape is: spatial = ['Central Park', 'Tokyo Tower', ...])"""
    # Ensure we have at least 100 unique locations
    if len(set(locations)) < N:
        print("Warning: Not enough unique locations in the list. Some may be duplicated.")
    # Shuffle the list to randomize the order
    tmp = locations.copy()
    random.Random(seed).shuffle(tmp)
    # Take the first N locations
    spatial = tmp[:N]
    if len(set(spatial)) < N:
        raise ValueError("Duplicates exist.")    
    return spatial

def generate_content(N, contents, seed = 1):
    """Generate contents (shape is: contents = ['Wedding Ceremony', 'Scientific Conference',  ...])"""
    # Ensure we have at least 100 unique contents
    if len(set(contents)) < N:
        print("Warning: Not enough unique contents in the list. Some may be duplicated.")
    # Shuffle the list to randomize the order
    tmp = contents.copy()
    random.Random(seed).shuffle(tmp)
    # Take the first N locations
    content = tmp[:N]
    if len(set(content)) < N:
        raise ValueError("Duplicates exist.")    
    return content

def generate_details(content, content_details):
    """Generate details (shape is: details = {'Wedding Ceremony': ['exchanged vows', 'celebrated with family', ...], 'Scientific Conference': ['presented research', ...]})"""
    return {c: content_details[c] for c in content}

def idx_candidate_func(p):
    return np.random.geometric(p=p, size=1).tolist()[0]-1

def censored_geometric_choice(p, my_list):
    '''
    Take within a list by following a geometric distribution, while
    ensuring that the selected candidate has an index within the list
    (since the geometric distribution is unbounded)
    '''
    n = len(my_list)
    # initial candidate
    idx_candidate = idx_candidate_func(p)
    if idx_candidate >= n: # this is uncommon for p large and/or n large
        rep = 0 # count repetitions to prevent infinite loop
        while idx_candidate >= n:
            #print(idx_candidate)
            idx_candidate = idx_candidate_func(p)
            rep = rep + 1
            if rep > 10:
                raise ValueError('too many iterations, p should be larger or the number of events in the universe should be larger')
    return my_list[idx_candidate]

def generate_event(temporal, entities, spatial, content, details, seed, distribution_events = {'name': 'geometric', 'param': 0.1}):
    '''Generate a single event based on a specified seed'''
    if distribution_events['name'] == 'uniform':
        # distribution_events = {'name': 'uniform'}
        random.seed(seed)
        t = random.choice(temporal)
        e = random.choice(entities)
        s = random.choice(spatial)
        c = random.choice(content)
        cd = random.choice(details[c])
    elif distribution_events['name'] == 'geometric':
        # distribution_events = {'name': 'geometric', 'param': 0.1}
        p = distribution_events['param']
        np.random.seed(seed)
        t = censored_geometric_choice(p, temporal)
        e = censored_geometric_choice(p, entities)
        s = censored_geometric_choice(p, spatial)
        c = censored_geometric_choice(p, content)
        cd = random.choice(details[c])
    else:
        raise ValueError('unknown distribution selected')
    return [t, s, e, c, cd]

def generate_meta_events(nb_events = 2000, seed_events = 0, name_styles = 'default'):
    parameters_styles = parameters_styles_func(name_styles)
    nb_paragraphs = parameters_styles['nb_paragraphs']
    styles = parameters_styles['styles']

    # Explanation of the seeds:
    # 1. We don't want to change the randomness when the parameter `nb_events` is changing,
    #  so we need to restate the seed at each generation.
    # 2. We don't want the randomess to be the same for each location, date, etc., 
    #  so we need to shift the random seed for each
    # 3. We don't want correlation between two different seed_events, so we multiply by 20
    #  so that [20*s+0:5] is disjoint from the next seed interval [20*(s+1)+0:5]
    random.seed(20*seed_events+0)
    events_nb_paragraphs = [random.choice(nb_paragraphs) for _ in range(nb_events)]
    random.seed(20*seed_events+1)
    events_idx_location = [random.randint(1, x) for x in events_nb_paragraphs]
    random.seed(20*seed_events+2)
    events_idx_date = [random.randint(1, x) for x in events_nb_paragraphs]
    random.seed(20*seed_events+3)
    events_idx_entity = [random.randint(1, x) for x in events_nb_paragraphs]
    random.seed(20*seed_events+4)
    events_idx_content = [random.randint(1, x) for x in events_nb_paragraphs]
    random.seed(20*seed_events+5)
    events_style = [random.choice(styles) for _ in range(nb_events)]

    events_idx_paragraph =  [{'location': s, 'date': t, 'entity': e, 'content': c} \
        for (s,t,e,c) in zip(events_idx_location, events_idx_date, events_idx_entity, events_idx_content)]
    
    return [{'nb_paragraphs': n1, 'idx_paragraph': n2, 'style': n3} \
            for (n1,n2,n3) in zip(events_nb_paragraphs, events_idx_paragraph, events_style)]

def unused_universe_func(prompt_parameters, events, N_universe = 100, seed_universe = 0):
    # retrieve the events that are used
    # events, _ = generate_events_and_meta_events_func(prompt_parameters)
    # events, _ = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)

    r=pd.DataFrame(events, columns=list('tsecd'))
    used_t = list(set(r['t']))
    used_s = list(set(r['s']))
    used_e = list(set(r['e']))
    used_c = list(set(r['c']))

    # retrieve the universe from which the events have been built
    parameters_universe = parameters_universe_func(prompt_parameters['name_universe'])
    temporal, entities, spatial, content, details = generate_universe(N_universe, seed_universe, parameters_universe)

    # since the distribution is geometric, there is (for a reasonable number of generated events)
    # unused elements in the universe.
    unused_t = [x for x in temporal if x not in used_t]
    unused_s = [x for x in spatial if x not in used_s]
    unused_e = [x for x in entities if x not in used_e]
    unused_c = [x for x in content if x not in used_c]
    unused_d = dict((k, details[k]) for k in unused_c)

    minimum_remaining_set_of_unused = min(len(unused_t), len(unused_s), len(unused_e), len(unused_c))
    if minimum_remaining_set_of_unused < 20:
        print(minimum_remaining_set_of_unused)
        raise ValueError("Very few unused content remaining in the universe")

    return unused_t, unused_s, unused_e, unused_c, unused_d
