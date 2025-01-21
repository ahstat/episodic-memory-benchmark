
from epbench.src.generation.raw_materials import parameters_styles_func
from epbench.src.generation.generate_1_events_and_meta_events import generate_events

def s_func(nb):
    return "" if nb == 1 else "s"

def idx_str_func(idx):
    return ",".join(map(str, idx)) if isinstance(idx, list) else idx

def complete_prompt_func(event=generate_events(1)[0], nb_paragraphs = 5, idx_paragraph = {'location': [1,2], 'date': 2, 'entity': [3], 'content': [4]}, style='polar', name_styles = 'default'):    
    idx_location = idx_paragraph['location']
    idx_date = idx_paragraph['date']
    idx_entity = idx_paragraph['entity']
    idx_content = idx_paragraph['content']

    idx_location_str = idx_str_func(idx_location) # 1,2
    idx_date_str = idx_str_func(idx_date) # 2
    idx_entity_str = idx_str_func(idx_entity) # 3
    idx_content_str = idx_str_func(idx_content) # 4

    date = event[0] # October 17, 2025
    location = event[1] # Empire State Building
    entity = event[2] # Julian Stewart
    first_name, last_name = entity.split() # it works because we tested that first_names and last_names have no space during generation
    content = event[3].lower() # tech hackathon
    content_single_detail = event[4][0].lower() + event[4][1:] # discussed data privacy

    parameters_styles = parameters_styles_func(name_styles)

    if style not in parameters_styles['styles']:
        raise ValueError('unknown style, should be within: ' + ', '.join(parameters_styles['styles']))

    style_description = ', '.join(parameters_styles['style_to_description'][style])

    # Add `s` in the prompt when the number of pages, paragraphs etc. are > 1
    s_paragraphs = s_func(nb_paragraphs)
    s_paragraphs_location = s_func(len(idx_location)) if isinstance(idx_location, list) else ""
    s_paragraphs_date = s_func(len(idx_date)) if isinstance(idx_date, list) else ""
    s_paragraphs_entity = s_func(len(idx_entity)) if isinstance(idx_entity, list) else ""
    s_paragraphs_content = s_func(len(idx_content)) if isinstance(idx_content, list) else ""

    if nb_paragraphs > 1:
        str_numbering = f"1. Divide the text into {nb_paragraphs} paragraph{s_paragraphs}. Number each paragraph (1), (2), etc."
    else:
        str_numbering = f"1. Write a single paragraph. Number (1) before this single paragraph"

    result = f"""\
Write a detailed novel excerpt in a {style} style about {entity} attending a {content}. \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, while maintaining novel-appropriate paragraph lengths.
2. Gradually reveal key information:
- Full location '{location}': must appear verbatim in paragraph{s_paragraphs_location} {idx_location_str} only and nowhere else in the text
- Full date '{date}': must appear verbatim in paragraph{s_paragraphs_date} {idx_date_str} only and nowhere else in the text
- Full name '{entity}': must appear verbatim in paragraph{s_paragraphs_entity} {idx_entity_str} only and nowhere else in the text
- Full detail that '{first_name} {content_single_detail}': must appear verbatim in paragraph{s_paragraphs_content} {idx_content_str} and nowhere else in the text
3. Subtly distribute details about location, date, main character, and event across all paragraphs.

Content and Setting:
1. Focus on {first_name}'s experiences, observations, and interactions during the {content}.
2. Vividly describe surroundings, atmosphere, and {first_name}'s emotions.
3. Include the detail that {first_name} {content_single_detail}.
4. Limit the timeframe to a single day and confine all action to {location}.

Characters:
1. Refer to other characters as $entity_X (where X is a number).
2. Omit background information about {first_name} and other characters.

Style and Tone:
1. Use vivid, sensory details to bring the scene to life.
2. Incorporate elements of the {style} style, including {style_description}.
3. Maintain a consistent narrative voice throughout the excerpt.

Restrictions:
1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.

Craft a seamless narrative that gradually reveals information while maintaining reader engagement throughout the excerpt.
"""
    
    if name_styles == "news":
        result = f"""\
Write a detailed news excerpt in a {style} style about {entity} witnessing a {content}. \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, while maintaining news-appropriate paragraph lengths.
2. Gradually reveal key information:
- Full location '{location}': must appear verbatim in paragraph{s_paragraphs_location} {idx_location_str} only and nowhere else in the text
- Full date '{date}': must appear verbatim in paragraph{s_paragraphs_date} {idx_date_str} only and nowhere else in the text
- Full name '{entity}': must appear verbatim in paragraph{s_paragraphs_entity} {idx_entity_str} only and nowhere else in the text
- Full detail that '{first_name} {content_single_detail}': must appear verbatim in paragraph{s_paragraphs_content} {idx_content_str} and nowhere else in the text
3. Subtly distribute details about location, date, main character, and event across all paragraphs.

Content and Setting:
1. Focus on {first_name}'s experiences, observations, and interactions during the {content}.
2. Vividly describe surroundings, atmosphere, and {first_name}'s emotions.
3. Include the detail that {first_name} {content_single_detail}.
4. Limit the timeframe to a single day and confine all action to {location}.

Characters:
1. Refer to other characters as $entity_X (where X is a number).
2. Omit background information about {first_name} and other characters.

Style and Tone:
1. Use vivid, sensory details to bring the scene to life.
2. Incorporate elements of the {style} style, including {style_description}.
3. Maintain a consistent news voice throughout the excerpt.

Restrictions:
1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.
5. Do not put any title

Craft a seamless news that gradually reveals information while maintaining reader engagement throughout the excerpt.
"""
        #print(result)

    if name_styles == "scifi":
        result = f"""\
Write a detailed scifi excerpt in a {style} style about {entity} witnessing a {content}. \
The story takes place on {date}, at {location}, where {entity} {content_single_detail}. Follow these guidelines:

Structure and Information Reveal:
{str_numbering}, while maintaining novel-appropriate paragraph lengths.
2. Gradually reveal key information:
- Full location '{location}': must appear verbatim in paragraph{s_paragraphs_location} {idx_location_str} only and nowhere else in the text
- Full date '{date}': must appear verbatim in paragraph{s_paragraphs_date} {idx_date_str} only and nowhere else in the text
- Full name '{entity}': must appear verbatim in paragraph{s_paragraphs_entity} {idx_entity_str} only and nowhere else in the text
- Full detail that '{first_name} {content_single_detail}': must appear verbatim in paragraph{s_paragraphs_content} {idx_content_str} and nowhere else in the text
3. Subtly distribute details about location, date, main character, and event across all paragraphs.

Content and Setting:
1. Focus on {first_name}'s experiences, observations, and interactions during the {content}.
2. Vividly describe surroundings, atmosphere, and {first_name}'s emotions.
3. Include the detail that {first_name} {content_single_detail}.
4. Limit the timeframe to a single day and confine all action to {location}.

Characters:
1. Refer to other characters as $entity_X (where X is a number).
2. Omit background information about {first_name} and other characters.

Style and Tone:
1. Use vivid, sensory details to bring the scene to life.
2. Incorporate elements of the {style} style, including {style_description}.
3. Maintain a consistent narrative voice throughout the excerpt.

Restrictions:
1. Only mention {location} and {date}; avoid other locations or dates.
2. Exclude explicit introductions, conclusions, or character backgrounds.
3. Focus exclusively on the events of this particular {content}.
4. Do not use a too common starting sentence.

Craft a seamless narrative that gradually reveals information while maintaining reader engagement throughout the excerpt.
"""
        #print(result)

    return result

def generate_prompts(events, meta_events, name_styles = 'default'):
    if len(events) != len(meta_events):
        raise ValueError('each event should be associated to a meta-event')
    #i=0
    #event=events[i]
    #print_events([events[i]])
    #print(meta_events[i])
    all_events_prompts = [complete_prompt_func(e, m['nb_paragraphs'], m['idx_paragraph'], m['style'], name_styles) for (e, m) in zip(events, meta_events)]
    return all_events_prompts

def system_prompt_func():
    return "You are a creative fiction writer specializing in detailed, atmospheric novel excerpts. \
Your task is to generate vivid, immersive scenes based on specific prompts."

def system_prompt_verification_func():
    return """You are a content checker AI. Your tasks:
1. Read the given text carefully.
2. Answer true/false questions about the text.
3. Respond in JSON format.
Be accurate and concise. Only use information explicitly stated in the text."""

def verification_prompt_func(text):
    # Closed questions worked best.
    # boolean_answer = "Answer only with yes or no."
    verification_prompt_location = f"Does the following text takes place in a single geographical (longitude, latitude)?"# {boolean_answer}"
    verification_prompt_date = f"Does the following text takes place in a single temporal day?"# {boolean_answer}"
    verification_prompt_entity = f"Does the following text has a single main character?"# {boolean_answer}"
    verification_prompt_content = f"Does the following text has a single main event happening at that location that day (further cut into the events of the day)?"# {boolean_answer}"
    # Open questions (e.g. as a list) didn't work with our prompts:
    # "when does the following text take place, answer as a python list"
    #   --> ["Hackathon event", "Empire State Building", "Rooftop observation area"]
    # "what are the explicitly named characters, answer as a python list"
    #   --> ["Julian", "Mr. Flibble", "Julian Stewart"]
    # "what are the main events, answer as a python list"
    #   --> [2025, 10, 17]

    verification_prompt = f"""Please analyze the following text enclosed between [TEXT START] and [TEXT END] markers, and answer the four questions below with a simple true or false. Provide your answers in a JSON format with the question numbers as keys and the boolean answers as values.

    [TEXT START]
    {text}
    [TEXT END]

    Questions:
    1. {verification_prompt_location}
    2. {verification_prompt_date}
    3. {verification_prompt_entity}
    4. {verification_prompt_content}

    Your response should be in this JSON format:
    {{
        "1": [boolean],
        "2": [boolean],
        "3": [boolean],
        "4": [boolean]
    }}
    """

    return verification_prompt

def verification_prompt_debug_func(text, kind = 'location'):
    verification_prompt_location = f"Does the following text takes place in a single geographical (longitude, latitude)?"# {boolean_answer}"
    verification_prompt_date = f"Does the following text takes place in a single temporal day?"# {boolean_answer}"
    verification_prompt_entity = f"Does the following text has a single main character?"# {boolean_answer}"
    verification_prompt_content = f"Does the following text has a single main event happening at that location that day (further cut into the events of the day)?"# {boolean_answer}"
    if kind == 'location':
        verification_prompt = verification_prompt_location
    elif kind == 'date':
        verification_prompt = verification_prompt_date
    elif kind == 'entity':
        verification_prompt = verification_prompt_entity
    elif kind == 'content':
        verification_prompt = verification_prompt_content
    else:
        raise ValueError('Unknown kind parameter')

    verification_prompt_debug = f"""Please analyze the following text enclosed between [TEXT START] and [TEXT END] markers, and answer the question below.

    [TEXT START]
    {text}
    [TEXT END]

    Question: {verification_prompt}

    Your response:
    """

    return verification_prompt_debug
