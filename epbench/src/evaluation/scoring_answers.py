from typing import Set, Dict, Any, List
import json
import re
from epbench.src.models.models_wrapper import ModelsWrapper
from scipy.stats import kendalltau
import numpy as np

def judge_prompt_func(retrieval_type, correct_answer, llm_answer, correct_answer_long = None):
    ## * Prompt 1:
    # the 1,2,3,4 not following the {...}
    #
    #    prompt = f"""
    #You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.
    #
    #Question type: {retrieval_type}
    #Groundtruth: {correct_answer}
    #AI-generated answer: {llm_answer}
    #
    #Your task:
    #1. Identify all relevant items in the AI-generated answer that correspond to the question type.
    #2. Determine which items from the AI answer match the groundtruth, considering synonyms, paraphrases, or close meanings.
    #3. List any additional correct items in the AI answer not present in the groundtruth.
    #4. Provide your evaluation in the following JSON format:
    #{{
    #    "matched_items": ["list", "of", "matched", "items"],
    #    "additional_correct_items": ["list", "of", "additional", "correct", "items"],
    #    "missed_items": ["list", "of", "groundtruth", "items", "not", "in", "AI", "answer"],
    #    "explanation": "Brief explanation of your evaluation"
    #}}
    #"""
    #
    ## * Prompt 2:
    #    prompt = f"""
    #You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.
    #
    #
    #Question type: {retrieval_type}
    #Groundtruth: {correct_answer}
    #AI-generated answer: {llm_answer}
    #
    #
    #Your task:
    #- Identify all items in the AI-generated answer that are relevant to the question type.
    #- Determine a matching score between 0 and 1 for each relevant item of the AI answer. Give 1 if the item match the groundtruth, considering synonyms, paraphrases, or close meanings.
    #- Provide a brief explanation of the evaluation
    #
    #Provide your evaluation in the following JSON format:
    #{{
    #    "matching_score": {{"relevant_item_1": 0.8, 
    #                       "relevant_item_2": 1,
    #                       "relevant_item_3": 0.1}},
    #    "explanation": "Brief explanation of your evaluation"
    #}}
    #"""
    ## * Prompt 3:
    # take the ground truth point of view, since all the elements are already known, it's easier to be sure about the quantity.
    
    
    d = [{x: "score_between_0_and_1"} for x in correct_answer] # to keep the order, needed for the chronological events
    #if(len(set(correct_answer)) != len(correct_answer)):
    #    correct_answer = append_number_if_duplicate(correct_answer)

    # first differentiate possible duplicated correct_answer
    # e.g., for a question "Enumerate in chronological order all the activities that Mila Gonzalez has been involved in.",
    # the activities (correct_answer) are ['Theater Performance','Theater Performance'] (that happens at different dates)
    # In this case, `d` without appending number would have only one element, which is incorrect. 
    # d = {x: 'score_between_0_and_1' for x in correct_answer}
    if correct_answer_long is None:
        correct_answer_long = correct_answer
        adding_text=''
    else:
        adding_text=f'- The matching score should be of length 1, only "matching_score": {json.dumps(d)}' # otherwise GPT4 always tries to list many elements

    prompt = f"""
You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.


Question type: {retrieval_type}
Groundtruth: {correct_answer_long}
AI-generated answer: {llm_answer}


Your task:
- Identify all unique items in the AI-generated answer that are relevant to the question type. Answer an empty list [] for this field in case of at least one negative information (e.g., when the answer begins by telling there is no information, or cannot answer)
- Determine a matching score between 0 and 1 for each ground truth item. Give 1 if the item has been found in the relevant items of the AI-generated answer, considering synonyms, paraphrases, or close meanings. Give 0.5 if the item could be considered related to any AI-generated item but without being explicitly stated as such. Give 0 if the item missed mentioning a specific AI-generated item.
- Provide a brief explanation of the evaluation
{adding_text}

Provide your evaluation in the following JSON format:
{{
    "identified_items_in_AI_answer": ["AI_answer_item_1", "AI_answer_item_2", ...],
    "matching_score": {json.dumps(d)}
    "explanation": "Brief explanation of your evaluation"
}}
"""
    # Note: Changed {d} to {json.dumps(d)} to replace simple quotes to double quotes:
    # d = [{'October 13, 2024': 0}, {'May 11, 2026': 1}, {'September 22, 2026': 1}]
    # output1 = f"{json.dumps(d)}" # produce double quotes!
    # output2 = f"{d}" # produce simple quotes (not JSON compliant)
    # print(output1)
    # print(output2)
    return prompt

def append_number_if_duplicate(mylist_in):
    # https://stackoverflow.com/questions/30650474
    newlist = []
    mylist = mylist_in.tolist()
    for i, v in enumerate(mylist):
        totalcount = mylist.count(v)
        count = mylist[:i].count(v)
        #newlist.append(f"{v} ({str(count + 1)})" if (totalcount > 1 & count > 0) else v)
        newlist.append(f"{v} ({str(count + 1)})" if (totalcount > 1) else v)
    return newlist

def f1_score_func(precision, recall):
    # Issue in precision/recall value happen either when both are 0, or when any is None
    if precision == 0 and recall == 0:
        f1_score = 0
    elif precision is not None and recall is not None:
        f1_score = 2 * (precision*recall) / (precision + recall)
    elif precision is None and recall is None:
        # happen only when nb_preds = nb_gt = 0, perfectly not identifying anything
        f1_score = 1
    else:
        # either no predictions but #gt>0, or at least one prediction but #gt = 0
        f1_score = 0
    return f1_score

def evaluate_answer(llm_answer: str, correct_answer: Set[str], retrieval_type: str, my_model: ModelsWrapper, correct_answer_long: str, get_style: str) -> Dict[str, Any]:
    # no policy or universe for the first evaluation /!\ do not add [the policy can be updated afterwards]
    if correct_answer_long is None:
        correct_answer_long = correct_answer

    # Prepare the prompt for the judge LLM
    judge_prompt = judge_prompt_func(retrieval_type, correct_answer, llm_answer, correct_answer_long)
    print(judge_prompt)
    
    # Get the judge LLM's evaluation
    judge_response = my_model.generate(user_prompt = judge_prompt, system_prompt = "You are an expert in memory tests.", max_new_tokens = 4096)
    print(judge_response)
    
    # Parse the judge's response
    try:
        evaluation = json.loads(judge_response)
    except json.JSONDecodeError:
        print("json decode error, used ast instead")
        # If JSON parsing fails, use regex to extract the JSON part
        import re
        import ast
        judge_response = re.sub(r'": "', '": """', judge_response)
        judge_response = re.sub(r'"\n}', '"""\n}', judge_response)
        evaluation = ast.literal_eval(judge_response)
        print(judge_response)
        
    return generate_metric_original(correct_answer, evaluation) # keep the original policy there

def remove_duplicates(input_list):
    seen = set()
    output = []
    for item in input_list:
        key = next(iter(item))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output

def generate_metric_original(correct_answer, evaluation):
    """
    Generate the metrics given the LLM part already computed (this one is saved to the disk, for consistency of the written file, do not change)
    The computation of the F1-score with other strategies is computed in another function
    """

    # Calculate metrics
    nb_gt = len(correct_answer)
    
    # matching_score from the groundtruth point of view
    predictions = evaluation['identified_items_in_AI_answer'] # set(evaluation['identified_items_in_AI_answer'])
    nb_preds = len(predictions)

    # in all cases, we are lenient w.r.t. nb_preds
    # Reason for almost all types:
    # a. in case of full event, there gt=[Full chapter] of length 1, but the predicted detail list can be > 1
    # b. in case of latest content, there is gt=[single content] of length 1, but the prediction can give more details regarding the content
    # c. in case of post entities retrieved, whereas only entities are needed
    if (nb_preds > nb_gt) & (nb_gt > 0):
        nb_preds = nb_gt
    # old code:
    #if retrieval_type == "Full event details":
    #    #in this case, there is only one or zero element to score (i.e. nb_gt = 1)
    #    if nb_preds >= 1: # if the element exists, there is only one prediction, so set to 1
    #        nb_preds = 1
    #    # otherwise it is nb_preds = 0, which is also valid
    #elif get_style == "latest":
    #    # in this case, there is only one or zero element to score (i.e. nb_gt = 1)
    #    if nb_preds >= 1: # if the element exists, there is only one prediction since we ask for the latest, so set to 1
    #        nb_preds = 1
    #    # otherwise it is nb_preds = 0, which is also valid
    
    gt_alt = [list(x.keys())[0] for x in evaluation['matching_score']] # set([x.keys() for x in evaluation['matching_score']])
    nb_gt_alt = len(gt_alt) # nb_gt computed differently
    if nb_gt != nb_gt_alt:
        raise ValueError('nb_gt has been found different to nb_gt_alt')

    sum_scores = sum([float(list(x.values())[0]) for x in evaluation['matching_score']]) # sum(evaluation['matching_score'].values()) # between 0 and nb_preds
    precision = sum_scores / nb_preds if nb_preds > 0 else None
    recall = sum_scores / nb_gt if nb_gt > 0 else None
    f1_score = f1_score_func(precision, recall)
    return {'predicted_items': list(predictions),
            'groundtruth_items': list(gt_alt),
            'matching_groundtruth_items_score': evaluation['matching_score'],
            'explanation': evaluation['explanation'],
            'nb_preds': nb_preds,
            'nb_gt': nb_gt,
            'sum_scores': sum_scores,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score}

def generate_metric(correct_answer, evaluation, policy = 'remove_duplicates'):
    """
    Generate the metrics given the LLM part already computed (useful for updated solely the metric computation)
    """
    if policy == 'remove_duplicates':
        evaluation['matching_score'] = remove_duplicates(evaluation['matching_score']) # remove duplicates in the list of dictionaries
        correct_answer = list(dict.fromkeys(correct_answer)) # remove duplicates while keeping the order

    # Calculate metrics
    nb_gt = len(correct_answer)
    
    # matching_score from the groundtruth point of view
    predictions = evaluation['identified_items_in_AI_answer'] # set(evaluation['identified_items_in_AI_answer'])
    nb_preds_harsh = len(predictions)

    if (nb_preds_harsh > nb_gt) & (nb_gt > 0):
        nb_preds_lenient = nb_gt
    else:
        nb_preds_lenient = nb_preds_harsh

    gt_alt = [list(x.keys())[0] for x in evaluation['matching_score']] # set([x.keys() for x in evaluation['matching_score']])
    nb_gt_alt = len(gt_alt) # nb_gt computed differently
    if nb_gt != nb_gt_alt:
        raise ValueError('nb_gt has been found different to nb_gt_alt')

    # common (old and new)
    # print(evaluation['matching_score'])
    sum_scores = sum([float(list(x.values())[0]) for x in evaluation['matching_score']]) # sum(evaluation['matching_score'].values()) # between 0 and nb_preds
    precision_lenient = sum_scores / nb_preds_lenient if nb_preds_lenient > 0 else None
    precision_harsh = sum_scores / nb_preds_harsh if nb_preds_harsh > 0 else None

    recall = sum_scores / nb_gt if nb_gt > 0 else None
    f1_score_lenient = f1_score_func(precision_lenient, recall)
    f1_score_harsh = f1_score_func(precision_harsh, recall)
    return {'predicted_items': predictions,
            'groundtruth_items': gt_alt,
            'matching_groundtruth_items_score': evaluation['matching_score'],
            'explanation': evaluation['explanation'],
            'nb_preds_lenient': nb_preds_lenient,
            'nb_preds_harsh': nb_preds_harsh,
            'nb_gt': nb_gt,
            'sum_scores': sum_scores,
            'precision_lenient': precision_lenient,
            'precision_harsh': precision_harsh,
            'recall': recall,
            'f1_score_lenient': f1_score_lenient,
            'f1_score_harsh': f1_score_harsh,
            'diff_f1': f1_score_lenient-f1_score_harsh}

def update_policy_of_evaluation_to(df_generated_evaluations, policy = 'remove_duplicates'):
    df_to_update = df_generated_evaluations.copy()
    elements_for_which_f1_should_be_recomputed = [(i, x) for i, x in enumerate(df_to_update['groundtruth_items'])] # if len(x) != len(set(x))]
    for i, x in elements_for_which_f1_should_be_recomputed:
        #print(i)
        #sum_scores = sum([float(list(x.values())[0]) for x in evaluation['matching_score']]) # sum(evaluation['matching_score'].values()) # between 0 and nb_preds
        #precision = sum_scores / nb_preds if nb_preds > 0 else None
        #recall = sum_scores / nb_gt if nb_gt > 0 else None
        #f1_score = f1_score_func(precision, recall)
        #print(i)
        current_sample = df_to_update.iloc[i]
        # print(current_sample) # extract the type there
        evaluation = {
            'identified_items_in_AI_answer': list(dict.fromkeys(current_sample['predicted_items'])), # remove duplicates while keeping the order
            'matching_score': current_sample['matching_groundtruth_items_score'],
            'explanation': current_sample['explanation']}
        correct_answer = current_sample['correct_answer']
        #print(evaluation)
        res = generate_metric(correct_answer, evaluation, policy = policy)
        #n_items_correct_answer = res['nb_gt']
        #df_to_update.loc[i, 'n_items_correct_answer'] = n_items_correct_answer
        #print(df_to_update.loc[i, 'groundtruth_items'])
        #print(res['groundtruth_items'])
        #df_to_update.at[i, 'groundtruth_items'] = res['groundtruth_items']
        #df_to_update.at[i, 'matching_groundtruth_items_score'] = res['matching_groundtruth_items_score']
        #df_to_update.at[i, 'nb_preds'] = res['nb_preds']
        #df_to_update.at[i, 'nb_gt'] = res['nb_gt']
        #df_to_update.at[i, 'sum_scores'] = res['sum_scores']
        #df_to_update.at[i, 'precision'] = res['precision']
        #df_to_update.at[i, 'recall'] = res['recall']
        #df_to_update.at[i, 'f1_score'] = res['f1_score']
        df_to_update.at[i, 'predicted_items'] = res['predicted_items']
        df_to_update.at[i, 'groundtruth_items'] = res['groundtruth_items']
        df_to_update.at[i, 'matching_groundtruth_items_score'] = res['matching_groundtruth_items_score']
        df_to_update.at[i, 'explanation'] = res['explanation']
        df_to_update.at[i, 'nb_preds_lenient'] = res['nb_preds_lenient']
        df_to_update.at[i, 'nb_preds_harsh'] = res['nb_preds_harsh']
        df_to_update.at[i, 'nb_gt'] = res['nb_gt']
        df_to_update.at[i, 'sum_scores'] = res['sum_scores']
        df_to_update.at[i, 'precision_lenient'] = res['precision_lenient']
        df_to_update.at[i, 'precision_harsh'] = res['precision_harsh']
        df_to_update.at[i, 'recall'] = res['recall']
        df_to_update.at[i, 'f1_score_lenient'] = res['f1_score_lenient']
        df_to_update.at[i, 'f1_score_harsh'] = res['f1_score_harsh'] # does not make sense for Full event and Latest, in this case only the other one is valid
        df_to_update.at[i, 'diff_f1'] = res['diff_f1']

        df_to_update.at[i, 'precision'] = np.nan
        df_to_update.at[i, 'f1_score'] = np.nan
        
    return df_to_update # updated one


def process_lists_and_compute_kendall_tau(l1, l2):
    # Step 1: Remove duplicates while preserving order
    l1_no_duplicates = list(dict.fromkeys(l1))
    l2_no_duplicates = list(dict.fromkeys(l2))
    
    # Step 2: Keep only elements that are in both lists
    common_elements = list(set(l1_no_duplicates) & set(l2_no_duplicates))
    
    # Create new lists with only common elements, preserving original order
    result_l1 = [x for x in l1_no_duplicates if x in common_elements]
    result_l2 = [x for x in l2_no_duplicates if x in common_elements]
    
    # Step 3: Compute Kendall tau
    tau, p_value = kendalltau(
        [result_l1.index(x) for x in common_elements],
        [result_l2.index(x) for x in common_elements]
    )
    
    return result_l1, result_l2, tau, p_value

def judge_prompt_chronological_func(groundtruth_items, predicted_items):
    groundtruth_indexes = [x for x in range(len(groundtruth_items))]
    chronological_prompt = f"""You are an expert judge evaluating the alignment between an AI-generated list and a known groundtruth list. Your task is to match items from the predicted list to the groundtruth list, considering their order and uniqueness.

    Given:
    Groundtruth list: {groundtruth_items}
    Groundtruth indexes: {groundtruth_indexes}
    Predicted list: {predicted_items}

    Instructions:
    1. For each item in the predicted list, find the first corresponding index from the groundtruth list that hasn't been used yet.
    2. Assign indexes based on these rules:
    a. If a match is found and the groundtruth index hasn't been used, assign that index.
    b. If no match is found, or if all matching indexes have already been used, assign -1.
    3. Always use the earliest matching index from the groundtruth list, even if there's an exact match later.
    4. Provide a brief explanation of your index assignments.

    Output your evaluation in the following JSON format:
    {{
        "groundtruth_indexes": {groundtruth_indexes},
        "predicted_indexes": [index1, index2, ...],
        "explanation": "Concise explanation of index assignments"
    }}

    Consider these examples:

    Example 1:
    Groundtruth list: ['Ice Preservation Discussions', 'Theater Show', 'Parkour Workshop']
    Predicted list: ['Theater Performance', 'Tech Hackathon', 'Ice Preservation Talks']
    {{
        "groundtruth_indexes": [0, 1, 2],
        "predicted_indexes": [1, -1, 0],
        "explanation": "Theater Performance matches Theater Show (index 1), Tech Hackathon has no match (-1), Ice Preservation Talks matches Ice Preservation Discussions (index 0)."
    }}

    Example 2:
    Groundtruth list: ['Ice Preservation Discussions', 'Theater Show', 'Parkour Workshop', 'Theater Performance']
    Predicted list: ['Theater Performance', 'Tech Hackathon', 'Ice Preservation Talks']
    {{
        "groundtruth_indexes": [0, 1, 2, 3],
        "predicted_indexes": [1, -1, 0],
        "explanation": "Theater Performance matches Theater Show (index 1, first available match), Tech Hackathon has no match (-1), Ice Preservation Talks matches Ice Preservation Discussions (index 0)."
    }}

    Now, please provide your evaluation for the given lists:
    """
    return chronological_prompt

def evaluate_chronological(groundtruth_items: List[str], predicted_items: List[str], my_model: ModelsWrapper) -> Dict[str, Any]:

    # Prepare the prompt for the judge LLM
    judge_prompt = judge_prompt_chronological_func(groundtruth_items, predicted_items)
    print(judge_prompt)
    system_prompt = "You are an expert judge evaluating the alignment between an AI-generated list and a known groundtruth list."

    # Get the judge LLM's evaluation
    judge_response = my_model.generate(user_prompt = judge_prompt, system_prompt = system_prompt, max_new_tokens = 4096)
    print(judge_response)
    
    # Parse the judge's response
    try:
        evaluation = json.loads(judge_response)
    except json.JSONDecodeError:
        # If JSON parsing fails, use regex to extract the JSON part
        json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
        if json_match:
            evaluation = json.loads(json_match.group())
        else:
            print(judge_response)
            raise ValueError("Failed to parse judge's response")
        
    return evaluation
