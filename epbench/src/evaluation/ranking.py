import pandas as pd
import numpy as np
from epbench.src.results.average_groups import extract_groups
# df below is obtained from from get_precomputed_results

## A. Helper functions

# Helper function for the `simple_recall_score`
def convert_uncertainty_format(df):
    # Create a copy to avoid modifying the original dataframe
    result = df.copy()
    
    # Function to convert a single cell
    def extract_value(cell):
        if isinstance(cell, str) and '±' in cell:
            try:
                # Split on '±' and take the first part
                value = float(cell.split('±')[0])
                return value
            except ValueError:
                return cell
        return cell
    
    # Apply the function to each column using map
    for column in result.columns:
        result[column] = result[column].map(extract_value)
    
    return result

# Helper functions for the `chronological_score`
def get_short_name_from_model_name(answering_model_name, answering_kind, answering_embedding_chunk):
    if 'gpt-4o-mini' in answering_model_name:
        model_name = 'gpt-4o-mini'
    elif 'gpt-4o' in answering_model_name:
        model_name = 'gpt-4o'
    elif 'claude-3-5-sonnet' in answering_model_name:
        model_name = 'cl-3.5-sonnet'
    elif 'claude-3-haiku' in answering_model_name:
        model_name = 'cl-3-haiku'
    elif 'o1-mini' in answering_model_name:
        model_name = 'o1-mini'
    elif 'o1-preview' in answering_model_name:
        model_name = 'o1-preview'
    elif 'o1' in answering_model_name:
        model_name = 'o1'
    elif 'o3-mini' in answering_model_name:
        model_name = 'o3-mini'
    elif 'llama-3.1-405b-instruct' in answering_model_name:
        model_name = 'llama-3.1'
    elif 'gemini-2.0-flash-thinking' in answering_model_name:
        model_name = 'gemini-2-flash-thinking'
    elif 'gemini-2.0-flash' in answering_model_name:
        model_name = 'gemini-2-flash'
    elif 'gemini-2.0-pro' in answering_model_name:
        model_name = 'gemini-2-pro'
    elif 'deepseek-reasoner' in answering_model_name:
        model_name = 'deepseek-reasoner'
    elif 'deepseek-chat' in answering_model_name:
        model_name = 'deepseek-chat'
    else:
        model_name = answering_model_name
        # raise ValueError('unknown model')
    
    if answering_kind == 'prompting':
        output = model_name
    elif answering_kind == 'rag':
        if answering_embedding_chunk == 'chapter':
            output = f"{model_name} (rag, {answering_embedding_chunk[0]})"
        else: 
            output = f"{model_name} (rag)"
    elif answering_kind == 'ftuning':
        output = f"{model_name} (ftuning)"

    return output

def get_short_name(i, df):
    res = df.iloc[i][['answering_kind', 'answering_model_name', 'answering_embedding_chunk']]
    model_name = get_short_name_from_model_name(res['answering_model_name'], res['answering_kind'], res['answering_embedding_chunk'])
    return model_name

def convert_percentages(df):
    result = df.copy()
    for column in result.columns:
        result[column] = result[column].map(lambda x: float(x.strip('%'))/100 if isinstance(x, str) and '%' in x else x)
    return result

def multiply_rows(df):
    result = df.copy()
    # Keep first row as is
    first_row = result.iloc[0]
    # Multiply elements of 2nd and 3rd rows, with special handling for 0 * NaN = 0
    row2, row3 = result.iloc[1], result.iloc[2]
    multiplied_row = pd.Series([0 if (row2[col] == 0 or row3[col] == 0) else row2[col] * row3[col] 
                               for col in df.columns], index=df.columns)
    multiplied_row = np.maximum(0, multiplied_row) # do not further impact the summarized chronological score if the kendall tau is negative

    # Create new DataFrame with 2 rows
    return pd.DataFrame([first_row, multiplied_row])

## B. Summarizing tables (paper level = detailed results)

def get_simple_results(df, nb_events = 200):
    # (df: obtained from get_precomputed_results)
    relative_to = ['get', 'bins_items_correct_answer'] # select the grouped elements as a list among:
    # 'get': type of question, among 'all' (simple recall questions), 'latest' (latest state questions), or 'chronological' (chronological questions)
    # 'bins_items_correct_answer': number of events for this question, binned into {0}, {1}, {2}, {3,4,5}, {6+} chapters
    # 'cue': type of cue for this question, e.g. (*,*,*,c)
    # 'retrieval_type': type of trace for this question, e.g. 'Spaces'
    df_results_simple = extract_groups(df, nb_events, relative_to) # group the results according to `relative_to`

    # Further filtering, e.g. for selecting only the simple recall questions:
    df_results_simple = df_results_simple[df_results_simple['get'] == 'all'].drop('get', axis = 1)
    return df_results_simple

def get_kendall_tau_results(df, nb_events = 200):
    # 1. adding the `All` and the `Kendall τ` results (in total, there are 39 questions involving temporal aspects with >= 2 linked events)
    # (df: obtained from get_precomputed_results)
    # Note: it is possible for the Kendall τ to be negative. We report the actual value in this table, but replace it with max(0,τ)
    # in the summarized chronological score.
    kendall_tau_results = pd.concat([x.kendall_summaries_for_this_experiment for x in df['evaluation_object']]).reset_index(drop=True)
    kendall_tau_results = pd.concat([df, kendall_tau_results], axis = 1)
    kendall_tau_results['%_exact_match_set_gt_with_pred2'] = [int(x[:-1]) for x in kendall_tau_results['%_exact_match_set_gt_with_pred']]
    kendall_tau_results['All'] = [f"{round(u/d * 100, 2)}%" for u,d in zip(kendall_tau_results['#exact_match_set_gt_with_pred'], kendall_tau_results['#gt_with_len_2+'])]
    kendall_tau_results['Kendall τ'] = [float(x.split('±')[0]) for x in kendall_tau_results['tau_exact_match_set_gt_with_pred']]
    kendall_tau_results['name'] = [get_short_name(i, kendall_tau_results) for i in range(len(kendall_tau_results))]
    kendall_tau_results = kendall_tau_results[kendall_tau_results['book_nb_events'] == nb_events]
    kendall_tau_results = kendall_tau_results.drop('book_nb_events', axis = 1).reset_index(drop = True)
    kendall_tau_results = kendall_tau_results.sort_values(['%_exact_match_set_gt_with_pred2', 'Kendall τ'], ascending = False)
    kendall_tau_results = kendall_tau_results[['name', 'All', 'Kendall τ']]
    kendall_tau_results = kendall_tau_results.set_index('name').transpose()

    # 2. adding the `Latest` results, by looking at the correct result for bins with >= 2 linked events
    from epbench.src.results.average_groups import extract_groups
    relative_to = ['get', 'bins_items_correct_answer']
    df_results = extract_groups(df, nb_events, relative_to, 'f1_score_lenient')
    df_results = df_results[df_results['get'] == 'latest']
    df_results = df_results[df_results['bins_items_correct_answer'].isin(['2', '3-5', '6+'])]
    # extract the average performance float element
    for col in df_results.columns:
        if col not in ['get', 'bins_items_correct_answer', 'count']:
            df_results[col] = df_results[col].str.extract(r'([\d.]+)').astype(float)
    # extract the percentage by computing sum(count*average) over all bins with >= 2 correct answers, for each model
    result = {}
    for col in df_results.columns:
        if col not in ['get', 'bins_items_correct_answer', 'count']:
            answering_kind, answering_model_name, answering_embedding_chunk = col
            current_short_name = get_short_name_from_model_name(answering_model_name, answering_kind, answering_embedding_chunk)
            result[current_short_name] = f"{round(100*(df_results[col] * df_results['count']).sum()/df_results['count'].sum(), 2)}%"
    new_row = pd.Series({col: result[col] if col in result else None for col in kendall_tau_results.columns}, name='Latest')
    # finally add those `Latest` results as a third row
    kendall_tau_results = pd.concat([kendall_tau_results, new_row.to_frame().T])
    kendall_tau_results = kendall_tau_results.loc[['Latest', 'All', 'Kendall τ']]

    return kendall_tau_results

## C. Summarizing scores (github level = summarized results)

def simple_recall_score(df_results_simple):
    # Simple Recall Score: Measures the model's ability to accurately recall episodic events
    # (df_results_simple: obtained from get_simple_results)
    return convert_uncertainty_format(df_results_simple).drop(['bins_items_correct_answer', 'count'], axis = 1).mean().round(3).sort_values(ascending=False)

def chronological_awareness(kendall_tau_results):
    # Chronological Awareness Score: Assesses how well the model tracks entity states and temporal sequences
    return multiply_rows(convert_percentages(kendall_tau_results)).mean().round(3).sort_values(ascending=False)

# Final table
def get_final_scores_table(df, nb_events):
    df_results_simple = get_simple_results(df, nb_events)
    simple_recall_score_table = simple_recall_score(df_results_simple)
    simple_recall_score_table.index = [get_short_name_from_model_name(x2,x1,x3) for x1,x2,x3 in simple_recall_score_table.index]
    kendall_tau_results = get_kendall_tau_results(df, nb_events)
    chronological_awareness_table = chronological_awareness(kendall_tau_results)

    added_text = ""
    if nb_events == 20:
        added_text = " (short book)"
    elif nb_events == 2000:
        added_text = " (million token book)"

    final_table = pd.concat([simple_recall_score_table, chronological_awareness_table], axis=1)
    final_table.columns = [f'🎯 Simple Recall{added_text}', f'⏱️ Chronological Awareness{added_text}']
    return final_table
