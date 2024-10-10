import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def extract_groups(df, nb_events, relative_to, metric = 'f1_score_lenient'):

    df_sliced = df[(df['book_nb_events'] == nb_events) & (df['book_model_name'] == 'claude-3-5-sonnet-20240620')]

    #print(f"(using the book with {df_sliced.iloc[0]['book_nb_events']} events)")

    # template
    i = 0
    df_res_0 = df_sliced.iloc[i]['evaluation_object'].get_pretty_summary_relative_to(relative_to, metric)
    df_results = df_res_0.iloc[:, :-1] # take all but last column

    # fill
    for i in range(len(df_sliced)):
        df_res_i = df_sliced.iloc[i]['evaluation_object'].get_pretty_summary_relative_to(relative_to, metric)
        df_results[(df_sliced.iloc[i]['answering_kind'], 
                    df_sliced.iloc[i]['answering_model_name'],
                    df_sliced.iloc[i]['answering_embedding_chunk'])] = [x for x in df_res_i.iloc[:, -1]] # average # [float(x.split('±')[0]) for x in df_res_i.iloc[:, -1]] # average

    # remove the nan
    df_results_tmp = df_results.copy()
    for col in relative_to + ['count']:
        df_results_tmp = df_results_tmp.loc[:, df_results_tmp.columns != col]
    nan_rows = [[k for i, x in enumerate(df_results_tmp.iloc[k]) if np.isnan(float(x.split('±')[0]))==True ] for k in range(len(df_results))]
    issue_rows = list(set([item for sublist in nan_rows for item in sublist]))
    df_results = df_results.drop(issue_rows)

    return df_results
