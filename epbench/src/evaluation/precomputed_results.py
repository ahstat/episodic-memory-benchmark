from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
from epbench.src.evaluation.generator_answers_2_rag import get_top_n
import pandas as pd

def get_precomputed_results(experiments, 
                            env_file, 
                            data_folder,
                            all_benchmarks = {'benchmark_claude_default_20': None,
                                              'benchmark_claude_default_200': None,
                                              'benchmark_claude_default_2000': None,
                                              'benchmark_claude_default_20_ordered': None,
                                              'benchmark_claude_default_200_ordered': None,
                                              'benchmark_gpt_default_20': None,
                                              'benchmark_gpt_default_200': None,
                                              'benchmark_claude_news_20': None,
                                              'benchmark_claude_news_200': None,
                                              'benchmark_claude_scifi_20': None,
                                              'benchmark_claude_scifi_200': None},
                            evaluation_policy = 'remove_duplicates'):
    df_list = []

    for i in range(len(experiments)):
        df_list.append(pd.DataFrame(experiments[i], index=[0]))
    # concatenate all DataFrames in the list
    df = pd.concat(df_list, ignore_index=True)
    df['evaluation_object'] = None

    for i in range(len(df)):
        df_cur = df.iloc[i]

        if df_cur['book_model_name'] == 'claude-3-5-sonnet-20240620':
            if df_cur['book_nb_events'] == 20:
                if 'book' in df_cur.index:
                    if df_cur['book'] == 'default':
                        my_benchmark = all_benchmarks['benchmark_claude_default_20']
                    elif df_cur['book'] == 'news':
                        my_benchmark = all_benchmarks['benchmark_claude_news_20']
                    elif df_cur['book'] == 'scifi':
                        my_benchmark = all_benchmarks['benchmark_claude_scifi_20']
                    else:
                        raise ValueError('Unknown book')
                elif ('ordered' in df_cur.index) and (df_cur['ordered']):
                    my_benchmark = all_benchmarks['benchmark_claude_default_20_ordered']
                else:
                    my_benchmark = all_benchmarks['benchmark_claude_default_20']
            elif df_cur['book_nb_events'] == 200:
                if 'book' in df_cur.index:
                    if df_cur['book'] == 'default':
                        my_benchmark = all_benchmarks['benchmark_claude_default_200']
                    elif df_cur['book'] == 'news':
                        my_benchmark = all_benchmarks['benchmark_claude_news_200']
                    elif df_cur['book'] == 'scifi':
                        my_benchmark = all_benchmarks['benchmark_claude_scifi_200']
                    else:
                        raise ValueError('Unknown book')
                elif ('ordered' in df_cur.index) and (df_cur['ordered']):
                    my_benchmark = all_benchmarks['benchmark_claude_default_200_ordered']
                else:
                    my_benchmark = all_benchmarks['benchmark_claude_default_200']
            elif df_cur['book_nb_events'] == 2000:
                my_benchmark = all_benchmarks['benchmark_claude_default_2000']
            else:
                raise ValueError('For `claude-3-5-sonnet-20240620`, only done with 20, 200, and 2000 target events')
        elif df_cur['book_model_name'] == 'gpt-4o-2024-05-13':
            if df_cur['book_nb_events'] == 20:
                my_benchmark = all_benchmarks['benchmark_gpt_default_20']
            elif df_cur['book_nb_events'] == 200:
                my_benchmark = all_benchmarks['benchmark_gpt_default_200']
            else:
                ValueError('For `gpt-4o-2024-05-13`, only done with 20 and 200 target events')
        else:
            raise ValueError('Only books generated with `claude-3-5-sonnet-20240620` and `gpt-4o-2024-05-13`')

        if df_cur['answering_kind'] == 'prompting':
            answering_parameters = {'kind': df_cur['answering_kind'],
                                    'model_name': df_cur['answering_model_name'],
                                    'max_new_tokens': 4096,
                                    'sleeping_time': 1,
                                    'policy': evaluation_policy}           
        elif df_cur['answering_kind'] == 'rag':
            answering_parameters = {'kind': df_cur['answering_kind'], 
                                    'model_name': df_cur['answering_model_name'], 
                                    'max_new_tokens': 4096, 
                                    'sleeping_time': 1, 
                                    'embedding_chunk': df_cur['answering_embedding_chunk'], 
                                    'embedding_model': "text-embedding-3-small", 
                                    'embedding_batch_size': 2048, 
                                    'top_n': get_top_n(df_cur['answering_embedding_chunk'], my_benchmark), 
                                    'policy': evaluation_policy}
        elif df_cur['answering_kind'] == 'ftuning':
            answering_parameters = {'kind': df_cur['answering_kind'], 
                                    'model_name': df_cur['answering_model_name'], 
                                    'max_new_tokens': 4096, 
                                    'sleeping_time': 0, 
                                    'ftuning_input_data_policy': 'single', 
                                    'ftuning_need_upload': False, 
                                    'ftuning_need_actual_tune': False, 
                                    'batch_size': 'auto', 
                                    'learning_rate_multiplier': 'auto', 
                                    'n_epochs': 10,
                                    'policy': evaluation_policy}
            # ad-hoc
            if df_cur['book_nb_events'] == 20:
                if df_cur['answering_model_name'] == 'gpt-4o-mini-2024-07-18':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-mini-2024-07-18:personal::AAzm9XtH'
                elif df_cur['answering_model_name'] == 'gpt-4o-2024-08-06':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-2024-08-06:personal::AB02Cbei'
                else:
                    raise ValueError('only done for gpt4o and gpt4o-mini')
            elif df_cur['book_nb_events'] == 200:
                if df_cur['answering_model_name'] == 'gpt-4o-mini-2024-07-18':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-mini-2024-07-18:personal::AB0B6H4o'
                elif df_cur['answering_model_name'] == 'gpt-4o-2024-08-06':
                    answering_parameters['fine_tuned_model_name'] = 'ft:gpt-4o-2024-08-06:personal::DISCARDED' # DISCARDED (~400 dollars)
                else:
                    raise ValueError('only done for gpt4o and gpt4o-mini')
        str_print = f"Document with {my_benchmark.nb_tokens()} tokens, answer with {df_cur['answering_kind']} using with {df_cur['answering_model_name']}"
        if df_cur['answering_kind'] == 'rag':
            str_print = f"{str_print} ({df_cur['answering_embedding_chunk']} chunks)"
        print(str_print)
        my_evaluation = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file)
        df.loc[i, 'evaluation_object'] = my_evaluation
    return df
