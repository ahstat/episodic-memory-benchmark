import pandas as pd
def length_based_on_nb_chapters(nb_chapters):
    if nb_chapters < 100:
        return 'short'
    elif nb_chapters < 1000:
        return 'long'
    else:
        return 'very long'

def get_summary_dict(my_benchmark):
    # name
    if 'default' in my_benchmark.prompt_parameters['name_universe']:
        name = 'default'
        if (my_benchmark.nb_tokens() == 102870) and (my_benchmark.book_parameters['indexing'] == 'default'):
            name = 'default (Synaptic Echoes)'
    else:
        name = my_benchmark.prompt_parameters['name_universe']

    # length
    length = length_based_on_nb_chapters(my_benchmark.nb_chapters())

    # generation with
    generation = my_benchmark.model_parameters['model_name']

    # variation
    if my_benchmark.book_parameters['indexing'] == 'ordered':
        variation = 'ordered'
    else:
        variation = '/'

    # chapters
    chapters = my_benchmark.nb_chapters()

    # tokens
    tokens = my_benchmark.nb_tokens()

    return {'name': name,  'length': length, 'generation': generation, 'variation': variation, 'chapters': chapters, 'tokens': tokens}

def get_summary_table(list_of_benchmarks):
    df_list = []
    for my_benchmark in list_of_benchmarks:
        dict_summary = get_summary_dict(my_benchmark)
        df_list.append(pd.DataFrame(dict_summary, index=[0]))
    # concatenate all DataFrames in the list
    summary_table = pd.concat(df_list, ignore_index=True)

    for i, my_benchmark in enumerate(list_of_benchmarks):
        summary_table.loc[i, 'benchmark_object'] = my_benchmark
    return summary_table