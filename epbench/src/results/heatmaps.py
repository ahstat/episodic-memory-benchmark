import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from epbench.src.results.average_groups import extract_groups
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from warnings import catch_warnings, filterwarnings

def get_short_name2(tuple_input):
    if tuple_input == "get":
        return "get"
    if tuple_input == "cue":
        return "cue"
    if len(tuple_input) != 3:
        return tuple_input
    model_name = tuple_input[1]
    if 'gpt-4o-mini' in tuple_input[1]:
        model_name = 'gpt-4o-mini'
    elif 'gpt-4o' in tuple_input[1]:
        model_name = 'gpt-4o'
    elif 'claude-3-5-sonnet' in tuple_input[1]:
        model_name = 'cl-3.5-sonnet'
    elif 'claude-3-haiku' in tuple_input[1]:
        model_name = 'cl-3-haiku'
    elif 'o1-mini' in tuple_input[1]:
        model_name = 'o1-mini'
    elif 'o1-preview' in tuple_input[1]:
        model_name = 'o1-preview'
    elif 'llama' in tuple_input[1]:
        model_name = 'llama-3.1'
    else:
        print("uknown model in get_short_name2")
        model_name = tuple_input[1]
        
    if tuple_input[0] == 'prompting':
        output = model_name
    elif tuple_input[0] == 'rag':
        if tuple_input[2] == 'chapter':
            output = f"{model_name}\n(rag, {tuple_input[2]})"
        else: 
            output = f"{model_name}\n(rag)"
    elif tuple_input[0] == 'ftuning':
        output = f"{model_name}\n(ftuning)"
    return output

def plot_clust(df, nb_events, relative_to, figsize=(16, 20), only_bins = None):
    with catch_warnings():
        filterwarnings('ignore')
        
        df_results = extract_groups(df, nb_events, relative_to)

        df_results = df_results[df_results['get'] == 'all']
        if only_bins is not None:
            df_results = df_results[df_results['bins_items_correct_answer'].isin(only_bins)]

        df_results = df_results[df_results['cue'].isin(['(t, *, *, *)', '(*, s, *, *)', '(*, *, ent, *)', '(*, *, *, c)'])]

        def clean_string(s):
            return ''.join(c for c in s if c not in '()*, ')

        # Apply the cleaning function to the 'cue' column
        df_results['cue'] = df_results['cue'].apply(clean_string)


        data = df_results.loc[:, df_results.columns != 'count']


        data['bins_items_correct_answer'] = pd.Categorical(data['bins_items_correct_answer'], ['0', '1', '2', '3-5', '6+'])

        data.columns = [get_short_name2(x) for x in data.columns]
        
        print(data.columns)

        data = data[['get', 'bins_items_correct_answer', 'cue', 
                    'gpt-4o', 'cl-3.5-sonnet\n(rag)', 'gpt-4o\n(rag)',
                    'gpt-4o-mini\n(rag)', 'cl-3-haiku\n(rag)', 'llama-3.1',
                    'cl-3-haiku', 'gpt-4o-mini',
                    'cl-3.5-sonnet', 'gpt-4o-mini\n(ftuning)', 'o1-mini']]

        # data.index = data.apply(lambda row: (row[relative_to]), axis=1)
        data.index = data['cue']
        data = data.drop('cue', axis = 1)
        for col in reversed(relative_to):
            data = data.sort_values(by=col, ascending=True)
            data = data.loc[:, data.columns != col]

        plt.figure(figsize=figsize)

        my_colors = ['darkred', 'red', 'orange', 'green', 'white']
        my_cmap = ListedColormap(my_colors)
        bounds = [-1, 0.49999, 0.69, 0.80, 0.90, 1]
        my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

        def remove_second_part(x):
            return round(float(x.split("±")[0]), 1) # rounding one digit!
        data = data.applymap(remove_second_part)

        legend = False
        if only_bins[0] == '2':
            legend = True

        #colors = [(0.8, 0.1, 0.1),         # Red at 0.2
        #          (0.6, 0.1, 0.1),         # Reddish until 0.5
        #          (1, 172/255, 28/255),    # Light gray as transition rgb(255, 172, 28)
        #          (0.1, 0.6, 0.1),         # Green start at 0.7
        #          (0.1, 0.8, 0.1)]         # Darker green at 1.0
        
        colors = [
            (68/255, 1/255, 84/255),
            (72/255, 36/255, 117/255),
            (65/255, 68/255, 135/255),
            (53/255, 95/255, 141/255),
            (42/255, 120/255, 142/255),
            (33/255, 145/255, 140/255),
            (34/255, 168/255, 132/255),
            (68/255, 191/255, 112/255),
            (122/255, 209/255, 81/255),
            (189/255, 223/255, 38/255),
            (253/255, 231/255, 37/255)]
        
        positions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Position of each color on the colormap (0 to 1)
        #positions = [0, 0.5, 0.6, 0.7, 1.0]

        # Create the custom colormap
        custom_cmap = LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)))

        # Your data plotting
        # Assuming 'data' is your input data
        #my_plot = sns.heatmap(data, 
        #                    annot=True, 
        #                    fmt='.1f', 
        #                    cmap=custom_cmap, 
        #                    cbar=legend)

        #my_plot = sns.heatmap(data, cmap=my_cmap, yticklabels=True, xticklabels=True, cbar_kws={'label': 'Value'}, norm=my_norm)
        my_plot = sns.heatmap(data, annot=True, fmt='.1f', cmap=custom_cmap, cbar=legend, vmin=0, vmax=1) # 'RdYlGn'
        fig = my_plot.get_figure()
        plt.xticks(rotation=90) 
        #plt.title(f'Bin {only_bins[0]}', fontsize = 12) 

        if only_bins[0] != '0':
            #plt.xticks([])
            plt.yticks([])
            plt.ylabel(None)
            #plt.gca().get_legend().remove()
                
        return fig