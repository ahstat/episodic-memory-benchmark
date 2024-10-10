# for the ECDF
from scipy import stats
# matplotlib
import matplotlib
import matplotlib.pyplot as plt

def strategy_rcParams(fontfamily = 'serif', figurefigsize = (4,2)):
    strategy_rcParams = {
        'font.family': fontfamily,
        'text.usetex': True,
        'figure.figsize': figurefigsize
    }
    return strategy_rcParams

# Define a unified strategy mapping
def strategy_map_func():
    strategy_map = {
        't': {'label': 't', 'color': 'C2', 'marker': '', 'linestyle': '-'}, # C2: #2ca02c (a shade of green) # t (date)
        's': {'label': 's', 'color': 'C0', 'marker': '', 'linestyle': '--'}, # C0: #1f77b4 (a shade of blue) # s (location)
        'e': {'label': 'e', 'color': 'C3', 'marker': '', 'linestyle': ':'}, # C3: #d62728 (a shade of red) # e (entity)
        'c': {'label': 'c', 'color': 'C8', 'marker': '', 'linestyle': '-.'} # C8: #bcbd22 (a shade of yellow-green) # c (content details)
    }
    return strategy_map

# 'marker': 's', '^', 'D', 'P', '*', 'v', 'X', 
# C1: #ff7f0e (a shade of orange)
# C4: #9467bd (a shade of purple)
# C5: #8c564b (a shade of brown)
# C6: #e377c2 (a shade of pink)
# C7: #7f7f7f (a shade of gray)
# C9: #17becf (a shade of cyan)

def plotting_ecdf(pos_t, pos_s, pos_e, pos_c, filepath, xtitle = 'Relative position of the cue within the chapter', strategy_map = strategy_map_func()):
    res_t = stats.ecdf(pos_t)
    res_s = stats.ecdf(pos_s)
    res_e = stats.ecdf(pos_e)
    res_c = stats.ecdf(pos_c)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update(strategy_rcParams())
    ax = plt.subplot()
    res_t.cdf.plot(ax, linestyle=strategy_map['t']['linestyle'], color=strategy_map['t']['color'], marker=strategy_map['t']['marker'], label=strategy_map['t']['label'])
    res_s.cdf.plot(ax, linestyle=strategy_map['s']['linestyle'], color=strategy_map['s']['color'], marker=strategy_map['s']['marker'], label=strategy_map['s']['label'])
    res_e.cdf.plot(ax, linestyle=strategy_map['e']['linestyle'], color=strategy_map['e']['color'], marker=strategy_map['e']['marker'], label=strategy_map['e']['label'])
    res_c.cdf.plot(ax, linestyle=strategy_map['c']['linestyle'], color=strategy_map['c']['color'], marker=strategy_map['c']['marker'], label=strategy_map['c']['label'])
    ax.set_xlabel(xtitle)
    ax.set_ylabel('ECDF')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.show()
