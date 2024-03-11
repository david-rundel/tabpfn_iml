import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import os

cmap_original = plt.get_cmap('magma')
cmap_inverted = ListedColormap(cmap_original.colors[::-1])

def plot_barplot(df, x, y, hue, hue_order, title, palette):
    plt.figure()
    sns.set_theme(style="ticks", font='sans-serif', font_scale=1.5)
    sorted_df = df.sort_values(by=y, ascending=True)
    sns.barplot(data=sorted_df,
                    x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, saturation=1.0)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    return plt

def get_latest_csv_file(directory:str,
                        prefix:str,
                        criterion:str=None):
    files = os.listdir(directory)
    if criterion is not None:
        filtered_files = [file for file in files if file.startswith(prefix) and criterion not in file]
    else:
        filtered_files = [file for file in files if file.startswith(prefix)]
    sorted_files = sorted(filtered_files, reverse=True)
    latest_file = sorted_files[0]
    return pd.read_csv(f'{directory}/{latest_file}')

df_pdp = get_latest_csv_file('experiments/ice_pd/results', 'benchmark_results_pdp_', 'openml')

barplot_pdp = plot_barplot(df=df_pdp,
                       x="Number of unique Feature Values",
                       y="Runtime in Seconds",
                       hue="Implementation",
                       hue_order=["tabpfn_iml PDP", "scikit-learn PDP"],
                       title="PDP Runtime Comparison",
                       palette={"tabpfn_iml PDP": cmap_inverted.colors[100], "scikit-learn PDP": cmap_inverted.colors[230]})


barplot_pdp.savefig('experiments/ice_pd/pdp_runtime_comparison_synth.pdf')


df_ale = get_latest_csv_file('experiments/ice_pd/results', 'benchmark_results_ale_', 'openml')

barplot_ale = plot_barplot(df=df_ale,
                       x="Number of unique Feature Values",
                       y="Runtime in Seconds",
                       hue="Implementation",
                       hue_order=["tabpfn_iml ALE", "PyALE ALE"],
                       title="PDP Runtime Comparison",
                       palette={"tabpfn_iml ALE": cmap_inverted.colors[100], "PyALE ALE": cmap_inverted.colors[230]})

barplot_ale.savefig('experiments/ice_pd/ale_runtime_comparison_synth.pdf')


df_pdp_openml = get_latest_csv_file('experiments/ice_pd/results', 'benchmark_results_pdp_openml_')

barplot_pdp_openml = plot_barplot(df=df_pdp_openml,
                          x="Number of unique Feature Values",
                          y="Runtime in Seconds",
                          hue="Implementation",
                          hue_order=["tabpfn_iml PDP", "scikit-learn PDP"],
                          title="PDP Runtime Comparison",
                          palette={"tabpfn_iml PDP": cmap_inverted.colors[100], "scikit-learn PDP": cmap_inverted.colors[230]})

barplot_pdp_openml.savefig('experiments/ice_pd/pdp_runtime_comparison_openml.pdf')


df_ale_openml = get_latest_csv_file('experiments/ice_pd/results', 'benchmark_results_ale_openml_')

barplot_ale_openml = plot_barplot(df=df_ale_openml,
                            x="Number of unique Feature Values",
                            y="Runtime in Seconds",
                            hue="Implementation",
                            hue_order=["tabpfn_iml ALE", "PyALE ALE"],
                            title="ALE Runtime Comparison",
                            palette={"tabpfn_iml ALE": cmap_inverted.colors[100], "PyALE ALE": cmap_inverted.colors[230]})

barplot_ale_openml.savefig('experiments/ice_pd/ale_runtime_comparison_openml.pdf')

