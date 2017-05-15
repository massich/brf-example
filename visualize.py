import pandas as pd
import seaborn.apionly as sns
import matplotlib.pyplot as plt

def get_results_as_DataFrame():
    from sklearn.externals import joblib
    from os.path import join as path_join

    data_path = '/mnt/anakim/random_thoughts/brf-example/'
    scores = joblib.load(path_join(data_path,'scores.Pl'))

    df = {}
    for balance_ratio, clf, results in scores:
        df[(balance_ratio, clf)] = pd.DataFrame(results)

    df = pd.concat(df, names=['balance_ratio', 'clf'])
    # set the Cross-Validation trail id as a column not part of the dataframe's index
    df = df.reset_index(level=2).rename(columns={'level_2':'CV_trail'})
    return df

d = get_results_as_DataFrame()

metrics = list(d.columns[1:])
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for ax, metric in zip((ax1, ax2, ax3, ax4), metrics):
    mean = d.groupby(['balance_ratio', 'clf'])[metric].mean().unstack()
    std = d.groupby(['balance_ratio', 'clf'])[metric].std().unstack()
    mean.plot(ax=ax, title=metric)
    for col in std.columns:
        ax.fill_between(std.index, mean[col] + std[col], mean[col] - std[col], alpha=0.3)
