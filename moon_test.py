from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
from imblearn.datasets import make_imbalance
from sklearn.datasets import make_moons
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42

classifiers = [
    ("Random Forest",
     RandomForestClassifier()),

    ("Random Forest (balanced)",
     RandomForestClassifier(class_weight='balanced')),

    ("Random Forest (balanced_subsample)",
     RandomForestClassifier(class_weight='balanced_subsample')),

    ("Balanced Random Forest",
     RandomForestClassifier(balanced=True)),

    ("Extra Trees",
     ExtraTreesClassifier())
]

original_dataset = make_moons(noise=0.3, random_state=0, n_samples=int(1e6))


def make_imbalance_dataset(d, ratio=.01):
    return make_imbalance(d[0], d[1], ratio=ratio, random_state=42)


datasets = [ make_imbalance_dataset(original_dataset, ratio=0.01) ]

# use a full grid over all parameters
param_grid = {"max_depth": [5, None],
              "criterion": ["gini", "entropy"],
              "n_estimators": np.linspace(10,150, num=3, dtype=int)}

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

for dataset in datasets:
    X,y = dataset
    for name, clf in classifiers:
        grid_search = GridSearchCV(clf, param_grid=param_grid)
        grid_search.fit(X, y)
        print("-----{0:s}------".format(name))
