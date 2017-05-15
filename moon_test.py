from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
from imblearn.datasets import make_imbalance
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

RANDOM_STATE = 42
N_JOBS = 40

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
    return make_imbalance(d[0], d[1], ratio=ratio, random_state=RANDOM_STATE)


def get_imbalance_ratios(num=6, min_ratio=0.01, max_ratio=0.5):
    return np.logspace( np.log10(min_ratio), np.log10(max_ratio), num=num)

datasets = [(r, make_imbalance_dataset(original_dataset, ratio=r))
            for r in get_imbalance_ratios(num=10)]

# we need stratifiedKFold to preserve class distribution within each fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)


# make scorer to evaluate the results
#  - http://scikit-learn.org/stable/modules/model_evaluation.html
#  - https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/metrics/classification.py
#
# Use custom scoring to track multiple metrics
from sklearn.metrics import precision_score
from sklearn.metrics.scorer import make_scorer

# The scorers can be a std scorer referenced by its name or one wrapped
# by sklearn.metrics.scorer.make_scorer
scoring = {'AUC Score': 'roc_auc', 'Precision': make_scorer(precision_score),
           'recall': 'recall', 'F1 Score': 'f1'}


scores = list()
for dataset_ratio, dataset in datasets:
    X,y = dataset
    for name, clf in classifiers:
        print("compute: {0:.2f} imbalance, {1} ...".format(dataset_ratio, name), end='', flush=True)
        this_score = cross_val_score(clf, X, y, cv=skf, n_jobs=N_JOBS, scoring=scoring)
        scores.append((dataset_ratio, name, this_score))
        print("done")


from sklearn.externals import joblib
joblib.dump(scores, "scores.pkl")

print('done')
