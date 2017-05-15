from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
from imblearn.datasets import make_imbalance
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.model_selection import StratifiedKFold

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
            for r in get_imbalance_ratios(num=2)]

# we need stratifiedKFold to preserve class distribution within each fold
skf = StratifiedKFold(n_splits=5)

from sklearn.base import clone as clone_estimator
from sklearn.externals.joblib import Memory
memory = Memory(cachedir="./", verbose=0)

@memory.cache
def compute_cross_val_predict(X, y, clf):
    # clf should be a clone
    print("..computing..", end='', flush=True)
    return cross_val_predict(X=X, y=y, estimator=clf, cv=skf, n_jobs=N_JOBS)

for dataset_ratio, dataset in datasets:
    X,y = dataset
    for name, clf in classifiers:
        print("compute: {0:.2f} imbalance, {1} ...".format(dataset_ratio, name), end='', flush=True)
        ypred = compute_cross_val_predict(X, y, clone_estimator(clf))
        print("done")


