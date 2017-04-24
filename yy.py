from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from my_helpfunctions import get_linearly_separable_dataset, get_display_meshgrid, get_display_boundaries, display_dataset, display_decision_boundary,get_imbalanced_linearly_separable_dataset


names = ["Random Forest",
         "Random Forest (balanced)",
         "Random Forest (balanced_subsample)",
         "Balanced Random Forest",
         "Extra Trees", ]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight='balanced'),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight='balanced_subsample'),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, balanced=True),
    ExtraTreesClassifier(n_estimators=10),
    ]


datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            get_linearly_separable_dataset(),
            get_imbalanced_linearly_separable_dataset()
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max, y_min, y_max = get_display_boundaries(X)
    xx, yy = get_display_meshgrid(X)  # to remove

    # just plot the dataset first
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    display_dataset(X_train, X_test, y_train, y_test, ax, [x_min, x_max], [y_min, y_max])
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title(name)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)


        display_decision_boundary(clf, xx, yy, ax)
        display_dataset(X_train, X_test, y_train, y_test, ax, [x_min, x_max], [y_min, y_max])

        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
