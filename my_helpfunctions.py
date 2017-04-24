import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def get_linearly_separable_dataset():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    return linearly_separable

def get_imbalanced_linearly_separable_dataset():
    X, y = make_classification(n_samples=1000, weights= [0.983, 0.017], n_features=2, n_redundant=0, n_informative=2,
                            random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    return linearly_separable

def get_display_meshgrid(X):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    return np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))

def get_display_boundaries(X):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    return x_min, x_max, y_min, y_max

def display_dataset(X_train, X_test, y_train, y_test, axis, xlim=None, ylim=None):
    # just plot the dataset first
    # Plot the training points
    axis.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    axis.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    if xlim is not None:
        axis.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        axis.set_ylim(ylim[0], ylim[1])
    axis.set_xticks(())
    axis.set_yticks(())

def display_decision_boundary(clf, xx, yy, ax):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
