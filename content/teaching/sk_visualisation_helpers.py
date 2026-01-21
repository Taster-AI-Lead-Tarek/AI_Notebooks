"""
Helpers to visualise decision trees

Written by Dustin Mason
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs
import numpy as np
import ast


import matplotlib as mpl

# plt.style.use('dark_background')

def visualize_tree(estimator, X, y, boundaries=True,
                   xlim=None, ylim=None, ax=None, cmap=None,
                   cb = False,
                   X_test = None, y_test = None):

    ax = ax or plt.gca()
    
    # Plot the training points

    if cmap is None:
        cmap = mpl.cm.viridis

    if X_test is not None and y_test is not None:
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10, alpha=0.4, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.4, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    
    bounds = range(0, max(y)+2)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N,)# extend='both')

    if cb:
        cb = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
            ax=ax,
            ticks=np.arange(0, len(set(y)), 1)+0.5)
        cb.ax.set_yticklabels(np.arange(len(set(y))))
        cb.ax.tick_params(size=0)
    ax.axis('tight')
    # ax.axis('off')
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()
    
    # fit the estimator
    estimator.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    n_classes = len(np.unique(y))
    Z = Z.reshape(xx.shape)
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap)

    ax.set(xlim=xlim, ylim=ylim)
    
    # Plot the decision boundaries
    def plot_boundaries(i, xlim, ylim):
        if i >= 0:
            tree = estimator.tree_
        
            if tree.feature[i] == 0:
                ax.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k', zorder=2)
                plot_boundaries(tree.children_left[i],
                                [xlim[0], tree.threshold[i]], ylim)
                plot_boundaries(tree.children_right[i],
                                [tree.threshold[i], xlim[1]], ylim)
        
            elif tree.feature[i] == 1:
                ax.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k', zorder=2)
                plot_boundaries(tree.children_left[i], xlim,
                                [ylim[0], tree.threshold[i]])
                plot_boundaries(tree.children_right[i], xlim,
                                [tree.threshold[i], ylim[1]])
            
    if boundaries:
        plot_boundaries(0, xlim, ylim)
    
class decision_animation:
    def __init__(self, n_axes = 1, max_depth=1, **data_gen_func_args, ):
        self.func = make_blobs
        self.func_args = data_gen_func_args
        self.cmap='viridis'
        self.classifier = DecisionTreeClassifier

        if n_axes > 1:
            assert n_axes % 2 == 0, 'n_axes must be even'
            nrows = int(n_axes/2)
            ncols = int(n_axes/2)
        else:
            nrows = 1
            ncols = 1
    
        self.fig, self.axs = plt.subplots(
            nrows = nrows, 
            ncols = ncols, 
            figsize = (16, 9), 
            gridspec_kw=dict(wspace=0.05, hspace=0.05)
            )

        self.animation = FuncAnimation(
            self.fig, 
            self.run_animation, 
            frames = max_depth, 
            interval = 500, 
            repeat = False, 
            init_func = self.setup_animation,
            blit = False,
            )
        
    def setup_animation(self, ):
        if isinstance(self.axs, np.ndarray):
            for ax in self.axs.flatten():
                ax.X, ax.y = self.func(**self.func_args)
                ax.scatter(ax.X[:, 0], ax.X[:, 1], c = ax.y, alpha=0.4, s = 10, cmap=self.cmap)
        else:
            self.axs.X, self.axs.y = self.func(**self.func_args)
            self.axs.scatter(self.axs.X[:, 0], self.axs.X[:, 1], c = self.axs.y, alpha=0.4, s = 10, cmap=self.cmap)

    def run_animation(self, depth):
        # self.ax.set_title(f'depth = {depth}')
        if depth == 0:
            # self.ax.scatter(self.X[:, 0], self.X[:, 1], c = self.y, s = 50, cmap='nipy_spectral')
            return
        else:
            if isinstance(self.axs, np.ndarray):
                for ax in self.axs.flatten():
                    ax.clear()
                    # self.ax.set_title(f'depth = {depth}')
                    # self.ax.scatter(self.X[:, 0], self.X[:, 1], c = self.y, s = 50, cmap='plasma')
                    model=self.classifier(max_depth=depth)
                    visualize_tree(model, ax.X, ax.y, boundaries=False, ax=ax, cmap=self.cmap)
            else:
                self.axs.clear()
                # self.ax.set_title(f'depth = {depth}')
                # self.ax.scatter(self.X[:, 0], self.X[:, 1], c = self.y, s = 50, cmap='plasma')
                model=self.classifier(max_depth=depth)
                visualize_tree(model, self.axs.X, self.axs.y, boundaries=False, ax=self.axs, cmap=self.cmap)
            


def plot_colour_tree(clf, ax, y):
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, len(set(y))))

    # print(colors)

    artists = plot_tree(
        clf, 
        ax = ax, 
        filled = False, 
        feature_names=['x', 'y'],
        )
    # print(len(artists))
    # print(clf.tree_.value)
    # print(artists)


    for artist in artists:
        text = artist.get_text()
        if 'False' in text or 'True' in text:
            continue
        elif text[0] == 'g':
            value = text[text.find('['):].replace('\n', ',')
            # print(value)
            value = ast.literal_eval(value)
            # print(value)
            c = colors[np.argmax(value)]
            # print(c)
            bbox = artist.get_bbox_patch()
            bbox.set_facecolor(c)
            # bbox.set_color(1-np.array(c))
            

            

def plot_with_tree(clf, X, y, boundaries=False, cb = True, X_test=None, y_test=None):
    fig, (plot_ax, tree_ax) = plt.subplots(nrows = 1, ncols = 2, dpi=600)
    visualize_tree(clf, X, y, boundaries=boundaries, ax=plot_ax, cb=cb, X_test=X_test, y_test=y_test)
    plot_colour_tree(clf, ax=tree_ax, y=y)

    plot_ax.set_xlabel('x')
    plot_ax.set_ylabel('y')


def plot_contour(clf, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=2, alpha=0.5)
    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax