import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, just save to file
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS
from ucimlrepo import fetch_ucirepo

# One colour per species, reused by every plot so the legend is consistent
SPECIES_COLORS = {
    'Iris-setosa': 'tab:blue',
    'Iris-versicolor': 'tab:orange',
    'Iris-virginica': 'tab:green',
}

'''
Fetch the Iris dataset from the UCI repository (id=53).

Let d = # of features (4), N = # of datapoints (150)

Returns:
- X: N x d dataframe
- y: length N series of the class label (species name).
'''
def load_dataset():
    iris = fetch_ucirepo(id=53)

    X = iris.data.features
    y = iris.data.targets['class']

    return X, y

# ------ Q2 ------
'''
1D plot (line), colour-coded by species. 

Save to `path`.
'''
def plot_one_feature(X, y, feature, path="q2_plots/one_feature.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(8, 2.5))
    # Draw one species at a time
    for species, color in SPECIES_COLORS.items():
        mask = (y == species)
        plt.scatter(X.loc[mask, feature], [0] * mask.sum(),
                    color=color, label=species, alpha=0.6, s=40)

    plt.xlabel(f'{feature} (cm)')
    plt.yticks([])  # no vertical axis for 1d
    plt.title(f'Iris marginalized onto one feature: {feature}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

'''
2D scatter of `feature_x` against `feature_y`, colour-coded by species.

Save to `path`.
'''
def plot_two_features(X, y, feature_x, feature_y, path="q2_plots/two_features.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(7, 6))
    for species, color in SPECIES_COLORS.items():
        mask = (y == species)
        plt.scatter(X.loc[mask, feature_x], X.loc[mask, feature_y],
                    color=color, label=species, alpha=0.7, s=40)

    plt.xlabel(f'{feature_x} (cm)')
    plt.ylabel(f'{feature_y} (cm)')
    plt.title(f'Iris marginalized onto two features: {feature_x} vs. {feature_y}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

'''
3D scatter of `feature_x`, `feature_y`, `feature_z`, colour-coded by species.

Save to `path`.
'''
def plot_three_features(X, y, feature_x, feature_y, feature_z,
                        path="q2_plots/three_features.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    for species, color in SPECIES_COLORS.items():
        mask = (y == species)
        ax.scatter(X.loc[mask, feature_x], X.loc[mask, feature_y], X.loc[mask, feature_z],
                   color=color, label=species, alpha=0.7, s=40)

    ax.set_xlabel(f'{feature_x} (cm)')
    ax.set_ylabel(f'{feature_y} (cm)')
    ax.set_zlabel(f'{feature_z} (cm)')
    ax.set_title(f'Iris marginalized onto three features:\n'
                 f'{feature_x}, {feature_y}, {feature_z}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ------ Q3 ------
'''
Run PCA on dataset using scikit learn.
'''
def run_pca(X):
    pca = PCA(n_components=4)
    Z = pca.fit_transform(X.values)
    return pca, Z

'''
Print the PC axis vectors and the variance each explains.
'''
def print_pca_summary(pca, feature_names):
    print("PCA axis vectors (rows are PCs, columns are the original features):")
    print("    features:", list(feature_names))
    for i, (vector, var, ratio) in enumerate(zip(pca.components_,
                                                 pca.explained_variance_,
                                                 pca.explained_variance_ratio_), start=1):
        print(f"  PC{i}: {np.round(vector, 4)}  "
              f"eigenvalue={var:.4f}  variance ratio={ratio:.4f}")
    print("Cumulative variance ratio:", np.round(np.cumsum(pca.explained_variance_ratio_), 4))

'''
Plot the data on one axis: PC1. 

Save to `path`.
'''
def plot_pc1(Z, y, path="q3_plots/pc1.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(8, 2.5))
    for species, color in SPECIES_COLORS.items():
        mask = (y == species).values
        plt.scatter(Z[mask, 0], [0] * mask.sum(),
                    color=color, label=species, alpha=0.6, s=40)

    plt.xlabel('PC1')
    plt.yticks([])  # the vertical axis carries no information
    plt.title('Iris projected onto the major axis (PC1)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

'''
Plot the data in two axes: PC1, PC2

Save to `path`.
'''
def plot_pc1_pc2(Z, y, path="q3_plots/pc1_pc2.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(7, 6))
    for species, color in SPECIES_COLORS.items():
        mask = (y == species).values
        plt.scatter(Z[mask, 0], Z[mask, 1],
                    color=color, label=species, alpha=0.7, s=40)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Iris projected onto the top 2 PCs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

'''
Plot the data on three axes: PC1, PC2, PC3

Save to `path`.
'''
def plot_pc1_pc2_pc3(Z, y, path="q3_plots/pc1_pc2_pc3.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    for species, color in SPECIES_COLORS.items():
        mask = (y == species).values
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                   color=color, label=species, alpha=0.7, s=40)

    ax.view_init(elev=20, azim=-120)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Iris projected onto the top 3 PCs')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ------ Q4 ------
'''
Returns the percentage of variance explained by the first n PCA components.

Format: 
- percentage (out of 100)
- rounded to 4 decimal places.
'''
def variance_explained_by_pca_components(pca, n):
    return round(pca.explained_variance_ratio_[:n].sum(), 4) * 100

'''
Returns the amount of PCs needed to be over `n%` variance explained.
'''
def number_of_pcs_for_variance_over(pca, n):
    current_variance = 0
    components = 0
    for var in pca.explained_variance_ratio_:
        current_variance += var * 100 # convert to percentage
        components += 1
        if current_variance > n:
            break
    return components
# ------ Q5 ------
'''
Fit ISOMAP with scikit-learn and embed the data in `n_components` dimensions.

Returns:
- isomap: fitted Isomap object (isomap.dist_matrix_ holds the N x N geodesic distances)
- Z: N x n_components embedding
'''
def run_isomap(X, n_neighbors=3, n_components=2):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    Z = isomap.fit_transform(X.values)
    return isomap, Z

'''
2D scatter of an embedding (ISOMAP or MDS), colour-coded by species. Save to `path`.
'''
def plot_embedding(Z, y, title, xlabel, ylabel, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(7, 6))
    for species, color in SPECIES_COLORS.items():
        mask = (y == species).values
        plt.scatter(Z[mask, 0], Z[mask, 1],
                    color=color, label=species, alpha=0.7, s=40)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == '__main__':
    # ------ Q2 ------
    X, y = load_dataset()
    print("Features:", list(X.columns))
    print("Classes:", y.unique())

    plot_one_feature(X, y, 'petal length')
    plot_two_features(X, y, 'petal length', 'petal width')
    plot_three_features(X, y, 'sepal length', 'petal length', 'petal width')
    print("Saved plots to q2_plots/")

    # ------ Q3 ------
    print("---- Q3 ----")
    pca, Z = run_pca(X)
    print_pca_summary(pca, X.columns)

    plot_pc1(Z, y)
    plot_pc1_pc2(Z, y)
    plot_pc1_pc2_pc3(Z, y)

    print("Saved plots to q3_plots/")
    # ------- Q4 ------
    print("---- Q4 ----")
    print(f"Variance explained by each component: {pca.explained_variance_}")
    print(f"Ratio of variance explained for each component: {list(map(lambda x: f"{(x * 100):.4f}%", pca.explained_variance_ratio_))}")
    print(f"Amount of variance explained by 2D representation: {variance_explained_by_pca_components(pca, 2)}%")
    print(f"Number of PCs to explain 50% of variance: {number_of_pcs_for_variance_over(pca, 50)}")

    # ------ Q5 ------
    print("---- Q5 ----")
    K = 3

    isomap, Z_iso = run_isomap(X, n_neighbors=K, n_components=2)
    print(f"ISOMAP (k={K}) reconstruction error: {isomap.reconstruction_error():.4f}")
    plot_embedding(Z_iso, y,
                   title=f'ISOMAP embedding of Iris ({K}-nearest neighbours)',
                   xlabel='ISOMAP component 1', ylabel='ISOMAP component 2',
                   path='q5_plots/isomap_3nn.png')
    print("Saved plots to q5_plots/")
