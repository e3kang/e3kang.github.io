import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, just save to file
import matplotlib.pyplot as plt

'''
Import data in file: 'data/housing.data'.

Let d = # of features (13), N = # of datapoints (rows)

Returns:
  X: N x d matrix of the 13 feature columns of the file, in order.
  y: length-N vector of targets (MEDV, the last column of the file) 
      as THOUSANDS OF DOLLARS.

Include datapoints from [start, end). Eg for the training data we need first 100 rows, so start=0, end=100.
If left blank, include all datapoints.
'''
def process_data_into_matrix(start=None, end=None):
    raw = np.loadtxt('data/housing.data')      # shape (506, 14), whitespace-separated
    raw = raw[start:end]                       # slicing handles None as "no bound"

    X = raw[:, :-1]                            # first 13 columns are the features
    y = raw[:, -1]                             # MEDV = last column is the target
    return X, y

'''
'''
def gaussian_basis_function(X, centers, a):
    columns = [np.ones((X.shape[0], 1))]

    for center in centers:
        # Sum squared feature differences for each town
        squared_distances = np.sum((X - center) ** 2, axis=1)

        values = np.exp(a * squared_distances)
        columns.append(values[:, None])  # Turn into a column

    return np.hstack(columns)

'''
Uses the pseudo-inverse to find the optimal weight vector: w*

Runs the operation: w* = (basis matrix of training set)^+  * (y of training set)

Example at degree 2: y = w0 + w1 * x1 + w2 * x2 + w3 * x1^2 + w4 * x2^2
'''
def get_optimal_weights(basis_matrix, y):
    weights = np.linalg.pinv(basis_matrix) @ y
    return weights

'''
Plot training and test RMS error versus the number of Gaussian basis functions
on a single axis, and save the figure to `path`.

num_basis_functions: list of K values (x-axis)
training_errors:     list of training RMS errors, one per K
test_errors:         list of test RMS errors, one per K
'''
def plot_error_vs_basis_functions(num_basis_functions, training_errors, test_errors,
                                  path="plots/2e_plot.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure()
    plt.plot(num_basis_functions, training_errors, marker='o', color='tab:blue',
             label='Training error')
    plt.plot(num_basis_functions, test_errors, marker='s', color='tab:orange',
             label='Test error')

    # Label each point with its value, in the colour of its series
    for K, error in zip(num_basis_functions, training_errors):
        plt.annotate(f"{error:.2f}", (K, error), color='tab:blue',
                     textcoords='offset points', xytext=(0, -12), ha='center', fontsize=8)
    for K, error in zip(num_basis_functions, test_errors):
        plt.annotate(f"{error:.2f}", (K, error), color='tab:orange',
                     textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    plt.xlabel('Number of Gaussian basis functions (K)')
    plt.ylabel('RMS error (MEDV, $1000s)')
    plt.title('Training and Test RMS Error vs. Number of Gaussian Basis Functions')
    plt.xticks(num_basis_functions)
    plt.margins(y=0.1)  # leave room for the point labels
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def find_optimal_a(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1, std)
    X_normalized = X_train #(X_train - mean) / std

    distances = []

    for i in range(len(X_normalized)):
        for j in range(i + 1, len(X_normalized)):
            distance = np.linalg.norm(
                X_normalized[i] - X_normalized[j]
            )

            if distance > 0:
                distances.append(distance)

    # One possible choice: half the typical distance between towns
    sigma = 0.5 * np.median(distances)
    return -1 / (2 * sigma**2)

def q2e():
    X_train, y_train = process_data_into_matrix(0, 100)
    X_test, y_test = process_data_into_matrix(100)

    # Use only the second feature: ZN
    X_train = X_train[:, [1]]
    X_test = X_test[:, [1]]

    # Find optimal a
    a = find_optimal_a(X_train)
    print("Q2e) Optimal a: ", a)

    # Shuffle the data points
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_train))
    num_basis_functions = list(range(5, 96, 10))  # 5, 15, ..., 95
    training_errors = []
    test_errors = []

    for K in num_basis_functions:
        centers = X_train[indices[:K]]
        phi_train = gaussian_basis_function(X_train, centers, a)
        optimal_weights = get_optimal_weights(phi_train, y_train)

        phi_test = gaussian_basis_function(X_test, centers, a)

        train_predictions = phi_train @ optimal_weights
        test_predictions = phi_test @ optimal_weights

        train_rmse = np.sqrt(np.mean((train_predictions - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
        training_errors.append(train_rmse)
        test_errors.append(test_rmse)
        print(f"K={K}: training RMSE = {train_rmse:.4f}, test RMSE = {test_rmse:.4f}")
        print(
            f"K={K}, "
            f"unique centers={len(np.unique(centers, axis=0))}, "
            f"rank={np.linalg.matrix_rank(phi_train)}, "
            f"train RMSE={train_rmse:.4f}, "
            f"test RMSE={test_rmse:.4f}, "
            f"prediction range=({test_predictions.min():.2f}, "
            f"{test_predictions.max():.2f})"
        )
        print(f"Optimal weights: ", optimal_weights)
    print("Training ZN range:", X_train.min(), X_train.max())
    print("Test ZN range:", X_test.min(), X_test.max())
    print("a:", a)
    plot_error_vs_basis_functions(num_basis_functions, training_errors, test_errors,
                                  "plots/2e_plot.png")

# ------ 1e) ------
def get_optimal_weights_regularized(basis_matrix, y, lamda):
    Phi = basis_matrix
    I = np.eye(Phi.shape[1])

    return np.linalg.solve(
        Phi.T @ Phi + lamda * I,
        Phi.T @ y
    )

def cross_validation_lambda_gaussian(X, y, lambdas, a):
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(len(y)), 10)

    average_errors = []

    for lam in lambdas:
        fold_errors = []

        for k in range(10):
            validation_indices = folds[k]
            training_indices = np.concatenate(
                [folds[j] for j in range(10) if j != k]
            )

            X_train = X[training_indices]
            y_train = y[training_indices]
            X_validation = X[validation_indices]
            y_validation = y[validation_indices]

            # Normalize using this fold's training data only
            mean = X_train.mean(axis=0)
            std = X_train.std(axis=0)
            std = np.where(std == 0, 1, std)

            X_train = (X_train - mean) / std
            X_validation = (X_validation - mean) / std

            # All 90 training points become Gaussian centres
            centers = X_train.copy()

            Phi_train = gaussian_basis_function(X_train, centers, a)
            Phi_validation = gaussian_basis_function(X_validation, centers, a)

            if lam == 0:
                weights = get_optimal_weights(Phi_train, y_train)
            else:
                weights = get_optimal_weights_regularized(
                    Phi_train, y_train, lam
                )

            predictions = Phi_validation @ weights
            error = np.sqrt(
                np.mean((predictions - y_validation) ** 2)
            )
            fold_errors.append(error)

        average_errors.append(np.mean(fold_errors))

    best_index = np.argmin(average_errors)
    return lambdas[best_index], np.array(average_errors)

def q2f():
    X, y = process_data_into_matrix(0, 100)
    lambdas = np.array([0, 0.01, 0.1, 1, 10, 100, 1000])

    # Choose the Gaussian width from standardized features, so that `a` is on
    # the same scale as the per-fold standardized data used inside the CV loop.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1, std)
    a = find_optimal_a((X - mean) / std)

    best_lambda, average_errors = cross_validation_lambda_gaussian(
        X, y, lambdas, a=a
    )

    for lam, error in zip(lambdas, average_errors):
        print(f"lambda={lam}: average validation RMSE={error:.6f}")
    print(f"Chosen lambda: {best_lambda}")
    print(f"Gaussian parameter a: {a}")
    # Zero cannot appear on a logarithmic x-axis.
    # Report its error separately above.
    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas[1:], average_errors[1:], "o-", label="Regularized (lambda > 0)")

    # Unregularized baseline: bold dashed line, labelled with its value
    plt.axhline(
        average_errors[0],
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Unregularized (lambda = 0): {average_errors[0]:.2f}"
    )
    plt.annotate(
        f"lambda = 0: {average_errors[0]:.2f}",
        (lambdas[1], average_errors[0]),
        textcoords="offset points", xytext=(0, 8), ha="left",
        fontsize=8, color="red"
    )

    # Label every regularized point with its value
    for lam, error in zip(lambdas[1:], average_errors[1:]):
        plt.annotate(f"{error:.2f}", (lam, error),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    if best_lambda > 0:
        best_index = np.argmin(average_errors)
        plt.scatter(
            best_lambda,
            average_errors[best_index],
            color="green",
            s=80,
            zorder=3,
            label=f"Chosen lambda = {best_lambda}"
        )
    plt.legend()

    plt.xlabel("Regularization constant (lambda)")
    plt.ylabel("Average validation RMSE (thousands of dollars)")
    plt.title("10-fold cross-validation: 90 Gaussian basis functions")
    plt.grid(True, which="both", alpha=0.3)
    plt.margins(y=0.1)
    plt.tight_layout()
    plt.savefig("plots/2f_gaussian_regression_reg.png", dpi=300)
    return best_lambda, average_errors

if __name__ == '__main__':
    print("--- Q2e) ---")
    q2e()
    print("--- Q2f) ---")
    q2f()