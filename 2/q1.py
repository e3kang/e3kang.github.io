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
Polynomial Basis 

Takes a N x d matrix and turns it into a polynomial basis matrix.
By definition, appends a column of 1s.

'''
def polynomial_basis(X, degree):
    columns = [np.ones((X.shape[0], 1))]

    for power in range(1, degree + 1):
        columns.append(X ** power)  # Raise each feature separately

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
Given the weights array and degree, return a function that can 
evaluate y given an array of x.

'''
def prediction_function(weights, degree):
    def predict(X):
        Phi = polynomial_basis(X, degree)
        return Phi @ weights

    return predict

'''
Evaluate the weights against a given matrix of data X and its labels Y.

Returns the Root Mean Squared Error.
'''
def evaluate_rmse(weights, degree, X, Y):
    predict = prediction_function(weights, degree)
    predictions = predict(X)
    squared_errors = (predictions - Y) ** 2
    return np.sqrt(np.mean(squared_errors))
    

'''
Plot RMS error versus polynomial degree and save the figure to `path`.

degrees: list of polynomial degrees (x-axis)
errors:  list of RMS errors, one per degree (y-axis)
title:   title of the plot (eg "Training error")
logScale: if True, plot the RMS error axis on a log scale
'''
def plot_error(degrees, errors, title, path, logScale=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure()
    plt.plot(degrees, errors, marker='o')
    if logScale:
        plt.yscale('log')
    for degree, error in zip(degrees, errors):
        # Label each point with its value, offset slightly above the marker
        plt.annotate(f"{error:.4g}", (degree, error),
                     textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    plt.xlabel('Polynomial degree')
    plt.ylabel('RMS error (log scale)' if logScale else 'RMS error')
    plt.title(title)
    plt.xticks(degrees)
    plt.grid(True)
    plt.savefig(path)
    plt.close()

# -------------- 1c --------------

'''
Plot a fitted 1D polynomial line against the training and test data points
on a single figure, and save to `path`.

weights:          weight vector w* from a 1-feature polynomial regression
degree:           degree of that polynomial
x_train, y_train: 1D training feature values and their targets
x_test, y_test:   1D test feature values and their targets
'''
def plot_1d_line(weights, degree, x_train, y_train, x_test, y_test,
                 title, xlabel, ylabel, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    predict = prediction_function(weights, degree)

    # Evaluate the fitted polynomial on a dense grid spanning both splits
    x_all = np.concatenate([x_train, x_test])
    x_line = np.linspace(x_all.min(), x_all.max(), 500).reshape(-1, 1)
    y_line = predict(x_line)

    plt.figure()
    plt.scatter(x_train, y_train, s=12, color='tab:blue', label='Training data')
    plt.scatter(x_test, y_test, s=12, color='tab:orange', label='Test data')
    plt.plot(x_line, y_line, color='red', label=f'Degree {degree} fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=9)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def polynomial_regression_1d_vis():
    ZN = 1  # column index of ZN: proportion of residential land zoned for lots > 25k sq ft
    X_train, y_train = process_data_into_matrix(0, 100)
    X_test, y_test = process_data_into_matrix(100)
    X_train = X_train[:, [ZN]]
    X_test = X_test[:, [ZN]]

    xlabel = 'Proportion of residential land zoned for lots over 25,000 sq ft (ZN, %)'
    ylabel = 'Median home value (MEDV, $1000s)'

    for degree in [1, 4, 7]:  # low, medium, high degree
        basis_matrix_X = polynomial_basis(X_train, degree)
        w_star = get_optimal_weights(basis_matrix_X, y_train)
        training_error = evaluate_rmse(w_star, degree, X_train, y_train)
        print(f"1c) At degree={degree}: training error = {training_error}")
        test_error = evaluate_rmse(w_star, degree, X_test, y_test)
        print(f"1c) At degree={degree}: test error = {test_error}")

        title = (f"Proportion of Land Zoned (>25k sqft) vs Median Home Price\n"
                 f"(Degree={degree}, Training and Test Data)")
        plot_1d_line(w_star, degree, X_train[:, 0], y_train, X_test[:, 0], y_test,
                     title, xlabel, ylabel, f"plots/1c_degree{degree}.png")

# -------------- 1d --------------
def get_optimal_weights_regularized(basis_matrix, y, lamda):
    Phi = basis_matrix
    I = np.eye(Phi.shape[1])

    return np.linalg.solve(
        Phi.T @ Phi + lamda * I,
        Phi.T @ y
    )

def cross_validation_lambda(X, y, degree, lambdas):
    # rng = np.random.default_rng(42)
    # folds = np.array_split(rng.permutation(len(y)), 10)
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

            Phi_train = polynomial_basis(X_train, degree)

            if lam == 0:
                weights = get_optimal_weights(Phi_train, y_train)
            else:
                weights = get_optimal_weights_regularized(
                    Phi_train, y_train, lam
                )

            error = evaluate_rmse(
                weights, degree, X_validation, y_validation
            )
            fold_errors.append(error)

        average_errors.append(np.mean(fold_errors))

    best_index = np.argmin(average_errors)
    return lambdas[best_index], np.array(average_errors)

def polynomial_regression_reg():
    ZN = 1  # column index of ZN: proportion of residential land zoned for lots > 25k sq ft
    X_train, y_train = process_data_into_matrix(0, 100)
    X_test, y_test = process_data_into_matrix(100)
    X_train = X_train[:, [ZN]]
    X_test = X_test[:, [ZN]]
    lambdas = np.array([0, 0.01, 0.1, 1, 10, 100, 1000])
    optimal_lambda, avg_errors = cross_validation_lambda(X_train, y_train, 8, lambdas)
    for lam, err in zip(lambdas, avg_errors):
        print(f"lambda={lam}: average validation RMSE = {err:.4f}")

    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.semilogx(lambdas[1:], avg_errors[1:], "o-", label="Regularized (lambda > 0)")
    plt.axhline(
      avg_errors[0],
      color="gray",
      linestyle="--",
      label=f"Unregularized (lambda = 0): {avg_errors[0]:.2f}"
    )
    for lam, err in zip(lambdas[1:], avg_errors[1:]):
        plt.annotate(f"{err:.2f}", (lam, err),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    plt.xlabel("Regularization constant (lambda)")
    plt.ylabel("Average validation RMSE")
    plt.title("10-fold cross-validation: degree 8 polynomial on ZN")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/1d_avg_validation_error.png")
    plt.close()

    print("Chosen lambda:", optimal_lambda)

# --------------------------------

def part1_a():
    # Get training data as a N x d matrix
    X_train, y_train = process_data_into_matrix(0, 100)
    X_test, y_test = process_data_into_matrix(100)

    degrees = list(range(1, 8))
    training_errors = []
    test_errors = []

    for degree in degrees:
      # Get basis function of training data X
      basis_matrix_X = polynomial_basis(X_train, degree)

      # Get optimal weights
      w_star = get_optimal_weights(basis_matrix_X, y_train)
      print("------")
      # print(f"At degree={degree}: w* = {w_star}")

      training_error = evaluate_rmse(w_star, degree, X_train, y_train)
      training_errors.append(training_error)
      print(f"At degree={degree}: training error = {training_error}")
      test_error = evaluate_rmse(w_star, degree, X_test, y_test)
      test_errors.append(test_error)
      print(f"At degree={degree}: test error = {test_error}")
      print("------")

    plot_error(degrees, training_errors, "Training error", "plots/1a_training_error.png")
    plot_error(degrees, test_errors, "Test error", "plots/1a_test_error.png")
    plot_error(degrees, test_errors, "Test error (log scale)", "plots/1a_test_error_log.png", logScale=True)

def part1_b():
    X_train, y_train = process_data_into_matrix(0, 100)
    X_test, y_test = process_data_into_matrix(100)
    basis_matrix_X = polynomial_basis(X_train, 1)
    w_star = get_optimal_weights(basis_matrix_X, y_train)
    weights_formatted = list(map(lambda val: float(round(val, 5)), w_star))
    print("------ Q1 b) ------")
    print(f"All weights: {weights_formatted}")
    print(f"Weight on 6th feature (Avg number of rooms/dwelling): {weights_formatted[6]}")
    print(f"Weight of 8th feature (Weighted distance to employment centers): {weights_formatted[8]}")
    print(f"Bias term: {weights_formatted[0]}")
    print("-------------------")

def part1_c():
    polynomial_regression_1d_vis()

def part1_d():
    polynomial_regression_reg()
  
if __name__ == '__main__':
    part1_a()
    part1_b()
    part1_c()
    part1_d()
