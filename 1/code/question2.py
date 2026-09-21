import os
import matplotlib.pyplot as plt
import numpy as np

# Question 2
def fit_line_mr_multivariate(x_points, y_points, m_guess, b_guess, tolerance=1e-7, max_iter=1000):
    m, b = m_guess, b_guess
    path = [[m, b]]
    n = len(x_points) # dimensions
    for _ in range(max_iter):
        m_old, b_old = m, b

        # Update m (hold b fixed)
        J_m  = (2/n) * sum(x_i * (m*x_i + b - y_i) for x_i, y_i in zip(x_points, y_points))   # dJ/dm
        J_mm = (2/n) * sum(x_i*x_i for x_i in x_points)                             # d2J/dm2
        m = m - J_m / J_mm

        # Update b (hold m fixed)
        J_b  = (2/n) * sum((m*x_i + b - y_i) for x_i, y_i in zip(x_points, y_points))       # dJ/db
        J_bb = 2.0                                                      # d2J/db2 = (2/N)*N
        b = b - J_b / J_bb
        path.append([m, b])
        if abs(m - m_old) < tolerance and abs(b - b_old) < tolerance:
            break

    mse = sum((m*x_i + b - y_i)**2 for x_i, y_i in zip(x_points, y_points)) / n
    return mse, m, b, path

def fit_parabola_nr_multivariate(x_points, y_points, a_guess, b_guess, c_guess, tolerance=1e-7, max_iter=1000):
    a, b, c = a_guess, b_guess, c_guess
    path = [[a, b, c]]
    n = len(x_points) # dimensions
    for _ in range(max_iter):
        a_old, b_old, c_old = a, b, c

        # Update a (hold b, c fixed)
        J_a  = (2/n) * sum(x_i**2 * (a*x_i**2 + b*x_i + c - y_i) for x_i, y_i in zip(x_points, y_points))   # dJ/da
        J_aa = (2/n) * sum(x_i**4 for x_i in x_points)                                        # d2J/da2
        a = a - J_a / J_aa

        # Update b (hold a, c fixed)
        J_b  = (2/n) * sum(x_i * (a*x_i**2 + b*x_i + c - y_i) for x_i, y_i in zip(x_points, y_points))      # dJ/db
        J_bb = (2/n) * sum(x_i*x_i for x_i in x_points)                                       # d2J/db2
        b = b - J_b / J_bb

        # Update c (hold a, b fixed)
        J_c  = (2/n) * sum((a*x_i**2 + b*x_i + c - y_i) for x_i, y_i in zip(x_points, y_points))            # dJ/dc
        J_cc = 2.0                                                      # d2J/dc2 = (2/N)*N
        c = c - J_c / J_cc
        path.append([a, b, c])
        if abs(a - a_old) < tolerance and abs(b - b_old) < tolerance and abs(c - c_old) < tolerance:
            break

    mse = sum((a*x_i**2 + b*x_i + c - y_i)**2 for x_i, y_i in zip(x_points, y_points)) / n
    return mse, a, b, c, path

# Generate a plot showing each iteration's line converging to the line of best fit
def generate_line_plot_q2(image_path, x_points, y_points, m_guess=0, b_guess=0):
    mse, m_final, b_final, path = fit_line_mr_multivariate(x_points, y_points, m_guess, b_guess)

    x_lo, x_hi = min(x_points), max(x_points)
    pad = max((x_hi - x_lo) * 0.2, 1)
    xs = np.linspace(x_lo - pad, x_hi + pad, 400)

    # each iteration's line drawn darker than the last to show it converging
    n = len(path)
    for i, (m, b) in enumerate(path[:-1]):
        shade = 0.8 - 0.6 * (i / (n - 1)) if n > 1 else 0.2
        plt.plot(xs, m * xs + b, color=str(shade), linewidth=1)

    plt.plot(xs, m_final * xs + b_final, color="green", zorder=4)
    plt.text(xs[-1], m_final * xs[-1] + b_final, f"y = {m_final:.2f}x + {b_final:.2f}\nMSE: {mse:.5f}", color="green", ha="right", va="bottom")

    plt.scatter(x_points, y_points, color="red", zorder=5)
    for x_i, y_i in zip(x_points, y_points):
        plt.annotate(f"({x_i}, {y_i})", (x_i, y_i), textcoords="offset points", xytext=(6, -12))

    plt.xlim(x_lo - pad, x_hi + pad)
    plt.margins(y=0.15)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()

# Generate a plot showing each iteration's parabola converging to the parabola of best fit
def generate_parabola_plot_q2(image_path, x_points, y_points, a_guess=0, b_guess=0, c_guess=0):
    mse, a_final, b_final, c_final, path = fit_parabola_nr_multivariate(x_points, y_points, a_guess, b_guess, c_guess)

    x_lo, x_hi = min(x_points), max(x_points)
    pad = max((x_hi - x_lo) * 0.2, 1)
    xs = np.linspace(x_lo - pad, x_hi + pad, 400)

    # each iteration's parabola drawn darker than the last to show it converging
    n = len(path)
    for i, (a, b, c) in enumerate(path[:-1]):
        shade = 0.8 - 0.6 * (i / (n - 1)) if n > 1 else 0.2
        plt.plot(xs, a * xs**2 + b * xs + c, color=str(shade), linewidth=1)

    plt.plot(xs, a_final * xs**2 + b_final * xs + c_final, color="green", zorder=4)
    plt.text(xs[-1], a_final * xs[-1]**2 + b_final * xs[-1] + c_final, f"y = {a_final:.2f}x^2 + {b_final:.2f}x + {c_final:.2f}\nMSE: {mse:.5f}", color="green", ha="right", va="bottom")

    plt.scatter(x_points, y_points, color="red", zorder=5)
    for x_i, y_i in zip(x_points, y_points):
        plt.annotate(f"({x_i}, {y_i})", (x_i, y_i), textcoords="offset points", xytext=(6, -12))

    plt.xlim(x_lo - pad, x_hi + pad)
    plt.margins(y=0.15)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()

def question2():
    x_points = [0, 2, 1, 3]
    y_points = [0.5, 3.5, 1.5, 7.5]
    mse, m, b, path = fit_line_mr_multivariate(
        x_points=x_points, y_points=y_points, 
        m_guess=0, b_guess=0
    )
    print(f"Line of best fit: y = {round(m, 5)}x + {round(b, 5)}")
    print(f"MSE: {mse}")
    print(f"Number of iterations: {len(path)}")
    generate_line_plot_q2("media_q2/line.png", x_points, y_points, m_guess=0, b_guess=0)

    mse, a, b, c, path = fit_parabola_nr_multivariate(
        x_points=x_points, y_points=y_points,
        a_guess=0, b_guess=0, c_guess=0
    )
    print(f"Parabola of best fit: y = {round(a, 5)}x^2 + {round(b, 5)}x + {round(c, 5)}")
    print(f"MSE: {mse}")
    print(f"Number of iterations: {len(path)}")
    generate_parabola_plot_q2("media_q2/parabola.png", x_points, y_points, a_guess=0, b_guess=0, c_guess=0)

if __name__ == "__main__":
    print("Q2: Running...")
    question2()
    print("Q2: Complete. Results posted and saved plots to media_q2/")
