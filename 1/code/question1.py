import math
import os
import matplotlib.pyplot as plt
import numpy as np

'''
Algorithm A: Newton-Raphson Method

This algorithm returns:
- shortest distance from line (f) to given point (x0, y0)
- point (x, y) on line (f) which is closest to given point (x0, y0)
- the path used to get there as an array of x: [initial_guess, x1, x2 ..., final x]
'''
def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100):
    x = initial_guess
    # Record the path taken for the graph
    path = [x]
    for _ in range(max_iter):
        # D'(x) (derivative of distance formula)
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        # D''(x) (2nd derivative of distance formula)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)
        # x_(i+1) = x_i - D'(x_i)/D''(x_i)
        next_x = x - D_prime / D_double_prime
        # Record this result
        path.append(next_x)
        if abs(next_x - x) < tolerance:
            break
        x = next_x

    # closest point on line: (x, f(x))
    # and distance: D(x, f(x))
    shortest_distance = ((x - x0)**2 + (f(x) - y0)**2)**0.5
    return shortest_distance, x, path

'''
Algorithm B: Golden Section Search
'''
def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    # Record the path taken
    path = [[a, b]]
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
        path.append([a, b])
    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x, path

# Generate a plot showing Newton-Raphson's plot and result
def generate_nr_plot(image_path, x0, y0, f, df, ddf, f_label):
    distance, x_final, path = find_distance_newton(x0, y0, f, df, ddf)
    print(f"NR Path for ({x0}, {y0}): Length={len(path)}\n{path}")
    y_final = f(x_final)

    # zoom to fit every iteration plus both points, with some padding
    xs_seen = path + [x0, x_final]
    x_lo, x_hi = min(xs_seen), max(xs_seen)
    pad = max((x_hi - x_lo) * 0.2, 1)
    xs = np.linspace(x_lo - pad, x_hi + pad, 400)

    plt.plot(xs, f(xs))
    plt.text(xs[len(xs) // 10], f(xs[len(xs) // 10]), f_label, color="tab:blue")

    plt.scatter([x0], [y0], color="red", zorder=5)
    plt.annotate(f"point: ({x0}, {y0})", (x0, y0), textcoords="offset points", xytext=(6, 6))

    path_y = [f(x) for x in path]
    plt.plot(path, path_y, "o--", color="gray")

    plt.scatter([x_final], [y_final], color="green", zorder=5)
    plt.annotate(f"closest point on line: ({x_final:.2f}, {y_final:.2f})", (x_final, y_final), textcoords="offset points", xytext=(6, 6))
    plt.plot([x0, x_final], [y0, y_final], ":", color="black")

    mid_x, mid_y = (x0 + x_final) / 2, (y0 + y_final) / 2
    plt.annotate(f"distance: {distance:.2f}", (mid_x, mid_y), textcoords="offset points", xytext=(6, 6))

    plt.xlim(x_lo - pad, x_hi + pad)
    plt.margins(y=0.15)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()

# Generate a plot showing Golden Section Search's plot and result
def generate_gs_plot(image_path, x0, y0, f, a, b, f_label):
    distance, x_final, path = golden_section_search(x0, y0, f, a, b)
    y_final = f(x_final)

    # zoom to fit a later, narrower bracket plus both points, with padding
    zoom_lo, zoom_hi = path[max(0, len(path) - 5)]
    x_lo, x_hi = min(zoom_lo, x0, x_final), max(zoom_hi, x0, x_final)
    pad = max((x_hi - x_lo) * 0.2, 1)
    xs = np.linspace(x_lo - pad, x_hi + pad, 400)

    plt.plot(xs, f(xs))
    plt.text(xs[len(xs) // 10], f(xs[len(xs) // 10]), f_label, color="tab:blue")

    plt.scatter([x0], [y0], color="red", zorder=5)
    plt.annotate(f"point: ({x0}, {y0})", (x0, y0), textcoords="offset points", xytext=(6, 6))

    # each [a, b] bracket drawn darker than the last to show it narrowing
    n = len(path)
    for i, (lo, hi) in enumerate(path):
        shade = 0.8 - 0.6 * (i / (n - 1)) if n > 1 else 0.2
        plt.axvspan(lo, hi, color=str(shade))

    plt.scatter([x_final], [y_final], color="green", zorder=5)
    plt.annotate(f"closest point on line: ({x_final:.2f}, {y_final:.2f})", (x_final, y_final), textcoords="offset points", xytext=(6, 6))
    plt.plot([x0, x_final], [y0, y_final], ":", color="black")

    mid_x, mid_y = (x0 + x_final) / 2, (y0 + y_final) / 2
    plt.annotate(f"distance: {distance:.2f}", (mid_x, mid_y), textcoords="offset points", xytext=(6, 6))

    plt.xlim(x_lo - pad, x_hi + pad)
    plt.margins(y=0.15)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()

def generate_plot_question1():
    # -- Question 1 General Equations: ---
    # f(x) = x^2 + 5
    def f(x): return x ** 2 + 5
    # df/dx = 2x
    def df(x): return 2 * x
    # ddf/dx = 2
    def ddf(x): return 2

    # 1a)
    x0 = 0
    y0 = 0
    generate_nr_plot("media_q1/1a_nr.png", x0, y0, f, df, ddf, "f(x) = x^2 + 5")
    generate_gs_plot("media_q1/1a_gs.png", x0, y0, f, -10, 10, "f(x) = x^2 + 5")

    # 1b)
    x0 = -4
    y0 = 0
    generate_nr_plot("media_q1/1b_nr.png", x0, y0, f, df, ddf, "f(x) = x^2 + 5")
    generate_gs_plot("media_q1/1b_gs.png", x0, y0, f, -10, 10, "f(x) = x^2 + 5")

    # 1c)
    x0 = -8
    y0 = 0
    generate_nr_plot("media_q1/1c_nr.png", x0, y0, f, df, ddf, "f(x) = x^2 + 5")
    generate_gs_plot("media_q1/1c_gs.png", x0, y0, f, -10, 10, "f(x) = x^2 + 5")

    # 1d)
    x0 = 2
    y0 = 0
    generate_nr_plot("media_q1/1d_nr.png", x0, y0, f, df, ddf, "f(x) = x^2 + 5")
    generate_gs_plot("media_q1/1d_gs.png", x0, y0, f, -10, 10, "f(x) = x^2 + 5")

    # 1e)
    x0 = 6
    y0 = 0
    generate_nr_plot("media_q1/1e_nr.png", x0, y0, f, df, ddf, "f(x) = x^2 + 5")
    generate_gs_plot("media_q1/1e_gs.png", x0, y0, f, -10, 10, "f(x) = x^2 + 5")


if __name__ == "__main__":
    # Question 1
    print("Q1: Running...")
    generate_plot_question1()
    print("Q1: Complete. Saved images to media_q1/")


