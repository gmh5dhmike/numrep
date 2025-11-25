#!/usr/bin/env python3
"""
Bessel plotting script for numrep assignment.

- Uses professor's up/down recursion implementation for spherical Bessel j_l(x)
- Computes j0, j3, j5, j8 for both upward and downward recursion
- Produces a single plot with 8 curves and saves it as bessel.png
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos


# =========================
# Upward and downward recursion (professor's code)
# =========================

def up(x, l):
    """
    Upward recursion for spherical Bessel j_l(x) using the professor's pattern.
    """
    j0 = sin(x) / x
    if l == 0:
        return j0
    j1 = (j0 - cos(x)) / x
    if l == 1:
        return j1
    # else use recursion relation recursively
    l1 = l - 1
    return (2 * l1 + 1) / x * up(x, l1) - up(x, l1 - 1)


def down(x, n):
    """
    Downward recursion for spherical Bessel j_n(x) using a large lMax and
    scaling so j0 matches sin(x)/x.
    """
    lMax = 50
    # j indices: 0..lMax+1
    j = [0.0] * (lMax + 2)
    j[lMax] = 1.0
    j[lMax + 1] = 1.0

    # downward recurrence:
    # j_{k-1} = ((2k + 1)/x) * j_k - j_{k+1}
    for k in range(lMax, 0, -1):
        j[k - 1] = ((2.0 * k + 1.0) / x) * j[k] - j[k + 1]

    # scale so that j0 matches sin(x)/x
    j0_exact = sin(x) / x
    scale = j0_exact / j[0]
    return j[n] * scale


# =========================
# Main: compute and plot j0, j3, j5, j8 (up & down)
# =========================

def main():
    # x range similar to professor's example
    xmin = 0.1
    xmax = 40.0
    step = 0.1
    x_vals = np.arange(xmin, xmax + step, step)

    orders = [0, 3, 5, 8]

    # storage: dict[order] -> arrays for up and down
    vals_up = {}
    vals_down = {}

    for l in orders:
        up_list = []
        down_list = []
        for x in x_vals:
            up_list.append(up(x, l))
            down_list.append(down(x, l))
        vals_up[l] = np.array(up_list)
        vals_down[l] = np.array(down_list)

    # Nice looking plot
    plt.figure(figsize=(8, 6))

    for l in orders:
        plt.plot(x_vals, vals_up[l], linestyle='-', label=f"j{l} up")
        plt.plot(x_vals, vals_down[l], linestyle='--', label=f"j{l} down")

    plt.xlabel("x")
    plt.ylabel(r"$j_\ell(x)$")
    plt.title("Spherical Bessel functions j₀, j₃, j₅, j₈\nUpward vs downward recursion")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("bessel.png", dpi=300)
    plt.close()

    print("Saved plot as bessel.png")


if __name__ == "__main__":
    main()
