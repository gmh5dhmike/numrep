#!/usr/bin/env python3
"""
Numerical differentiation test script for cos(t) and exp(t).

- Uses forward, central, and extrapolated differences
- Evaluates at t = 0.1, 1.0, 100.0
- Prints derivative and relative error as functions of h
- Plots log10(error) vs log10(h) and saves cos.png and exp.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Difference formulas
# =========================

def forward_diff(f, t, h):
    """Forward difference approximation to f'(t)."""
    return (f(t + h) - f(t)) / h


def central_diff(f, t, h):
    """Central difference approximation to f'(t)."""
    return (f(t + 0.5 * h) - f(t - 0.5 * h)) / h


def extrapolated_diff(f, t, h):
    """
    Extrapolated (Richardson) difference based on central difference.

    D_e(h) = (4 * D_c(h/2) - D_c(h)) / 3
    """
    Dc_h = central_diff(f, t, h)
    Dc_h2 = central_diff(f, t, 0.5 * h)
    return (4.0 * Dc_h2 - Dc_h) / 3.0


# =========================
# Error utilities
# =========================

def relative_error(approx, exact):
    """Relative error |approx - exact| / |exact|."""
    return abs((approx - exact) / exact)


def derivative_sweep(f, fprime, t, h_values):
    """
    For a given function f, its exact derivative fprime, a point t,
    and an array of step sizes h_values, compute derivative approximations
    and relative errors for forward, central, and extrapolated differences.
    """
    exact = fprime(t)

    fwd = np.zeros_like(h_values)
    cen = np.zeros_like(h_values)
    ext = np.zeros_like(h_values)

    err_fwd = np.zeros_like(h_values)
    err_cen = np.zeros_like(h_values)
    err_ext = np.zeros_like(h_values)

    for i, h in enumerate(h_values):
        fwd[i] = forward_diff(f, t, h)
        cen[i] = central_diff(f, t, h)
        ext[i] = extrapolated_diff(f, t, h)

        err_fwd[i] = relative_error(fwd[i], exact)
        err_cen[i] = relative_error(cen[i], exact)
        err_ext[i] = relative_error(ext[i], exact)

    return {
        "exact": exact,
        "h": h_values,
        "fwd": fwd,
        "cen": cen,
        "ext": ext,
        "err_fwd": err_fwd,
        "err_cen": err_cen,
        "err_ext": err_ext,
    }


def log10_safe(x):
    """log10 with protection against log10(0)."""
    tiny = np.finfo(float).tiny
    return np.log10(np.clip(x, tiny, None))


# =========================
# Main experiment
# =========================

def main():
    # Points where we evaluate the derivative
    t_values = [0.1, 1.0, 100.0]

    # Step sizes from 1e-1 down to 1e-16 (roughly machine precision)
    h_values = np.logspace(-1, -16, num=16)
    eps = np.finfo(float).eps

    print("Machine epsilon (double precision):", eps)
    print("h range from", h_values[0], "to", h_values[-1])

    # Define functions and their exact derivatives
    def f_cos(t):
        return math.cos(t)

    def fprime_cos(t):
        return -math.sin(t)

    def f_exp(t):
        return math.exp(t)

    def fprime_exp(t):
        return math.exp(t)

    results_cos = {}
    results_exp = {}

    # ---- cos(t) ----
    print("\n===== cos(t) =====")
    for t in t_values:
        res = derivative_sweep(f_cos, fprime_cos, t, h_values)
        results_cos[t] = res

        print(f"\n--- cos(t) at t = {t} ---")
        print(f"exact derivative = {res['exact']:.16e}")
        print(f"{'h':>12} {'forward':>18} {'rel.err_fwd':>18} "
              f"{'central':>18} {'rel.err_cen':>18} "
              f"{'extrap':>18} {'rel.err_ext':>18}")
        for i, h in enumerate(h_values):
            print(f"{h:12.1e} "
                  f"{res['fwd'][i]:18.10e} {res['err_fwd'][i]:18.10e} "
                  f"{res['cen'][i]:18.10e} {res['err_cen'][i]:18.10e} "
                  f"{res['ext'][i]:18.10e} {res['err_ext'][i]:18.10e}")

    # ---- exp(t) ----
    print("\n===== exp(t) =====")
    for t in t_values:
        res = derivative_sweep(f_exp, fprime_exp, t, h_values)
        results_exp[t] = res

        print(f"\n--- exp(t) at t = {t} ---")
        print(f"exact derivative = {res['exact']:.16e}")
        print(f"{'h':>12} {'forward':>18} {'rel.err_fwd':>18} "
              f"{'central':>18} {'rel.err_cen':>18} "
              f"{'extrap':>18} {'rel.err_ext':>18}")
        for i, h in enumerate(h_values):
            print(f"{h:12.1e} "
                  f"{res['fwd'][i]:18.10e} {res['err_fwd'][i]:18.10e} "
                  f"{res['cen'][i]:18.10e} {res['err_cen'][i]:18.10e} "
                  f"{res['ext'][i]:18.10e} {res['err_ext'][i]:18.10e}")

    # =========================
    # Plotting
    # =========================

    # Nice default plotting style
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["font.size"] = 11

    logh = np.log10(h_values)

    # ---- cos(t) plot ----
    plt.figure()
    for t in t_values:
        res = results_cos[t]
        plt.plot(logh, log10_safe(res["err_fwd"]), "-o", label=f"forward, t={t}")
        plt.plot(logh, log10_safe(res["err_cen"]), "-s", label=f"central, t={t}")
        plt.plot(logh, log10_safe(res["err_ext"]), "-^", label=f"extrap, t={t}")

    plt.xlabel(r"$\log_{10}|h|$")
    plt.ylabel(r"$\log_{10}|\text{relative error}|$")
    plt.title(r"Numerical derivative errors for $\cos(t)$")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("cos.png", dpi=300)
    plt.close()

    # ---- exp(t) plot ----
    plt.figure()
    for t in t_values:
        res = results_exp[t]
        plt.plot(logh, log10_safe(res["err_fwd"]), "-o", label=f"forward, t={t}")
        plt.plot(logh, log10_safe(res["err_cen"]), "-s", label=f"central, t={t}")
        plt.plot(logh, log10_safe(res["err_ext"]), "-^", label=f"extrap, t={t}")

    plt.xlabel(r"$\log_{10}|h|$")
    plt.ylabel(r"$\log_{10}|\text{relative error}|$")
    plt.title(r"Numerical derivative errors for $\exp(t)$")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("exp.png", dpi=300)
    plt.close()

    print("\nPlots saved as cos.png and exp.png")


if __name__ == "__main__":
    main()
