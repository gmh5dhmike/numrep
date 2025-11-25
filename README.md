# numrep

Starter code for numerical representation exercise

# Numerical Limits, Bessel Functions, and Numerical Derivatives

This repository contains code and plots for the numerical exercises on:

1. **Floating-point and integer overflows/underflows**
2. **Spherical Bessel functions and round-off error (upward vs downward recursion)**
3. **Numerical derivatives and error behavior (forward, central, and extrapolated differences)**

All code is Python

---

## 1. Overflows and Underflows

### 1.1 Floating-point underflow/overflow (float32 and float64)

I implemented a simple loop to repeatedly divide by 2 (for underflow) and multiply by 2 (for overflow), using NumPy types:

- `np.float32` (single precision, like C `float`)
- `np.float64` (double precision, like C `double`)

Pseudocode:

```python
under = dtype(1.0)
over = dtype(1.0)
for i in range(N):
    under /= 2
    over *= 2
    # record last nonzero under and last finite over



