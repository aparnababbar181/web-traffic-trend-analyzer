import numpy as np

# Replicate exactly the two interpolation functions from the main file

def lagrange_interpolate(x_known, y_known, x_query):
    x_known = np.array(x_known, dtype=float)
    y_known = np.array(y_known, dtype=float)
    n = len(x_known)
    scalar_input = np.isscalar(x_query)
    x_query = np.atleast_1d(np.array(x_query, dtype=float))
    result = np.zeros_like(x_query)
    for i in range(n):
        Li = np.ones_like(x_query)
        for j in range(n):
            if j == i:
                continue
            Li *= (x_query - x_known[j]) / (x_known[i] - x_known[j])
        result += y_known[i] * Li
    return float(result[0]) if scalar_input else result

def _divided_diff_table(x, y):
    n = len(x)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table

def newton_dd(x_known, y_known, x_query):
    x_known = np.array(x_known, dtype=float)
    y_known = np.array(y_known, dtype=float)
    n = len(x_known)
    table = _divided_diff_table(x_known, y_known)
    coeffs = table[0, :]
    scalar_input = np.isscalar(x_query)
    x_query = np.atleast_1d(np.array(x_query, dtype=float))
    result = np.zeros_like(x_query)
    for i in range(n):
        term = np.ones_like(x_query) * coeffs[i]
        for j in range(i):
            term *= (x_query - x_known[j])
        result += term
    return float(result[0]) if scalar_input else result

# User's data
days     = [1, 3, 5, 7, 9, 12]
visitors = [100, 150, 220, 300, 390, 500]
last     = visitors[-1]

future_days = np.arange(13, 23)

lag = lagrange_interpolate(days, visitors, future_days)
ndd = newton_dd(days, visitors, future_days)

print(f"{'Day':<6} {'Lagrange':>18} {'Newton DD':>18}  {'Trend (Lagrange)':>16}")
print("-" * 64)
for d, l, n in zip(future_days, lag, ndd):
    chg = ((l - last) / last) * 100
    print(f"Day {int(d):<3} {int(l):>18,} {int(n):>18,}  {chg:>+15.1f}%")
