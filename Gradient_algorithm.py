import copy
import numpy as np


def squig_f(xk, yk, dfx_func, dfy_func):
    """
    Computes the search direction.

    Parameters
    -------
    xk : float or numpy array
        Current x-value.
    yk : float or numpy array
        Current y-value.
    dfx_func : callable
        Gradient of f in the y direction.
    dfy_func : callable
        Gradient of f in the x direction.

    Returns
    -------
    s_kz : numpy array
        The search direction.

    """
    s_kz = np.array([dfx_func(xk, yk), -dfy_func(xk, yk)])
    return s_kz


def gradient_based(xk, yk, dfx_func, dfy_func, tau=0.5, beta=0.001, epsilon=10 ** (-10), alpha=0.75):
    """
    Gradient algorithm

    Parameters
    -------
    xk : float or numpy array
        Initial x-value.
    yk : float or numpy array
        Initial y-value.
    dfx_func : callable
        Gradient of f in the x direction.
    dfy_func : callable
        Gradient of f in the y direction.
    tau : float, optional
        Backtracking multiplier.
    beta : float, optional
        Backtraking constant.
    epsilon : float, optional
        Stopping criteria constant.
    alpha : float, optional
        Initial stepsize.

    Returns
    -------
    xk : float
        Final x value.
    yk : float
        Final y value.
    xkvals : list
        List of all x values.
    ykvals : list
        List of all y values.
    ind_coords : list
        List of coordinates reached.
    kvals
        List of k values.

    """

    # Generate initial values and lists:
    k = 0
    global singledim
    xkvals = [xk]
    ykvals = [yk]
    ind_coords = [[xk, yk]]
    kvals = [k]
    s_k = epsilon + 1  # Initial value to allow for the stopping criteria to be False.

    # Check dimensions of x and y:
    if isinstance(dfx_func(xk, yk), (int, float)):
        singledim = True
    elif isinstance(dfx_func(xk, yk), (np.ndarray)):
        singledim = False
    else:
        print(type(dfx_func))
        singledim = False

    # Stopping criteria:
    while (np.linalg.norm(s_k) > epsilon) and k < 10000:
        k += 1
        s_k = -squig_f(xk, yk, dfx_func, dfy_func)  # Search direction
        alpha_i = copy.deepcopy(alpha)

        # Perform backtracking algorithm:
        while np.linalg.norm(squig_f(xk + alpha_i * s_k[0], yk + alpha_i * s_k[1], dfx_func, dfy_func)) ** 2 / 2 > (
                1 / 2 - beta * alpha_i) * np.linalg.norm(s_k) ** 2:
            alpha_i *= tau

        # Update lists and values:
        xk = xk + alpha_i * s_k[0]
        yk = yk + alpha_i * s_k[1]
        kvals.append(k)
        xkvals.append(xk)
        ykvals.append(yk)
        ind_coords.append([xk, yk])
    return xk, yk, xkvals, ykvals, ind_coords, kvals
