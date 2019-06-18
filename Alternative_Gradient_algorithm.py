import copy
import numpy as np


def squig_f(xk, yk, dfx_func, dfy_func, ddfx_func, ddfy_func, ddfxy_func):
    """
    Computes the search direction.

    Parameters
    -------
    xk : float or numpy array
        Current x-value.
    yk : float or numpy array
        Current y-value.
    f_func : callable
        Function f(x,y).
    dfx_func : callable
        Gradient of f in the x direction.
    dfy_func : callable
        Gradient of f in the y direction.
    ddfx_func : callable, optional (though necessary for Newton algorithm)
        Hessian of f in x.
    ddfy_func : callable, optional (though necessary for Newton algorithm)
        Hessian of f in y.

    Returns
    -------
    s_k : numpy array

    """

    global singledim
    if singledim:
        s_k = np.array([(dfx_func(xk, yk) * ddfx_func(xk, yk) + dfy_func(xk, yk) * ddfxy_func(xk, yk)) / NVal(xk, yk,
                                                                                                              dfx_func,
                                                                                                              dfy_func),
                        (dfy_func(xk, yk) * ddfy_func(xk, yk) + dfx_func(xk, yk) * ddfxy_func(xk, yk)) / NVal(xk, yk,
                                                                                                              dfx_func,
                                                                                                              dfy_func)])
        return s_k
    else:
        s_k = np.array([(np.matmul(dfx_func(xk, yk), ddfx_func(xk, yk)) + np.matmul(ddfxy_func(xk, yk), np.transpose(
            dfy_func(xk, yk)))) / NVal(xk, yk, dfx_func, dfy_func), (
                                    np.matmul(dfy_func(xk, yk), ddfy_func(xk, yk)) + np.matmul(dfx_func(xk, yk),
                                                                                               ddfxy_func(xk,
                                                                                                          yk))) / NVal(
            xk, yk, dfx_func, dfy_func)])
        return s_k


def NVal(xk, yk, dfx_func, dfy_func):
    """
    Computes the Euclidean norm of the gradient.

    Parameters
    -------
    xk : float or numpy array
        Current x-value.
    yk : float or numpy array
        Current y-value.
    f_func : callable
        Function f(x,y).
    dfx_func : callable
        Gradient of f in the x direction.
    dfy_func : callable
        Gradient of f in the y direction.

    Returns
    -------
    float : Norm value of the gradient

    """

    global singledim
    result = 0
    s_x = dfx_func(xk, yk)
    s_y = dfy_func(xk, yk)
    if not singledim:
        # Iterates through each component
        for i in range(len(s_x)):
            result += s_x[i] ** 2
        for i in range(len(s_y)):
            result += s_y[i] ** 2
        return np.sqrt(result)
    else:
        return np.sqrt(s_x ** 2 + s_y ** 2)


def gradient_based2(xk, yk, dfx_func, dfy_func, ddfx_func, ddfy_func, ddfxy_func, tau=0.5, beta=0.001,
                    epsilon=10 ** (-10), alpha=0.75):
    """
    Computes a single step in the x direction

    Parameters
    -------
    x0 : float or numpy array
        Initial x-value.
    y0 : float or numpy array
        Initial y-value.
    f_func : callable
        Function f(x,y).
    dfx_func : callable
        Gradient of f in the x direction.
    dfy_func : callable
        Gradient of f in the y direction.
    ddfx_func : callable
        Hessian of f in x.
    ddfy_func : callable
        Hessian of f in y.
    ddfxy_func: callable
        Second derivative of f in x and then y.
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
    global singledim
    k = 0
    xkvals = [xk]
    ykvals = [yk]
    ind_coords = [[xk, yk]]
    kvals = [0]
    s_k = epsilon + 1

    # Determine type of x and y
    if isinstance(dfx_func(xk, yk), (int, float)):
        singledim = True
    elif isinstance(dfx_func(xk, yk), (np.ndarray)):
        singledim = False
    else:
        print(type(dfx_func))
        singledim = False

    # Stopping criteria:
    while NVal(xk, yk, dfx_func, dfy_func) > epsilon and k < 10000:
        k += 1
        NValOld = np.linalg.norm(NVal(xk, yk, dfx_func, dfy_func))
        alpha_i = copy.deepcopy(alpha)

        # Perform backtracking algorithm:
        while NVal(xk + alpha_i * s_k[0], yk + alpha_i * s_k[1], dfx_func, dfy_func) ** 2 / 2 > (
                1 / 2 - beta * alpha_i) * NValOld ** 2:
            alpha_i *= tau

        # Update values:
        xk = xk + alpha_i * s_k[0]
        yk = yk + alpha_i * s_k[1]
        kvals.append(k)
        xkvals.append(xk)
        ykvals.append(yk)
        ind_coords.append([xk, yk])
    return xk, yk, xkvals, ykvals, ind_coords, kvals