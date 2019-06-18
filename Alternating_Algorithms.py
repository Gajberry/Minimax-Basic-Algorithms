import math
import numpy as np
import copy


def one_step_x(xk, yk, f_func, s_k, dfx_func, tau, beta, alpha, k, extraTest, dfx, alg):
    
    """
    Computes a single step in the x direction

    Parameters
    -------
    xk : float or numpy array
        Current x-value.
    yk : float or numpy array
        Current y-value. 
    f_func : callable
        Function f(x,y).
    s_k : numpy array
        Search direction.
    dfx_func : callable
        Gradient of f in the x direction.
    tau : float
        Backtracking multiplier.
    beta : float
        Backtraking constant.
    alpha : float
        Initial stepsize.
    k : integer
        Iteration number.
    extraTest : boolean
        Indicates whether the extra stepsize test should be carried out.
    dfx : float
        dfx_func evaluated at (xk, yk).
    alg : string
        The algorithm used.

    Returns
    -------
    xk+alpha_i*s_k : float or numpy array
        The next value of x.

    """

    global rho # Necessary for those algorithms using rho
    
    alpha_i = alpha
    
    # Assigning values so that they do not need to be repeatedly calculated:
    gT = dfx
    dotprodBeta = beta * np.dot(gT, s_k)
    fVal = f_func(xk, yk)
    
    if alg != 'Quasi-Newton':
        # Backtracking step size algorithm
        while f_func(xk + alpha_i * s_k, yk) > fVal + alpha_i * dotprodBeta or (extraTest and np.linalg.norm(alpha_i * s_k) > 1 / (math.sqrt(k))):
        
            alpha_i *= tau
    else:
        found = False
        while found == False:
            # Backtracking step size algorithm
            while f_func(xk + alpha_i * s_k, yk) > fVal + alpha_i * dotprodBeta or (extraTest and np.linalg.norm(alpha_i * s_k) > 1 / (math.sqrt(k))):
            
                alpha_i *= tau
            last_x_Armi = xk + alpha_i * s_k
            last_x_Fail_armi = xk + alpha_i * s_k / tau
            
            # Check if the stepsize satisfies the Wolfe conditions
            if abs(np.dot(s_k, dfx_func(last_x_Armi, yk))) < abs(np.dot(s_k, gT)):
                found = True
            else:
                # If not, repeat with a new initial point
                minx = last_x_Armi
                maxx = last_x_Fail_armi
                alpha_i = (1 - tau) * alpha_i
                xk = minx
    return xk + alpha_i * s_k


def one_step_y(xk, yk, f_func, s_k, dfy_func, tau, beta, alpha, k, extraTest, dfy, alg):
    
    """
    Computes a single step in the y direction

    Parameters
    -------
    xk : float or numpy array
        Current x-value.
    yk : float or numpy array
        Current y-value. 
    f_func : callable
        Function f(x,y).
    s_k : numpy array
        Search direction.
    dfy_func : callable
        Gradient of f in the y direction.
    tau : float
        Backtracking multiplier.
    beta : float
        Backtraking constant.
    alpha : float
        Initial stepsize.
    k : integer
        Iteration number.
    extraTest : boolean
        Indicates whether the extra stepsize test should be carried out.
    dfy : float
        dfy_func evaluated at (xk,yk).
    alg : string
        The algorithm used.

    Returns
    -------
    yk+alpha_i*s_k : float or numpy array
        The next value of y.

    """
    
    # Identical to one_step_x except making sure that f increases rather than decreases
    global rho
    alpha_i = alpha
    gT = -dfy
    dotprodBeta = beta * np.dot(gT, s_k)
    fVal = f_func(xk, yk)
    if alg != 'Quasi-Newton':
        while -f_func(xk, yk + alpha_i * s_k) > -fVal + alpha_i * dotprodBeta or (extraTest and np.linalg.norm(alpha_i * s_k) > 1 / (math.sqrt(k))):
        
            alpha_i *= tau
    else:
        found = False
        while found == False:
            while -f_func(xk, yk + alpha_i * s_k) > -fVal + alpha_i * dotprodBeta or (extraTest and np.linalg.norm(alpha_i * s_k) > 1 / (math.sqrt(k))):
            
                alpha_i *= tau
            last_y_Armi = yk + alpha_i * s_k
            last_y_Fail_armi = yk + alpha_i * s_k / tau
            if abs(np.dot(s_k, dfy_func(xk, last_y_Armi))) < abs(np.dot(s_k, gT)):
                found = True
            else:
                miny = last_y_Armi
                maxy = last_y_Fail_armi
                alpha_i = (1 - tau) * alpha_i
                yk = miny
    return yk + alpha_i * s_k


def Hessian_approx(Ck, deltak, gammak, singledim, var):
    
    """
    Computes the inverse of the next Hessian approximation.

    Parameters
    -------
    Ck : numpy array
        Inverse of the previous Hessian approximation.
    deltak : float or numpy array
        Difference in variable.
    gammak : float or numpy array
        Difference in gradiant.
    singledim: boolean
        Whether or not the variable in question is in a single dimension.
    var: 1 or -1
        1 if var=x, -1 if var=y

    Returns
    -------
    Ck2 : numpy array
        Inverse of the next Hessian approximation.

    """

    if singledim:
        Ck2 = deltak / gammak
    else:
        # Assigning values so they do not need to be repeatedly computed.
        Dot = np.dot(gammak, deltak)
        A = np.outer(deltak, gammak) / Dot
        I = np.identity((len(Ck)))
        Ck2 = np.matmul(np.matmul(I - A, Ck), I - np.transpose(A)) + np.outer(deltak, deltak) / Dot
    return Ck2


def descent_direction(xk, yk, algz, singledimz, dfz_func, ddfz_func, Ckz, var, dfz):
    
    """
    Computes the search direction in a specific variable.

    Parameters
    -------
    xk : float or numpy array
        Current x-value.
    yk : float or numpy array
        Current y-value. 
    algz : string
        The algorithm used.
    dfz_func : callable
        Gradient of f in the direction of the variable in question.
    ddfz_func : callable
        Hessian of f in the variable in question.
    Ck : numpy array
        Inverse of the previous Hessian approximation.
    var: 1 or -1
        1 if var=x, -1 if var=y
    dfz : float
        dfz_func evaluated at (xk,yk).


    Returns
    -------
    s_kz : numpy array
        The search direction.

    """

    if algz == 'Newton':
        if singledimz:
            s_kz = -dfz / ddfz_func(xk, yk)
        else:
            s_kz = -np.matmul(np.linalg.inv(ddfz_func(xk, yk)), dfz)
    elif algz == 'Quasi-Newton':
        if singledimz:
            s_kz = -dfz * Ckz
        else:
            s_kz = -np.matmul(Ckz, dfz)
    elif algz == 'Descent':
        s_kz = -ind * dfz
    else:
        # In case the algorithm name is not Newton, Quasi-Newton or Descent:
        print('Wrong algorithm name')
        print(alg[0])
    return s_kz


def Alternating_alg(x0, y0, f_func, dfx_func, dfy_func, ddfx_func=None, ddfy_func=None, tau=0.5, beta=0.0001,epsilon=10 ** (-10), alpha=0.75, alg=['Newton', 'Newton'], maxk=1000, extraTest=False):

    """
    Alternating algorithm

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
    ddfx_func : callable, optional (though necessary for Newton algorithm)
        Hessian of f in x.
    ddfy_func : callable, optional (though necessary for Newton algorithm)
        Hessian of f in y.
    tau : float, optional
        Backtracking multiplier.
    beta : float, optional
        Backtraking constant.
    epsilon : float, optional
        Stopping criteria constant.
    alpha : float, optional
        Initial stepsize.
    alg : list of length 2, optional
        List of algorithms for x and y.
    maxk : integer, optional
        Maximum number of iterations.
    extraTest : boolean, optional
        Indicates whether the extra stepsize test should be carried out.

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


    xk = copy.deepcopy(x0)
    yk = copy.deepcopy(y0)
    
    # Determine whether x,y values are floats or arrays:
    if isinstance(dfx_func(xk, yk), (int, float)):
        singledimx = True
    elif isinstance(dfx_func(xk, yk), (np.ndarray)):
        singledimx = False
    else:
        print(type(dfx_func))
        singledimx = False
    
    if isinstance(dfy_func(xk, yk), (int, float)):
        singledimy = True
    elif isinstance(dfy_func(xk, yk), (np.ndarray)):
        singledimy = False
    else:
        print(type(dfy_func))
        singledimy = False
    
    # Generate initial values and lists
    k = 1
    xkvals = [xk]
    ykvals = [yk]
    ind_coords = [[xk, yk]]
    kvals = [0]
    
    # Generate inverse of Hessian approximations if required:
    if alg[0] == 'Quasi-Newton':
        if singledimx:
            Ckx = alpha / np.linalg.norm(dfx_func(xk, yk))
        else:
            Ckx = alpha / np.linalg.norm(dfx_func(xk, yk)) * np.identity(len(xk))
    else:
        Ckx = None
    if alg[1] == 'Quasi-Newton':
        if singledimy:
            Cky = -alpha / np.linalg.norm(dfy_func(xk, yk))
        else:
            Cky = - alpha / np.linalg.norm(dfy_func(xk, yk)) * np.identity(len(yk))
    else:
        Cky = None
        
    # Stopping criteria
    while k < maxk and (singledimx and abs(dfx_func(xk, yk)) > epsilon or abs(dfy_func(xk, yk)) > epsilon) or (not singledimx and (np.linalg.norm(dfx_func(xk, yk)) < 1 / epsilon or np.linalg.norm(dfy_func(xk, yk)) < 1 / epsilon))):
    
        # Perform one step in x:
        dfx = dfx_func(xk, yk)
        s_kx = descent_direction(xk, yk, alg[0], singledimx, dfx_func, ddfx_func, Ckx, 1, dfx)
        if (singledimx and s_kx != 0) or (not singledimx and np.linalg.norm(s_kx) != 0):
            new_x = one_step_x(xk, yk, f_func, s_kx, dfx_func, tau, beta, alpha, k, extraTest, dfx, alg[0])
        else:
            new_x = xk
            
        # Perform one step in y:
        dfy = dfy_func(new_x, yk)
        s_ky = descent_direction(new_x, yk, alg[1], singledimy, dfy_func, ddfy_func, Cky, -1, dfy)
        if (singledimy and s_ky != 0) or (not singledimy and np.linalg.norm(s_ky) != 0):
            new_y = one_step_y(new_x, yk, f_func, s_ky, dfy_func, tau, beta, alpha, k, extraTest, dfy, alg[1])
        else:
            new_y = yk
        
        # Compute inverse of new Hessian approximations:
        if alg[0] == 'Quasi-Newton':
            deltakx = new_x - xk
            gammakx = dfx_func(new_x, new_y) - dfx
            if (singledimx and deltakx * gammakx > 0) or (not singledimx and np.dot(deltakx, gammakx) > 0):
                Ckx = Hessian_approx(Ckx, deltakx, gammakx, singledimx, 'x')
            if alg[1] == 'Quasi-Newton':
                deltaky = new_y - yk
            gammaky = dfy_func(new_x, new_y) - dfy
            if (singledimy and deltaky * gammaky < 0) or (not singledimy and np.dot(deltaky, gammaky) < 0):
                Cky = Hessian_approx(Cky, deltaky, gammaky, singledimy, 'y')
        
        # Add to lists and update values:
        ind_coords.append([new_x, yk])
        ind_coords.append([new_x, new_y])
        xk = new_x
        yk = new_y
        xkvals.append(new_x)
        ykvals.append(new_y)
        k += 1
        kvals.append(k)
    return xk, yk, xkvals, ykvals, ind_coords, kvals
