from typing import Tuple
import warp as wp

@wp.func
def eval_quintic(t: float, p0: float, p1: float, p2: float, p3: float, p4: float, p5: float):
    """
    Helper: Evaluate value and derivative of a 5th degree Bezier polynomial at t.
    Returns (value, derivative)
    """
    # Clamp t to [0, 1] for safety
    t_c = wp.min(wp.max(t, 0.0), 1.0)
    nt = 1.0 - t_c

    # Precompute powers
    nt2 = nt * nt
    nt3 = nt2 * nt
    nt4 = nt2 * nt2
    nt5 = nt4 * nt
    
    t2 = t_c * t_c
    t3 = t2 * t_c
    t4 = t2 * t2
    t5 = t4 * t_c

    # --- Value Calculation (Bernstein Basis) ---
    # B0=(1-t)^5, B1=5(1-t)^4t, B2=10(1-t)^3t^2, B3=10(1-t)^2t^3, B4=5(1-t)t^4, B5=t^5
    b0 = nt5
    b1 = 5.0 * nt4 * t_c
    b2 = 10.0 * nt3 * t2
    b3 = 10.0 * nt2 * t3
    b4 = 5.0 * nt * t4
    b5 = t5
    
    val = b0*p0 + b1*p1 + b2*p2 + b3*p3 + b4*p4 + b5*p5

    # --- Derivative Calculation ---
    # Derivative of 5th degree Bezier is a 4th degree Bezier with weights 5*(P_i+1 - P_i)
    # Basis: B0=(1-t)^4, B1=4(1-t)^3t, B2=6(1-t)^2t^2, B3=4(1-t)t^3, B4=t^4
    d0 = nt4
    d1 = 4.0 * nt3 * t_c
    d2 = 6.0 * nt2 * t2
    d3 = 4.0 * nt * t3
    d4 = t4

    # Control point differences
    q0 = 5.0 * (p1 - p0)
    q1 = 5.0 * (p2 - p1)
    q2 = 5.0 * (p3 - p2)
    q3 = 5.0 * (p4 - p3)
    q4 = 5.0 * (p5 - p4)

    grad = d0*q0 + d1*q1 + d2*q2 + d3*q3 + d4*q4

    return val, grad

@wp.func
def solve_t_at_x(x: float, px0: float, px1: float, px2: float, px3: float, px4: float, px5: float) -> float:
    """
    Uses Newton-Raphson to find t such that BezierX(t) = x
    """
    # Initial guess: linear interpolation
    t = (x - px0) / (px5 - px0 + 1.0e-6)
    t = wp.min(wp.max(t, 0.0), 1.0)

    # Newton-Raphson iterations
    # Warp loops are unrolled in kernels, usually 5-10 iterations are enough for smooth curves
    for i in range(10):
        xt, dxdt = eval_quintic(t, px0, px1, px2, px3, px4, px5)
        
        error = xt - x
        if wp.abs(error) < 1.0e-5:
            break
            
        # Avoid division by zero
        if wp.abs(dxdt) < 1.0e-6:
            break
            
        t = t - error / dxdt
        
        # Clamp to keep inside valid range
        t = wp.min(wp.max(t, 0.0), 1.0)
        
    return t

@wp.func
def calc_bezier(x: float, 
                px0: float, px1: float, px2: float, px3: float, px4: float, px5: float,
                py0: float, py1: float, py2: float, py3: float, py4: float, py5: float) -> float:
    '''
    Calculate the parameter t and value y of a 5th degree Bezier curve at position x.
    Requires both X and Y control points.
    '''
    # 1. Solve for t using X control points
    t = solve_t_at_x(x, px0, px1, px2, px3, px4, px5)
    
    # 2. Calculate y using Y control points and t
    y, dydt = eval_quintic(t, py0, py1, py2, py3, py4, py5)
    
    return y

@wp.func
def calc_bezier_deriv(x: float, 
                      px0: float, px1: float, px2: float, px3: float, px4: float, px5: float,
                      py0: float, py1: float, py2: float, py3: float, py4: float, py5: float) -> float:
    '''
    Calculate the slope dy/dx of a 5th degree Bezier curve at position x.
    dy/dx = (dy/dt) / (dx/dt)
    '''
    # 1. Solve for t
    t = solve_t_at_x(x, px0, px1, px2, px3, px4, px5)
    
    # 2. Get derivatives w.r.t t
    _, dxdt = eval_quintic(t, px0, px1, px2, px3, px4, px5)
    _, dydt = eval_quintic(t, py0, py1, py2, py3, py4, py5)
    
    # 3. Chain rule: dy/dx
    dydx = 0.0
    if wp.abs(dxdt) > 1.0e-8:
        dydx = dydt / dxdt
        
    return dydx