import numpy as np
from scipy.interpolate import RectBivariateSpline

def legacy_interp2d_wrapper(x, y, z, kind='linear'):
    """
    Robust wrapper for RectBivariateSpline that mimics interp2d behavior:
    1. Handles unsorted definition grids.
    2. Handles Transpose logic (z shapes).
    3. Handles both Grid (Cartesian) and Point-wise (Element-wise) queries.
    """
    # --- SETUP (Same as before) ---
    x, y, z = np.array(x), np.array(y), np.array(z)
    
    # Sort definition axes
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        z = z[:, idx]
    if np.any(np.diff(y) < 0):
        idy = np.argsort(y)
        y = y[idy]
        z = z[idy, :]

    # Transpose z for RectBivariateSpline (expects z[x, y])
    z_new = z.T
    
    # Initialize Spline (Linear)
    if kind == 'linear':
        rbs = RectBivariateSpline(x, y, z_new, kx=1, ky=1)
    elif kind == 'cubic':
        rbs = RectBivariateSpline(x, y, z_new, kx=3, ky=3)
    else:
        rbs = RectBivariateSpline(x, y, z_new, kx=1, ky=1)

    # --- CALL WRAPPER (Updated for N-D arrays) ---
    def call_wrapper(new_x, new_y):
        new_x = np.asarray(new_x)
        new_y = np.asarray(new_y)
        
        # Scenario A: Scalar inputs -> Return Scalar
        if new_x.ndim == 0 and new_y.ndim == 0:
            return rbs(new_x, new_y, grid=False)

        # Scenario B: "Grid" behavior (classic interp2d default)
        # This is used when x and y have different lengths and likely define axes.
        # interp2d typically defaulted to this if inputs were 1D vectors.
        if new_x.ndim == 1 and new_y.ndim == 1 and new_x.size != new_y.size:
             result = rbs(new_x, new_y, grid=True)
             return result.T # Transpose back to (ny, nx)

        # Scenario C: "Element-wise" / High-Dimension behavior
        # If inputs are arrays of same shape (e.g. 50x50x50), user wants mapped points.
        # RectBivariateSpline requires flattened 1D inputs for grid=False.
        flat_x = new_x.ravel()
        flat_y = new_y.ravel()
        
        # grid=False means: evaluate at pairs (x[0], y[0]), (x[1], y[1])...
        result_flat = rbs(flat_x, flat_y, grid=False)
        
        # Reshape result back to the original input shape (e.g., 50x50x50)
        return result_flat.reshape(new_x.shape)
        
    return call_wrapper
