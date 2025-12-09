import numpy as np
from scipy.interpolate import RectBivariateSpline

def legacy_interp2d_wrapper(x, y, z, kind='linear'):
    """
    Universal wrapper to make RectBivariateSpline behave exactly like interp2d.
    """
    # 1. SETUP: Handle 'definition' data (x, y, z)
    x, y, z = np.array(x), np.array(y), np.array(z)
    
    # Sort inputs (RectBivariateSpline requires strictly ascending axes)
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        z = z[:, idx]
    if np.any(np.diff(y) < 0):
        idy = np.argsort(y)
        y = y[idy]
        z = z[idy, :]

    # Transpose Z (interp2d expected z[y, x], RBS expects z[x, y])
    z_new = z.T
    
    # Create Spline (Linear)
    if kind == 'linear':
        rbs = RectBivariateSpline(x, y, z_new, kx=1, ky=1)
    elif kind == 'cubic':
        rbs = RectBivariateSpline(x, y, z_new, kx=3, ky=3)

    # 2. CALL: Handle 'query' data (new_x, new_y)
    def call_wrapper(new_x, new_y):
        flat_x = np.atleast_1d(new_x).ravel()
        flat_y = np.atleast_1d(new_y).ravel()
        result = rbs(flat_x, flat_y, grid=True)
        
        # Transpose back to match expected shape (N_y, N_x)
        result = result.T
        
        return result
        
    return call_wrapper
