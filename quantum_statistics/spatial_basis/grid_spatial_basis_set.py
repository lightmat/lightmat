import numpy as np
from typing import Union, Sequence, Callable
from astropy.units import Quantity, Unit
import astropy.units as u
from numpy import ndarray
from scipy.integrate import simps
import sparse

from .spatial_basis_set import SpatialBasisSet
    

class GridSpatialBasisSet(SpatialBasisSet):
    def __init__(
            self, 
            domain: Union[Sequence[float], np.ndarray, Quantity] = [(-100, 100), (-100, 100), (-100, 100)] * u.um,
            num_grid_points: Union[int, Sequence[int], np.ndarray] = (101, 101, 101),
            potential_function: Union[Callable, None] = None,
        ):
        """Initialize a grid spatial basis set.
        
           Args:    
               domain: Domain of the basis set in [um], if sequence of length 2, the same domain is assumed for each axis. If
                       sequence of length 3, a different domain is assumed for each axis. If Quantity, the domain is assumed
                       to be in units of length. Default: [(-100, 100), (-100, 100), (-100, 100)] * u.um
               num_grid_points: Number of grid points for each axis, if int, the same number of grid points is assumed for
                                each axis. If sequence of length 3, a different number of grid points is assumed for each
                                axis. Default: (101, 101, 101)
               potential_function: Potential function to use for adaptive grid point creation. If None, an equidistant grid 
                                   is created. Default: None
        """
        super().__init__()

        self.domain = domain
        self.num_grid_points = num_grid_points
        self.potential_function = potential_function
        self._check_and_process_input("init")

        if self.potential_function is not None:
            self.grid_points_x, self.grid_points_y, self.grid_points_z = self._create_adaptive_grid_points()
        else:
            self.grid_points_x = np.linspace(self.domain[0, 0], self.domain[0, 1], self.num_grid_points[0]) * u.um
            self.grid_points_y = np.linspace(self.domain[1, 0], self.domain[1, 1], self.num_grid_points[1]) * u.um
            self.grid_points_z = np.linspace(self.domain[2, 0], self.domain[2, 1], self.num_grid_points[2]) * u.um

        self.num_basis_funcs = self.num_grid_points[0] * self.num_grid_points[1] * self.num_grid_points[2]

        X, Y, Z = np.meshgrid(self.grid_points_x.value, self.grid_points_y.value, self.grid_points_z.value, indexing='ij')
        self.grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T * u.um # Shape: (num_basis_funcs, 3)
        self.volumes = self._calculate_volumes() # Shape: (num_basis_funcs,)



    def get_coeffs(
            self,
            f: Callable,
    ) -> ndarray:
        """Calculate the coefficients of the basis functions for the function f.
        
           Args:
               f (Callable): Function to calculate the coefficients for.

           Returns:
               coeffs (ndarray): 1d array of coefficients for each basis function.
        """
        # Calculate the function values at each grid point
        f_vals = f(self.grid_points[:, 0], self.grid_points[:, 1], self.grid_points[:, 2])
        return f_vals # Shape: (num_basis_funcs,)
    

    def expand_coeffs(
            self,
            coeffs: ndarray,
            x: Union[float, Sequence[float], ndarray, Quantity],
            y: Union[float, Sequence[float], ndarray, Quantity],
            z: Union[float, Sequence[float], ndarray, Quantity],
    ) -> Union[float, ndarray]:
        """Expand the coefficients in the basis set and return the function evaluated at the given position(s).
           You can either provide an individual point or a meshgrid of points for x, y, z.
        
           Args:
               coeffs (ndarray): 1d array of coefficients for each basis function.
                x: x coordinate(s) to evaluate the function at.
                y: y coordinate(s) to evaluate the function at.
                z: z coordinate(s) to evaluate the function at.

           Returns:
               Evaluated function at the given position(s) as array of same shape as x, y, and z.
        """
        # Convert inputs to numpy arrays for vectorized operations
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        if np.min(x) < self.domain[0, 0] or np.max(x) > self.domain[0, 1]:
            raise ValueError(f"x coordinate(s) must be within the x domain [{self.domain[0, 0]}, {self.domain[0, 1]}].")
        if np.min(y) < self.domain[1, 0] or np.max(y) > self.domain[1, 1]:
            raise ValueError(f"y coordinate(s) must be within the y domain [{self.domain[1, 0]}, {self.domain[1, 1]}].")
        if np.min(z) < self.domain[2, 0] or np.max(z) > self.domain[2, 1]:
            raise ValueError(f"z coordinate(s) must be within the z domain [{self.domain[2, 0]}, {self.domain[2, 1]}].")

        # Find indices of the nearest grid points in each dimension
        mid_points_x = (self.grid_points_x[:-1] + self.grid_points_x[1:]) / 2
        mid_points_y = (self.grid_points_y[:-1] + self.grid_points_y[1:]) / 2
        mid_points_z = (self.grid_points_z[:-1] + self.grid_points_z[1:]) / 2
        idx_x = np.digitize(x, mid_points_x.value)
        idx_y = np.digitize(y, mid_points_y.value)
        idx_z = np.digitize(z, mid_points_z.value)

        # Retrieve corresponding coefficients (assuming coeffs is a flattened array of the 3D grid)
        coeffs_values = coeffs[idx_x * self.num_grid_points[1] * self.num_grid_points[2] +
                               idx_y * self.num_grid_points[2] +
                               idx_z]
        return coeffs_values
        

    def gradient(
            self, 
            coeffs,
    ) -> ndarray:
        """Calculate the gradient of the function f expanded in the basis set.
        
           Args:
               coeffs (ndarray): 1d array of coefficients for each basis function.

           Returns:
               gradient (ndarray): 3d array of the gradient coefficients for each basis function.
        """
        # Reshape coeffs into a 3D array
        coeffs_3d = coeffs.reshape((self.num_grid_points[0], self.num_grid_points[1], self.num_grid_points[2]))

        # Calculate gradients
        if self.potential_function is None: # equidistant grid
            # Regular grid: Use np.gradient without specifying spacings
            gradients = np.gradient(coeffs_3d)
        else:
            # Adaptive grid: Pass the grid spacings for each dimension
            spacings = [self.grid_points_x.value, self.grid_points_y.value, self.grid_points_z.value]
            gradients = np.gradient(coeffs_3d, *spacings)

        # Combine gradients into a single array
        gradient = np.stack(gradients, axis=-1)

        return gradient.reshape((self.num_basis_funcs, 3))
    

    def laplacian(
            self,
            coeffs,
    ) -> ndarray:
        """Calculate the laplacian of the function f expanded in the basis set.
        
           Args:
               coeffs (ndarray): 1d array of coefficients for each basis function.

           Returns:
               laplacian (ndarray): 1d array of the laplacian coefficients for each basis function.
        """
        # Reshape coeffs into a 3D array
        coeffs_3d = coeffs.reshape((self.num_grid_points[0], self.num_grid_points[1], self.num_grid_points[2]))

        # Calculate second derivatives
        if self.potential_function is None:  # equidistant grid
            # Use np.gradient twice to calculate second derivatives
            d2x = np.gradient(np.gradient(coeffs_3d, axis=0), axis=0)
            d2y = np.gradient(np.gradient(coeffs_3d, axis=1), axis=1)
            d2z = np.gradient(np.gradient(coeffs_3d, axis=2), axis=2)
        else:  # adaptive grid
            # Calculate second derivatives with varying spacings
            spacings = [self.grid_points_x.value, self.grid_points_y.value, self.grid_points_z.value]
            d2x = np.gradient(np.gradient(coeffs_3d, spacings[0], axis=0), spacings[0], axis=0)
            d2y = np.gradient(np.gradient(coeffs_3d, spacings[1], axis=1), spacings[1], axis=1)
            d2z = np.gradient(np.gradient(coeffs_3d, spacings[2], axis=2), spacings[2], axis=2)

        # The laplacian is the sum of the second derivatives in each direction
        laplacian_coeffs = d2x + d2y + d2z

        # Flatten the laplacian to return a 1D array
        return laplacian_coeffs.ravel()
    

    def integral(
            self,
            coeffs,
    ) -> float:
        """Calculate the integral over the entire domain of the function f expanded in the basis set.
        
           Args:
               coeffs (ndarray): 1d array of coefficients for each basis function.

           Returns:
               integral (float): Integral of the function.
        """
        return np.sum(coeffs * self.volumes)


    def _calculate_volumes(
            self,
        ) -> ndarray:
        """Calculate the volumes of each grid point as cube with half spacing to each neighbor.
        
           Returns:
               volumes (ndarray): 1d array of volumes for each grid point.
        """
        # Add the half distances to right and left grid neighbors 
        right_half_diffs_x = np.append(np.diff(self.grid_points_x), 0) / 2  
        left_half_diffs_x = np.insert(right_half_diffs_x, 0, 0)[:-1]  
        half_spacings_x = left_half_diffs_x + right_half_diffs_x
        right_half_diffs_y = np.append(np.diff(self.grid_points_y), 0) / 2
        left_half_diffs_y = np.insert(right_half_diffs_y, 0, 0)[:-1]
        half_spacings_y = left_half_diffs_y + right_half_diffs_y
        right_half_diffs_z = np.append(np.diff(self.grid_points_z), 0) / 2
        left_half_diffs_z = np.insert(right_half_diffs_z, 0, 0)[:-1]
        half_spacings_z = left_half_diffs_z + right_half_diffs_z
        
        # Calculate the total volume for each grid point
        volumes = half_spacings_x[:, None, None] * half_spacings_y[None, :, None] * half_spacings_z[None, None, :]
        return volumes.flatten()


    def _create_adaptive_grid_points(
        self, 
        density_factor: int = 20
    ) -> tuple:
        # Sample the domain
        x_samples = np.linspace(self.domain[0, 0], self.domain[0, 1], self.num_grid_points[0])
        y_samples = np.linspace(self.domain[1, 0], self.domain[1, 1], self.num_grid_points[1])
        z_samples = np.linspace(self.domain[2, 0], self.domain[2, 1], self.num_grid_points[2])

        # Create a grid and evaluate the potential function
        X, Y, Z = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')
        V = self.potential_function(X, Y, Z)

        # Calculate the gradient magnitude
        grad_x, grad_y, grad_z = np.gradient(V, x_samples, y_samples, z_samples)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # Function to create adaptive grid points
        def create_grid_points(samples, grad_magnitude, num_grid_points):
            # Calculate the density distribution for grid points
            grad_agg = np.max(np.max(grad_magnitude, axis=1), axis=1)
            normalized_grad = grad_agg / np.max(grad_agg)
            density = normalized_grad * density_factor + 1
            density /= np.sum(density)
            cumulative_density = np.cumsum(density)

            # Determine grid point positions
            return np.interp(np.linspace(0, 1, num_grid_points), cumulative_density, samples)

        # Create grid points for each axis
        grid_points_x = create_grid_points(x_samples, grad_magnitude, self.num_grid_points[0]) * u.um
        grid_points_y = create_grid_points(y_samples, grad_magnitude, self.num_grid_points[1]) * u.um
        grid_points_z = create_grid_points(z_samples, grad_magnitude, self.num_grid_points[2]) * u.um

        assert len(grid_points_x) == self.num_grid_points[0] and \
               len(grid_points_y) == self.num_grid_points[1] and \
               len(grid_points_z) == self.num_grid_points[2]   


        return grid_points_x, grid_points_y, grid_points_z


    def _check_and_process_input(
            self,
            which_method: str,
    ):
        if which_method == "init":
            # Check that the domain is either a sequence of lenth 3 containing sequences of length 2 or
            # a sequence of length 2, then domain becomes [domain, domain, domain]
            if isinstance(self.domain, Quantity):
                if self.domain.unit.is_equivalent(u.um):
                    self.domain = self.domain.to(u.um).value
                else:
                    raise u.UnitsError("domain must be in units of length.")
            if isinstance(self.domain, (Sequence, np.ndarray)):
                if len(self.domain) == 2:
                    if all(isinstance(d, (float, int)) for d in self.domain):
                        self.domain = np.array([np.asarray(self.domain), np.asarray(self.domain), \
                                                np.asarray(self.domain)])
                    else:
                        raise TypeError("domain must be a sequence of length 3 containing sequences of length 2.")
                if len(self.domain) == 3:
                    if all(isinstance(d, (Sequence, np.ndarray)) for d in self.domain):
                        if all(len(d) == 2 for d in self.domain):
                            if all(isinstance(d_i, (float, int)) for d in self.domain for d_i in d):
                                self.domain = np.array([np.asarray(d) for d in self.domain])
                            else:
                                raise TypeError("domain must be a sequence of length 3 containing sequences of length 2.")
                        else:
                            raise ValueError("domain must be a sequence of length 3 containing sequences of length 2.")
                    else:
                        raise TypeError("domain must be a sequence of length 3 containing sequences of length 2.")
                else:
                    raise ValueError("domain must be a sequence of length 3 containing sequences of length 2.")

            
            # Check that num_grid_points is a sequence of length 3 containing ints or a single int, then num_grid_points
            # becomes [num_grid_points, num_grid_points, num_grid_points]
            if isinstance(self.num_grid_points, int):
                self.num_grid_points = np.array([self.num_grid_points, self.num_grid_points, self.num_grid_points])
            elif isinstance(self.num_grid_points, (Sequence, np.ndarray)):
                if all(isinstance(n, int) for n in self.num_grid_points):
                    self.num_grid_points = np.array(self.num_grid_points)
                else:
                    raise TypeError("num_grid_points must be an int or a sequence of ints.")
            else:
                raise TypeError("num_grid_points must be an int or a sequence of ints.")
            
            # Check that the potential_function is a Callable or None
            if self.potential_function is None or isinstance(self.potential_function, Callable):
                self.potential_function = self.potential_function
            else:
                raise TypeError("potential_function must be a Callable or None.")
