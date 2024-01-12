import numpy as np
from typing import Union, Sequence, Callable
from astropy.units import Quantity, Unit
import astropy.units as u
from numpy import ndarray
from scipy.integrate import simps
import sparse

from .spacial_basis_set import SpacialBasisSet
    

class GridSpacialBasisSet(SpacialBasisSet):
    def __init__(
            self, 
            domain: Union[Sequence[float], np.ndarray, Quantity],
            num_grid_points: Union[int, Sequence[int], np.ndarray],
            potential_function: Union[Callable, None] = None,
        ):
        super().__init__()

        self.domain = domain
        self.num_grid_points = num_grid_points
        self.potential_function = potential_function
        self._check_and_process_input("init")

        if self.potential_function is not None:
            grid_points_x, grid_points_y, grid_points_z = self._create_adaptive_grid_points()
            self.grid = np.meshgrid(grid_points_x, grid_points_y, grid_points_z, indexing='ij')
        else:
            grid_points_x = np.linspace(self.domain[0, 0], self.domain[0, 1], self.num_grid_points[0])
            grid_points_y = np.linspace(self.domain[1, 0], self.domain[1, 1], self.num_grid_points[1])
            grid_points_z = np.linspace(self.domain[2, 0], self.domain[2, 1], self.num_grid_points[2])
            self.grid = np.meshgrid(grid_points_x, grid_points_y, grid_points_z, indexing='ij')


    def expand(
            self,
    ):
        pass


    def overlap_matrix(
            self,
    ):
        pass

        


    def _create_adaptive_grid_points(
        self, 
        density_factor: int = 10
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
        grid_points_x = create_grid_points(x_samples, grad_magnitude, self.num_grid_points[0])
        grid_points_y = create_grid_points(y_samples, grad_magnitude, self.num_grid_points[1])
        grid_points_z = create_grid_points(z_samples, grad_magnitude, self.num_grid_points[2])

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
