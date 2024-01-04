import numpy as np
from scipy.sparse import csr_matrix
from typing import Union, Sequence, Callable
from astropy.units import Quantity
import astropy.units as u
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, kron
from scipy.spatial import ConvexHull

from .spacial_basis_set import SpacialBasisSet
from .bspline_spacial_basis_function import BSplineSpacialBasisFunction
    

class BSplineSpacialBasisSet(SpacialBasisSet):
    def __init__(
            self, 
            x_domain: Union[Sequence[float], np.ndarray, Quantity],
            y_domain: Union[Sequence[float], np.ndarray, Quantity],
            z_domain: Union[Sequence[float], np.ndarray, Quantity],
            degree: Union[int, Sequence[int], np.ndarray],
            num_knots: Union[int, Sequence[int], np.ndarray],
            potential_function: Union[Callable, None] = None,
        ):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.z_domain = z_domain
        self.degree = degree
        self.num_knots = num_knots
        self.potential_function = potential_function
        self._check_input()

        if self.potential_function is not None:
            knots_x, knots_y, knots_z = self._create_adaptive_knots_3d()
            self.knots = [knots_x, knots_y, knots_z]
        else:
            knots_x = np.linspace(self.x_domain[0], self.x_domain[1], self.num_knots[0])
            knots_y = np.linspace(self.y_domain[0], self.y_domain[1], self.num_knots[1])
            knots_z = np.linspace(self.z_domain[0], self.z_domain[1], self.num_knots[2])
            self.knots = [knots_x, knots_y, knots_z]

        self.splines_x, self.splines_y, self.splines_z = self._generate_splines()
        self.basis_functions = []
        for spline_x in self.splines_x:
            for spline_y in self.splines_y:
                for spline_z in self.splines_z:
                    self.basis_functions.append(BSplineSpacialBasisFunction(spline_x, spline_y, spline_z))
        self.basis_functions = np.array(self.basis_functions)

        self.N_basis_functions = len(self.basis_functions)

        

    def expand(
        self,
        coeffs: Union[float, Sequence[float], np.ndarray],
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ) -> Union[float, np.ndarray]:
        self.x = x
        self.y = y
        self.z = z
        self._check_xyz()
        if isinstance(coeffs, (float, int)):
            self.coeffs = np.array([coeffs])
        elif isinstance(coeffs, (Sequence, np.ndarray)):
            self.coeffs = np.asarray(coeffs)
        else:
            raise TypeError("coeffs must be a float, Sequence, or np.ndarray.")
        if self.coeffs.shape[0] != self.N_basis_functions:
            raise ValueError("coeffs must have the same length as the number of basis functions.")

        ## Handle different cases based on the dimensionality of x, y, z
        #if np.isscalar(self.x) and np.isscalar(self.y) and np.isscalar(self.z):
        #    # Point evaluation (0D)
        #    result = sum(coeffs[i] * self.basis_functions[i].eval(self.x, self.y, self.z, sparse=False) \
        #                 for i in range(self.N_basis_functions))
        #    return result
        #elif np.isscalar(self.x) and np.isscalar(self.y):
        #    result = csr_matrix((len(self.z),))
        #elif np.isscalar(self.x) and np.isscalar(self.z):
        #    result = csr_matrix((len(self.y),))
        #elif np.isscalar(self.y) and np.isscalar(self.z):
        #    result = csr_matrix((len(self.x),))
        #elif np.isscalar(self.x):
        #    result = csr_matrix((len(self.y), len(self.z)))
        #elif np.isscalar(self.y):
        #    result = csr_matrix((len(self.x), len(self.z)))
        #elif np.isscalar(self.z):
        #    result = csr_matrix((len(self.x), len(self.y)))
        #    
        #for i in range(self.N_basis_functions):
        #        result += coeffs[i] * self.basis_functions[i].eval(self.x, self.y, self.z, sparse=True)
        #return result

        result = np.sum(coeffs*self.basis_functions.eval(self.x, self.y, self.z, sparse=True), axis=0)
        return result



    def overlap_matrix(
        self,
    ) -> np.ndarray:
        pass



    def _create_adaptive_knots_3d(
        self, 
        num_samples: int = 101,
        density_factor: int = 10
    ) -> tuple:
        # Sample the domain
        x_samples = np.linspace(self.x_domain[0], self.x_domain[1], num_samples)
        y_samples = np.linspace(self.y_domain[0], self.y_domain[1], num_samples)
        z_samples = np.linspace(self.z_domain[0], self.z_domain[1], num_samples)

        # Create a grid and evaluate the potential function
        X, Y, Z = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')
        V = self.potential_function(X, Y, Z)

        # Calculate the gradient magnitude
        grad_x, grad_y, grad_z = np.gradient(V, x_samples, y_samples, z_samples)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # Function to create adaptive knots
        def create_knots(axis_samples, grad_magnitude, num_knots, degree):
            grad_agg = np.max(np.max(grad_magnitude, axis=1), axis=1)
            normalized_grad = grad_agg / np.max(grad_agg)
            density = normalized_grad * density_factor + 1
            density /= np.sum(density)
            cumulative_density = np.cumsum(density)

            knot_values = np.linspace(0, 1, num_knots - (2 * degree), endpoint=False)
            knot_positions = np.searchsorted(cumulative_density, knot_values)
            knot_positions = np.clip(knot_positions, 0, len(axis_samples) - 1)

            # Construct the internal knot vector without the repeated endpoints
            internal_knots = axis_samples[knot_positions]

            # Add the repeated endpoints to the internal knots
            knots = np.concatenate(([axis_samples[0]] * degree, internal_knots, [axis_samples[-1]] * degree))
            return knots

        # Create knots for each axis
        knots_x = create_knots(x_samples, grad_magnitude, self.num_knots[0], self.degree[0])
        knots_y = create_knots(y_samples, grad_magnitude, self.num_knots[1], self.degree[1])
        knots_z = create_knots(z_samples, grad_magnitude, self.num_knots[2], self.degree[2])

        assert len(knots_x) == self.num_knots[0] and len(knots_y) == self.num_knots[1] and len(knots_z) == self.num_knots[2]

        return knots_x, knots_y, knots_z



    def _generate_splines(
            self,
        ) -> list:
        splines = []
        for i in range(3):
            splines_1d = []
            n = self.num_knots[i] - self.degree[i] - 1
            for j in range(n):
                coef = np.zeros(n)
                coef[j] = 1
                spline = BSpline(self.knots[i], coef, self.degree[i])
                splines_1d.append(spline)
            splines.append(splines_1d)
        return splines[0], splines[1], splines[2]



    def _check_input(
            self, 
        ):
        # Check that the x_domain is a tuple of floats or a Quantity
        if isinstance(self.x_domain, (Sequence, np.ndarray)) and len(self.x_domain) == 2:
            if self.x_domain[0] < self.x_domain[1]:
                if isinstance(self.x_domain, Quantity): # if x_domain is a Quantity, it's still also a sequence
                    self.x_domain = self.x_domain.to(u.um).value
                elif all(isinstance(x, (int, float)) for x in self.x_domain):
                    self.x_domain = np.array(self.x_domain) 
                else:
                    raise TypeError("x_domain must be a sequence of floats or a Quantity.")
            else:
                raise ValueError("x_domain[0] must be less than x_domain[1].")
        else:
            raise TypeError("x_domain must be a sequence of floats or a Quantity.")
        
        # Check that the y_domain is a tuple of floats or a Quantity
        if isinstance(self.y_domain, (Sequence, np.ndarray)) and len(self.y_domain) == 2:
            if self.y_domain[0] < self.y_domain[1]:
                if isinstance(self.y_domain, Quantity):
                    self.y_domain = self.y_domain.to(u.um).value
                elif all(isinstance(y, (int, float)) for y in self.y_domain):
                    self.y_domain = np.array(self.y_domain) 
                else:
                    raise TypeError("y_domain must be a sequence of floats or a Quantity.")
            else:
                raise ValueError("y_domain[0] must be less than y_domain[1].")
        else:
            raise TypeError("y_domain must be a sequence of floats or a Quantity.")
        
        # Check that the z_domain is a tuple of floats or a Quantity
        if isinstance(self.z_domain, (Sequence, np.ndarray)) and len(self.z_domain) == 2:
            if self.z_domain[0] < self.z_domain[1]:
                if isinstance(self.z_domain, Quantity):
                    self.z_domain = self.z_domain.to(u.um).value
                elif all(isinstance(z, (int, float)) for z in self.z_domain):
                    self.z_domain = np.array(self.z_domain) 
                else:
                    raise TypeError("z_domain must be a sequence of floats or a Quantity.")
            else:
                raise ValueError("z_domain[0] must be less than z_domain[1].")
        else:
            raise TypeError("z_domain must be a sequence of floats or a Quantity.")
        
        # Check that the degree is an int or a Sequence or np.ndarray of ints
        if isinstance(self.degree, int):
            self.degree = np.array([self.degree, self.degree, self.degree])
        elif isinstance(self.degree, (Sequence, np.ndarray)):
            if all(isinstance(d, int) for d in self.degree):
                self.degree = np.array(self.degree)
            else:
                raise TypeError("degree must be an int or a sequence of ints.")
        else:
            raise TypeError("degree must be an int or a sequence of ints.")
        
        # Check that the num_knots is an int or a Sequence or np.ndarray of ints
        if isinstance(self.num_knots, int):
            self.num_knots = np.array([self.num_knots, self.num_knots, self.num_knots])
        elif isinstance(self.num_knots, (Sequence, np.ndarray)):
            if all(isinstance(n, int) for n in self.num_knots):
                self.num_knots = np.array(self.num_knots)
            else:
                raise TypeError("num_knots must be an int or a sequence of ints.")
        else:
            raise TypeError("num_knots must be an int or a sequence of ints.")
        
        # Check that the potential_function is a Callable or None
        if self.potential_function is None or isinstance(self.potential_function, Callable):
            self.potential_function = self.potential_function
        else:
            raise TypeError("potential_function must be a Callable or None.")
       

    def _check_xyz(
        self,
    ):
        # Check input
        if isinstance(self.x, Quantity):
            self.x = self.x.to(u.um).value
        if isinstance(self.x, (float, int)):
            self.x = self.x
        elif isinstance(self.x, (Sequence, np.ndarray)):
            self.x = np.array(self.x)
        else:
            raise TypeError("x must be a float, int, Sequence, or Quantity.")

        if isinstance(self.y, Quantity):
            self.y = self.y.to(u.um).value
        if isinstance(self.y, (float, int)):
            self.y = self.y
        elif isinstance(self.y, (Sequence, np.ndarray)):
            self.y = np.array(self.y)
        else:
            raise TypeError("y must be a float, int, Sequence, or Quantity.")
        
        if isinstance(self.z, Quantity):
            self.z = self.z.to(u.um).value
        if isinstance(self.z, (float, int)):
            self.z = np.array([self.z])
        elif isinstance(self.z, (Sequence, np.ndarray)):
            self.z = self.z
        else:
            raise TypeError("z must be a float, int, Sequence, or Quantity.")