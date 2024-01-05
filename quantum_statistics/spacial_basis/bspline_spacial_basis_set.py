import numpy as np
from typing import Union, Sequence, Callable
from astropy.units import Quantity, Unit
import astropy.units as u
from scipy.interpolate import BSpline
from scipy.integrate import nquad
import sparse

from .spacial_basis_set import SpacialBasisSet
    

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
        self._check_and_process_input("init")

        if self.potential_function is not None:
            knots_x, knots_y, knots_z = self._create_adaptive_knots_3d()
            self.knots = [knots_x, knots_y, knots_z]
        else:
            knots_x = np.linspace(self.x_domain[0], self.x_domain[1], self.num_knots[0])
            knots_y = np.linspace(self.y_domain[0], self.y_domain[1], self.num_knots[1])
            knots_z = np.linspace(self.z_domain[0], self.z_domain[1], self.num_knots[2])
            self.knots = [knots_x, knots_y, knots_z]

        self.splines_x, self.splines_y, self.splines_z = self._generate_splines()
        
        basis_functions = self._generate_3d_bspline_callables_as_tensorproducts()
        super().__init__(basis_functions)


    def eval(
        self,
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ) -> Union[np.ndarray, sparse.COO]:
        """Evaluate all basis functions at the given position(s) and return the result as a sparse.COO
           object of shape (N_basis_functions, len(x), len(y), len(z)).
           
           Args:
               x: The x position(s) to evaluate the basis functions at in [um].
               y: The y position(s) to evaluate the basis functions at in [um].
               z: The z position(s) to evaluate the basis functions at in [um].
               
           Returns:
               The evaluated basis functions of shape (N_basis_functions, len(x), len(y), len(z)).
        """
        self.x = x
        self.y = y
        self.z = z
        self._check_and_process_input("eval") 

        # Evaluate the B-splines
        evaluated_splines_x = sparse.COO.from_numpy(np.array([spline(self.x) for spline in self.splines_x]))
        evaluated_splines_y = sparse.COO.from_numpy(np.array([spline(self.y) for spline in self.splines_y]))
        evaluated_splines_z = sparse.COO.from_numpy(np.array([spline(self.z) for spline in self.splines_z]))

        # Calculate the tensor product of the evaluated splines giving shape 
        # (num_splines_x, num_splines_y, num_splines_z, len(x), len(y), len(z))
        tensor_product = sparse.tensordot(evaluated_splines_x, sparse.tensordot(evaluated_splines_y, \
                                          evaluated_splines_z, axes=0), axes=0)

        # Reshape to (N_basis_functions, len(x), len(y), len(z))
        return tensor_product.reshape((self.N_basis_functions, len(self.x), len(self.y), len(self.z)))
    

    def eval_laplacian(
        self,
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ) -> Union[np.ndarray, sparse.COO]:
        """Evaluate the Laplacian of all basis functions at the given position(s) and return the result as a sparse.COO
           object of shape (N_basis_functions, len(x), len(y), len(z)).
              
           Args:
             x: The x position(s) to evaluate the basis functions at in [um].
             y: The y position(s) to evaluate the basis functions at in [um].
             z: The z position(s) to evaluate the basis functions at in [um].
                
           Returns:
             The evaluated Laplacian of the basis functions of shape (N_basis_functions, len(x), len(y), len(z)).
          """
        self.x = x
        self.y = y
        self.z = z
        self._check_and_process_input("eval")

        # Evaluate the B-splines
        evaluated_splines_x = sparse.COO.from_numpy(np.array([spline(self.x) for spline in self.splines_x]))
        evaluated_splines_y = sparse.COO.from_numpy(np.array([spline(self.y) for spline in self.splines_y]))
        evaluated_splines_z = sparse.COO.from_numpy(np.array([spline(self.z) for spline in self.splines_z]))

        # Evaluate the second derivatives of the B-splines
        d2x_evaluated_splines_x = sparse.COO.from_numpy(np.array([spline.derivative(2)(self.x) for spline in self.splines_x]))
        d2y_evaluated_splines_y = sparse.COO.from_numpy(np.array([spline.derivative(2)(self.y) for spline in self.splines_y]))
        d2z_evaluated_splines_z = sparse.COO.from_numpy(np.array([spline.derivative(2)(self.z) for spline in self.splines_z]))

        # Calculate the laplacian of the evaluated splines as tensor product giving the shape 
        # (num_splines_x, num_splines_y, num_splines_z, len(x), len(y), len(z))
        tensor_product = sparse.tensordot(d2x_evaluated_splines_x, sparse.tensordot(evaluated_splines_y, \
                                          evaluated_splines_z, axes=0), axes=0) + \
                         sparse.tensordot(evaluated_splines_x, sparse.tensordot(d2y_evaluated_splines_y, \
                                          evaluated_splines_z, axes=0), axes=0) + \
                         sparse.tensordot(evaluated_splines_x, sparse.tensordot(evaluated_splines_y, \
                                          d2z_evaluated_splines_z, axes=0), axes=0) 

        # Reshape to (N_basis_functions, len(x), len(y), len(z))
        return tensor_product.reshape((self.N_basis_functions, len(self.x), len(self.y), len(self.z)))

    

    def expand(
        self,
        coeffs: Union[float, Sequence[float], np.ndarray],
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ) -> Union[np.ndarray, sparse.COO]:
        """Expand the basis functions with the given coefficients at the given position(s) and return the result as 
           sparse.COO object of shape (len(x), len(y), len(z)).
           
           Args:
               coeffs: The coefficients to expand the basis functions with. Must be of length `N_basis_functions`.
               x: The x position(s) to evaluate the basis functions at in [um].
               y: The y position(s) to evaluate the basis functions at in [um].
               z: The z position(s) to evaluate the basis functions at in [um].
               
           Returns:
               The expansion value(s) of shape (len(x), len(y), len(z)).
        """
        self.coeffs = coeffs
        self.x = x
        self.y = y
        self.z = z
        self._check_and_process_input("expand")        

        # Evaluate the basis functions at the given positions (shape = (N_basis_functions, len(x), len(y), len(z))
        basis_funcs_evaluated = self.eval(self.x, self.y, self.z)

        # Reshape the coefficients to (N_basis_functions, 1, 1, 1) to be broadcastable with the basis functions
        self.coeffs = self.coeffs.reshape((self.N_basis_functions, 1, 1, 1))

        # Do weighted sum of basis functions resulting in the shape (len(x), len(y), len(z))
        return (self.coeffs * basis_funcs_evaluated).sum(axis=0)


    def get_kinetic_energy_matrix(
        self,
        unit: Union[str, Unit] = u.eV,
    ) -> np.ndarray:
        """Return the kinetic energy matrix in terms of the basis functions, which has matrix elements
           T_ij = -hbar^2/2m int d^3r b*_i(r) nabla^2 b_j(r)
           Here b_i(r) is the ith basis function and nabla^2 is the Laplacian operator.
           
           Args:
               unit: The unit of the kinetic energy matrix. Defaults to [eV].
               
           Returns: 
               The kinetic energy matrix of shape (`N_basis_functions`, `N_basis_functions`).
        """
        # Vectorized integration
        limits = (self.x_domain, self.y_domain, self.z_domain)
        v_integral = np.vectorize(lambda f: nquad(f, limits))

        # Integrate b*_i(r) nabla^2 b_j(r) over the domain
        def integrand(x, y, z):
            basis_funcs = self.eval(x, y, z)
            laplacians = self.eval_laplacian(x, y, z)

            # Reshape, so that basis_funcs has shape (N_basis_functions, 1, len(x), len(y), len(z)) and
            # laplacians has shape (1, N_basis_functions, len(x), len(y), len(z))
            extended_basis_funcs = sparse.COO(basis_funcs.data, basis_funcs.coords, shape=(basis_funcs.shape[0], 1, *basis_funcs.shape[1:]))
            extended_laplacians = sparse.COO(laplacians.data, laplacians.coords, shape=(1, *laplacians.shape))

            # Perform element-wise multiplication, utilizing broadcasting
            # Result shape: (N_basis_funcs, N_basis_funcs, len(x), len(y), len(z))
            return extended_basis_funcs * extended_laplacians
        
        # Integrate over the domain
        integral = v_integral(integrand)

        return integral



    def get_potential_energy_matrix(
        self,
        V: Callable,
    ) -> np.ndarray:
        pass


    def overlap_matrix(
        self,
    ) -> np.ndarray:
        pass


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
    

    def _generate_3d_bspline_callables_as_tensorproducts(
            self,
        ) -> list:
        basis_functions = []
        for spline_x in self.splines_x:
            for spline_y in self.splines_y:
                for spline_z in self.splines_z:
                    def f(x, y, z, coeff=1):
                        return coeff * np.tensordot(spline_x(x), np.tensordot(spline_y(y), spline_z(z), axes=0), axes=0)
                    basis_functions.append(f)
        return basis_functions


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

    
        
    def _check_and_process_input(
            self,
            which_method: str,
    ):
        if which_method == "init":
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



        if  which_method == "eval":
            # Check x, y, and z
            if isinstance(self.x, Quantity):
                self.x = self.x.to(u.um).value
            if isinstance(self.x, (float, int)):
                self.x = np.atleast_1d(self.x)
            elif isinstance(self.x, (Sequence, np.ndarray)):
                if all(isinstance(x_i, (float, int)) for x_i in self.x):
                    self.x = np.asarray(self.x)
                else:
                    raise TypeError("x must be float or sequence of floats.")
            else:
                raise TypeError("x must be float or sequence of floats.")
            
            if isinstance(self.y, Quantity):
                self.y = self.y.to(u.um).value
            if isinstance(self.y, (float, int)):
                self.y = np.atleast_1d(self.y)
            elif isinstance(self.y, (Sequence, np.ndarray)):
                if all(isinstance(y_i, (float, int)) for y_i in self.y):
                    self.y = np.asarray(self.y)
                else:
                    raise TypeError("y must be float or sequence of floats.")
            else:
                raise TypeError("y must be float or sequence of floats.")

            if isinstance(self.z, Quantity):
                self.z = self.z.to(u.um).value
            if isinstance(self.z, (float, int)):
                self.z = np.atleast_1d(self.z)
            elif isinstance(self.z, (Sequence, np.ndarray)):
                if all(isinstance(z_i, (float, int)) for z_i in self.z):
                    self.z = np.asarray(self.z)
                else:
                    raise TypeError("z must be float or sequence of floats.")
            else:
                raise TypeError("z must be float or sequence of floats.")
            


        if which_method == "expand":
            #Check coeffs
            if isinstance(self.coeffs, (float, int)):
                self.coeffs = np.atleast_1d(self.coeffs)
            elif isinstance(self.coeffs, (Sequence, np.ndarray)):
                if all(isinstance(c, (float, int)) for c in self.coeffs):
                    self.coeffs = np.asarray(self.coeffs)
            else:
                raise TypeError("self.coeffs must be a float or a sequence of floats.")
            if self.coeffs.shape[0] != self.N_basis_functions:
                raise ValueError(f"Length of self.coeffs ({self.coeffs.shape[0]}) does not match number of basis functions \
                                ({self.N_basis_functions}).")
       