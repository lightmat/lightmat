import numpy as np
from typing import Union, Sequence, Callable
from astropy.units import Quantity, Unit
import astropy.units as u
from scipy.interpolate import BSpline
import sparse

from .nquad_vec import nquad_vec
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
        points: Union[Sequence[float], np.ndarray, Quantity],
        return_sparse: bool = True,
    ) -> Union[np.ndarray, sparse.COO]:
        """Evaluate all basis functions at the given point(s) and return the result as a sparse.COO object 
           if `return_sparse` is True, otherwise return a dense numpy array. The shape of the result is 
           (`N_basis_functions`, `num_points`).
           
           Args:
               points: The point(s) to evaluate the basis functions at in [um].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to True.
               
           Returns:
               The evaluated basis functions of shape (`N_basis_functions`, `num_points`).
        """
        self.points = points
        self.return_sparse = return_sparse
        self._check_and_process_input("eval") 
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        # Evaluate the B-splines
        evaluated_splines_x = sparse.COO.from_numpy(np.array([spline(x) for spline in self.splines_x]))
        evaluated_splines_y = sparse.COO.from_numpy(np.array([spline(y) for spline in self.splines_y]))
        evaluated_splines_z = sparse.COO.from_numpy(np.array([spline(z) for spline in self.splines_z]))

        # Use broadcasting to calculate the tensor product of the evaluated splines at each point
        # resulting in the shape (num_splines_x, num_splines_y, num_splines_z, num_points)
        tensor_product = evaluated_splines_x[:, np.newaxis, np.newaxis, :] * \
                         evaluated_splines_y[np.newaxis, :, np.newaxis, :] * \
                         evaluated_splines_z[np.newaxis, np.newaxis, :, :]
        assert tensor_product.shape == (len(evaluated_splines_x), len(evaluated_splines_y), \
               len(evaluated_splines_z),len(self.points)), (f"tensor_product.shape = {tensor_product.shape}, "
               f"but should be ({len(evaluated_splines_x)}, {len(evaluated_splines_y)}, "
               f"{len(evaluated_splines_z)}, {len(self.points)})")
        
        # Reshape to (N_basis_functions, num_points)
        tensor_product = tensor_product.reshape((self.N_basis_functions, len(self.points)))

        if return_sparse:
            return tensor_product
        else:
            return tensor_product.todense()
    

    def eval_laplacian(
        self,
        points: Union[Sequence[float], np.ndarray, Quantity],
        return_sparse: bool = True,
    ) -> Union[np.ndarray, sparse.COO]:
        """Evaluate all the laplacian of all basis functions at the given point(s) and return the result 
           as a sparse.COO object if `return_sparse` is True, otherwise return a dense numpy array. 
           The shape of the result is (`N_basis_functions`, `num_points`).
              
           Args:
               points: The point(s) to evaluate the basis functions at in [um].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to True.
                
           Returns:
               The evaluated laplacians of basis functions of shape (`N_basis_functions`, `num_points`).
          """
        self.points = points
        self.return_sparse = return_sparse
        self._check_and_process_input("eval") 
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        # Evaluate the B-splines
        evaluated_splines_x = sparse.COO.from_numpy(np.array([spline(x) for spline in self.splines_x]))
        evaluated_splines_y = sparse.COO.from_numpy(np.array([spline(y) for spline in self.splines_y]))
        evaluated_splines_z = sparse.COO.from_numpy(np.array([spline(z) for spline in self.splines_z]))

        # Evaluate the second derivatives of the B-splines
        d2x_evaluated_splines_x = sparse.COO.from_numpy(np.array([spline.derivative(2)(x) for spline in \
                                                                  self.splines_x]))
        d2y_evaluated_splines_y = sparse.COO.from_numpy(np.array([spline.derivative(2)(y) for spline in \
                                                                    self.splines_y]))
        d2z_evaluated_splines_z = sparse.COO.from_numpy(np.array([spline.derivative(2)(z) for spline in \
                                                                    self.splines_z]))

        # Use broadcasting to calculate the laplacian of the evaluated splines as tensor productat each 
        # point resulting in the shape (num_splines_x, num_splines_y, num_splines_z, num_points)
        tensor_product = d2x_evaluated_splines_x[:, np.newaxis, np.newaxis, :] * \
                            evaluated_splines_y[np.newaxis, :, np.newaxis, :] * \
                            evaluated_splines_z[np.newaxis, np.newaxis, :, :] + \
                         evaluated_splines_x[:, np.newaxis, np.newaxis, :] * \
                            d2y_evaluated_splines_y[np.newaxis, :, np.newaxis, :] * \
                            evaluated_splines_z[np.newaxis, np.newaxis, :, :] + \
                         evaluated_splines_x[:, np.newaxis, np.newaxis, :] * \
                            evaluated_splines_y[np.newaxis, :, np.newaxis, :] * \
                            d2z_evaluated_splines_z[np.newaxis, np.newaxis, :, :]
        assert tensor_product.shape == (len(evaluated_splines_x), len(evaluated_splines_y), \
               len(evaluated_splines_z),len(self.points)), (f"tensor_product.shape = {tensor_product.shape}, "
               f"but should be ({len(evaluated_splines_x)}, {len(evaluated_splines_y)}, "
               f"{len(evaluated_splines_z)}, {len(self.points)})")
        
        # Reshape to (N_basis_functions, num_points)
        tensor_product = tensor_product.reshape((self.N_basis_functions, len(self.points)))

        if return_sparse:
            return tensor_product
        else:
            return tensor_product.todense()



    def expand(
        self,
        coeffs: Union[float, Sequence[float], np.ndarray],
        points: Union[Sequence[float], np.ndarray, Quantity], 
        return_sparse: bool = False,  
    ) -> Union[np.ndarray, sparse.COO]:
        """Expand the basis functions with the given coefficients at the given point(s) and return the result 
           as a sparse.COO object if `return_sparse` is True, otherwise return a dense numpy array. 
           The shape of the result is (`N_basis_functions`, `num_points`).
           
           Args:
               coeffs: The coefficients to expand the basis functions with. Must be of length 
                       `N_basis_functions`.
               points: The point(s) to evaluate the basis functions at in [um].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to False.
               
               
           Returns:
               The expansion value(s) of shape (`num_points`).
        """
        self.coeffs = coeffs
        self.points = points
        self.return_sparse = return_sparse
        self._check_and_process_input("expand") 

        # Evaluate the basis functions at the given points resulting in the shape
        # (N_basis_functions, num_points)
        basis_funcs_evaluated = self.eval(self.points, return_sparse=True)
        assert basis_funcs_evaluated.shape == (self.N_basis_functions, len(self.points)), \
               (f"basis_funcs_evaluated.shape = {basis_funcs_evaluated.shape}, but should be "
               f"({self.N_basis_functions}, {len(self.points)})")

        # Reshape the coefficients to (N_basis_functions, 1) to be broadcastable with the basis functions
        self.coeffs = self.coeffs.reshape((self.N_basis_functions, 1))

        # Do weighted sum of basis functions resulting in the shape (num_points,)
        result = (self.coeffs * basis_funcs_evaluated).sum(axis=0)

        if return_sparse:
            return result
        else:
            return result.todense()


    def get_kinetic_energy_matrix(
        self,
        unit: Union[str, Unit] = u.eV,
        return_sparse: bool = True,
    ) -> np.ndarray:
        """Return the kinetic energy matrix in terms of the basis functions, which has matrix elements
           T_ij = -hbar^2/2m int d^3r b*_i(r) nabla^2 b_j(r)
           Here b_i(r) is the ith basis function and nabla^2 is the Laplacian operator.
           
           Args:
               unit: The unit of the kinetic energy matrix. Defaults to [eV].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to True.
               
           Returns: 
               The kinetic energy matrix of shape (`N_basis_functions`, `N_basis_functions`).
        """
        # Define a integrand function to be used by nquad_vec()
        def integrand(points):
            # Evaluate basis functions and their Laplacians
            basis_funcs = self.eval(points)  # shape: (N_basis_functions, num_points)
            laplacians = self.eval_laplacian(points)  # Same shape as basis_funcs

            # Calculate the tensor product resulting in the shape 
            # (N_basis_functions, N_basis_functions, num_points)
            tensor_product = basis_funcs[:, np.newaxis, :] * laplacians[np.newaxis, :, :]
            assert tensor_product.shape == (self.N_basis_functions, self.N_basis_functions, len(points)), \
            (f"tensor_product.shape = {tensor_product.shape}, but should be ({self.N_basis_functions}, "
            f"{self.N_basis_functions}, {len(points)})")

            ## Reshape to 2D array (num_points, N_basis_functions**2) as required by cubature
            #return tensor_product.reshape((self.N_basis_functions**2, len(points))).T
            nz_elements = tensor_product.data
            self.nz_indices = tensor_product.coords # we need these later to reconstruct the matrix

            return nz_elements


        # Define integration bounds
        xmin = np.array([self.x_domain[0], self.y_domain[0], self.z_domain[0]])
        xmax = np.array([self.x_domain[1], self.y_domain[1], self.z_domain[1]])

        return integrand





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



        if  which_method == "eval" or which_method == "expand":
            # Check points
            if isinstance(self.points, Quantity) and self.points.unit.is_equivalent(u.um):
                self.points = self.points.to(u.um).value
            if isinstance(self.points, np.ndarray):
                if len(self.points.shape) == 1 and  self.points.shape[0] == 3 and \
                    all(isinstance(p, (float, int)) for p in self.points):
                    self.points = np.atleast_2d(self.points)
                elif len(self.points.shape) == 2 and self.points.shape[1] == 3 and \
                      all(isinstance(p, (float, int)) for p in self.points.flatten()):
                    pass
                else:
                    raise ValueError("points must be a 1D or 2D array of shape (3,) or (N, 3).")
            elif isinstance(self.points, Sequence):
                if len(self.points) == 3 and all(isinstance(p, (float, int)) for p in self.points):
                    self.points = np.atleast_2d(self.points)
                elif all(len(p) == 3 and all(isinstance(p_i, (float, int)) for p_i in p) for p in self.points):
                    self.points = np.array(self.points)
                else:
                    raise ValueError("points must be a sequence of length 3 or a sequence of sequences of length 3.")
            else:
                raise TypeError("points must be a sequence or a numpy array.")
            
            # Check return_sparse
            if not isinstance(self.return_sparse, bool):
                raise TypeError("return_sparse must be a bool.")
                      
            
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
       