import numpy as np
from typing import Union, Sequence, Callable
from astropy.units import Quantity, Unit
import astropy.units as u
from scipy.interpolate import BSpline
from scipy.integrate import simps
import sparse

from .spacial_basis_set import SpacialBasisSet
    

class BSplineSpacialBasisSet(SpacialBasisSet):
    def __init__(
            self, 
            domain: Union[Sequence[float], np.ndarray, Quantity],
            degree: Union[int, Sequence[int], np.ndarray],
            num_knots: Union[int, Sequence[int], np.ndarray],
            potential_function: Union[Callable, None] = None,
        ):
        super().__init__()

        self.domain = domain
        self.degree = degree
        self.num_knots = num_knots
        self.potential_function = potential_function
        self._check_and_process_input("init")

        if self.potential_function is not None:
            knots_x, knots_y, knots_z = self._create_adaptive_knots_3d()
            self.knots = [knots_x, knots_y, knots_z]
        else:
            knots_x = np.linspace(self.domain[0, 0], self.domain[0, 1], self.num_knots[0])
            knots_y = np.linspace(self.domain[1, 0], self.domain[1, 1], self.num_knots[1])
            knots_z = np.linspace(self.domain[2, 0], self.domain[2, 1], self.num_knots[2])
            self.knots = [knots_x, knots_y, knots_z]

        splines_x, splines_y, splines_z = self._generate_splines()
        self.splines = [splines_x, splines_y, splines_z]
        self.d2splines = [np.array([spline.derivative(2) for spline in splines_x]), \
                                     np.array([spline.derivative(2) for spline in splines_y]), \
                                     np.array([spline.derivative(2) for spline in splines_z])]
        self.num_basis_funcs = len(splines_x) * len(splines_y) * len(splines_z)



    def eval(
        self,
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        return_sparse: bool = True,
    ) -> Union[np.ndarray, sparse.COO]:
        """Evaluate all basis functions at the given position(s) and return the result as a sparse.COO object 
           if `return_sparse` is True, otherwise return a dense numpy array. The shape of the result is 
           (`num_basis_funcs`, `num_x_points`, `num_y_points`, `num_z_points`).
           
           Args:
               x: The x position(s) to evaluate the basis functions at in [um].
               y: The y position(s) to evaluate the basis functions at in [um].
               z: The z position(s) to evaluate the basis functions at in [um].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to True.
               
           Returns:
               The evaluated basis functions of shape 
               (`num_basis_funcs`, `num_x_points`, `num_y_points`, `num_z_points`).
        """
        self.x = x
        self.y = y
        self.z = z
        self.return_sparse = return_sparse
        self._check_and_process_input("eval") 

        # Evaluate the B-splines
        evaluated_splines_x = sparse.COO.from_numpy(np.array([spline(self.x) for spline in self.splines[0]]))
        evaluated_splines_y = sparse.COO.from_numpy(np.array([spline(self.y) for spline in self.splines[1]]))
        evaluated_splines_z = sparse.COO.from_numpy(np.array([spline(self.z) for spline in self.splines[2]]))

        # Calculate the tensor product of the evaluated splines resulting in shape
        # (num_splines_x, num_x_points, num_splines_y, num_y_points, num_splines_z, num_z_points)
        tensor_product = sparse.tensordot(evaluated_splines_x, sparse.tensordot(evaluated_splines_y, \
                                          evaluated_splines_z, axes=0), axes=0)
        assert tensor_product.shape == (len(evaluated_splines_x), len(self.x), len(evaluated_splines_y), \
                                        len(self.y), len(evaluated_splines_z), len(self.z))
        
        # Transpose to order the dimensions correctly to the desired shape
        # (num_splines_x, num_splines_y, num_splines_z, num_x_points, num_y_points, num_z_points)
        tensor_product = tensor_product.transpose((0, 2, 4, 1, 3, 5))
        assert tensor_product.shape == (len(evaluated_splines_x), len(evaluated_splines_y), \
                                        len(evaluated_splines_z), len(self.x), len(self.y), len(self.z))
        
        # Reshape to (num_basis_funcs, num_x_points, num_y_points, num_z_points)
        tensor_product = tensor_product.reshape((self.num_basis_funcs, len(self.x), len(self.y), len(self.z)))
        assert tensor_product.shape == (self.num_basis_funcs, len(self.x), len(self.y), len(self.z))

        if return_sparse:
            return tensor_product
        else:
            return tensor_product.todense()
    

    def eval_laplacian(
        self,
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        return_sparse: bool = True,
    ) -> Union[np.ndarray, sparse.COO]:
        """Evaluate the laplacians of all basis functions at the given position(s) and return the result as 
           a sparse.COO object if `return_sparse` is True, otherwise return a dense numpy array. The shape 
           of the result is (`num_basis_funcs`, `num_x_points`, `num_y_points`, `num_z_points`).
           
           Args:
               x: The x position(s) to evaluate the basis functions at in [um].
               y: The y position(s) to evaluate the basis functions at in [um].
               z: The z position(s) to evaluate the basis functions at in [um].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to True.
               
           Returns:
               The evaluated laplacians of all basis functions of shape 
               (`num_basis_funcs`, `num_x_points`, `num_y_points`, `num_z_points`).
        """
        self.x = x
        self.y = y
        self.z = z
        self.return_sparse = return_sparse
        self._check_and_process_input("eval") 

        # Evaluate the B-splines
        evaluated_splines_x = sparse.COO.from_numpy(np.array([spline(self.x) for spline in self.splines[0]]))
        evaluated_splines_y = sparse.COO.from_numpy(np.array([spline(self.y) for spline in self.splines[1]]))
        evaluated_splines_z = sparse.COO.from_numpy(np.array([spline(self.z) for spline in self.splines[2]]))

        # Evaluate the second derivatives of the B-splines
        d2x_evaluated_splines_x = sparse.COO.from_numpy(np.array([spline.derivative(2)(self.x) for spline in \
                                                                  self.splines[0]]))
        d2y_evaluated_splines_y = sparse.COO.from_numpy(np.array([spline.derivative(2)(self.y) for spline in \
                                                                    self.splines[1]]))
        d2z_evaluated_splines_z = sparse.COO.from_numpy(np.array([spline.derivative(2)(self.z) for spline in \
                                                                    self.splines[2]]))

        # Calculate the laplacian tensor product of the evaluated splines and derivatives respectively resulting
        # in shape (num_splines_x, num_x_points, num_splines_y, num_y_points, num_splines_z, num_z_points)
        tensor_product = sparse.tensordot(d2x_evaluated_splines_x, sparse.tensordot(evaluated_splines_y, \
                                            evaluated_splines_z, axes=0), axes=0) + \
                         sparse.tensordot(evaluated_splines_x, sparse.tensordot(d2y_evaluated_splines_y, \
                                            evaluated_splines_z, axes=0), axes=0) + \
                         sparse.tensordot(evaluated_splines_x, sparse.tensordot(evaluated_splines_y, \
                                            d2z_evaluated_splines_z, axes=0), axes=0)
        assert tensor_product.shape == (len(evaluated_splines_x), len(self.x), len(evaluated_splines_y), \
                                        len(self.y), len(evaluated_splines_z), len(self.z))
        
        # Transpose to order the dimensions correctly to the desired shape
        # (num_splines_x, num_splines_y, num_splines_z, num_x_points, num_y_points, num_z_points)
        tensor_product = tensor_product.transpose((0, 2, 4, 1, 3, 5))
        assert tensor_product.shape == (len(evaluated_splines_x), len(evaluated_splines_y), \
                                        len(evaluated_splines_z), len(self.x), len(self.y), len(self.z))

        # Reshape to (num_basis_funcs, num_x_points, num_y_points, num_z_points)
        tensor_product = tensor_product.reshape((self.num_basis_funcs, len(self.x), len(self.y), len(self.z)))
        assert tensor_product.shape == (self.num_basis_funcs, len(self.x), len(self.y), len(self.z))

        if return_sparse:
            return tensor_product
        else:
            return tensor_product.todense()



    def expand(
        self,
        coeffs: Union[float, Sequence[float], np.ndarray],
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        return_sparse: bool = False,  
    ) -> Union[np.ndarray, sparse.COO]:
        """Expand the basis functions with the given coefficients at the given positions(s) and return the 
           result as a sparse.COO object if `return_sparse` is True, otherwise return a dense numpy array. 
           The shape of the result is (`num_basis_funcs`, `num_x_points`, `num_y_points`, `num_z_points`).
           
           Args:
               coeffs: The coefficients to expand the basis functions with. Must be of length `num_basis_funcs`.
               x: The x position(s) to evaluate the basis functions at in [um].
               y: The y position(s) to evaluate the basis functions at in [um].
               z: The z position(s) to evaluate the basis functions at in [um].
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to False.
               
               
           Returns:
               The expansion value(s) of shape (`num_points`).
        """
        self.coeffs = coeffs
        self.return_sparse = return_sparse
        self._check_and_process_input("expand") # don't need to check x,y,z as they were checked in eval()

        # Evaluate the basis functions at the given points resulting in the shape
        # (num_basis_funcs, num_x_points, num_y_points, num_z_points)
        basis_funcs_evaluated = self.eval(x, y, z, return_sparse=True)
        assert basis_funcs_evaluated.shape == (self.num_basis_funcs, len(np.atleast_1d(x)), \
                                               len(np.atleast_1d(y)), len(np.atleast_1d(z)))

        # Reshape the coefficients to (num_basis_funcs, 1, 1, 1) to be broadcastable with the basis functions
        self.coeffs = self.coeffs.reshape((self.num_basis_funcs, 1, 1, 1))

        # Do weighted sum of basis functions resulting in the shape (num_x_points, num_y_points, num_z_points)
        result = (self.coeffs * basis_funcs_evaluated).sum(axis=0)

        if return_sparse:
            return result
        else:
            return result.todense()



    def get_kinetic_energy_matrix(
        self,
        unit: Union[str, Unit] = u.eV,
        num_samples: int = 101,
        return_sparse: bool = True,
    ) -> np.ndarray:
        """Return the kinetic energy matrix in terms of the basis functions, which has matrix elements
           T_ij = -hbar^2/2m int d^3r b*_i(r) nabla^2 b_j(r)
           Here b_i(r) is the ith basis function and nabla^2 is the Laplacian operator.
           
           Args:
               unit: The unit of the kinetic energy matrix. Defaults to [eV].
               num_samples: The number of samples to use when evaluating the integral. Defaults to 101.
               return_sparse: If True return a sparse.COO object, else a dense numpy array. Defaults to True.
               
           Returns: 
               The kinetic energy matrix of shape (`num_basis_funcs`, `num_basis_funcs`).
        """
        self.unit = unit
        self.return_sparse = return_sparse
        self._check_and_process_input("get_kinetic_energy_matrix")

        # Get array of index pairs (i, j) for the non-zero overlap Bsplines spline_i(x)*spline_j(x) in each 
        # direction x,y,z  (using that BSplines are only non-zero in the interval [knots[i], knots[i+degree+1]])
        index_pairs = [] # we need to do it for each direction x, y, z
        for dim in range(3):
            num_splines = len(self.splines[dim])
            i_indices, j_indices = np.indices((num_splines, num_splines))
            mask = (i_indices < j_indices+self.degree[dim]+1) & (i_indices+self.degree[dim]+1 > j_indices)
            pairs = np.argwhere(mask)
            index_pairs.append(pairs)

        # Create a 1d integrand function for each direction x,y,z, then we can later obtain the 3d integral
        # result as a tensor product of the 1d integrals
        def integrand_1d(x, dim, pairs):
            # Evaluate the B-splines and derivatives, each resulting in shape (num_splines, num_points)
            evaluated_splines = np.array([spline(x) for spline in self.splines[dim]])
            d2_evaluated_splines = np.array([d2spline(x) for d2spline in self.d2splines[dim]])              
         
            # Calculate the integrand for the non-zero overlap functions
            i = pairs[:, 0]
            j = pairs[:, 1]
            integrand = np.zeros((self.num_knots[dim], self.num_knots[dim], len(x)))
            integrand[i, j, :] = evaluated_splines[i, :] * (d2_evaluated_splines[j, :] + \
                                                            2 * evaluated_splines[j, :])
            assert integrand.shape == (self.num_knots[dim], self.num_knots[dim], len(x))
            
            return integrand

        # Use scipy.integrate.simps to integrate the 1d integrand functions because then we can evaluate
        # the integrand in a vectorized way both in terms of the x points and the index pairs
        x = np.linspace(self.domain[0, 0], self.domain[0, 1], num_samples)
        integrand_x = integrand_1d(x, 0, index_pairs[0])
        result_x = simps(integrand_x, x, axis=2)
        assert result_x.shape == (self.num_knots[0], self.num_knots[0])

        y = np.linspace(self.domain[1, 0], self.domain[1, 1], num_samples)
        integrand_y = integrand_1d(y, 1, index_pairs[1])
        result_y = simps(integrand_y, y, axis=2)
        assert result_y.shape == (self.num_knots[1], self.num_knots[1])

        z = np.linspace(self.domain[2, 0], self.domain[2, 1], num_samples)
        integrand_z = integrand_1d(z, 2, index_pairs[2])
        result_z = simps(integrand_z, z, axis=2)
        assert result_z.shape == (self.num_knots[2], self.num_knots[2])

        return result_x, result_y, result_z



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
            splines.append(np.array(splines_1d))
        return splines[0], splines[1], splines[2]
    

    def _create_adaptive_knots_3d(
        self, 
        num_samples: int = 101,
        density_factor: int = 10
    ) -> tuple:
        # Sample the domain
        x_samples = np.linspace(self.domain[0, 0], self.domain[0, 1], num_samples)
        y_samples = np.linspace(self.domain[1, 0], self.domain[1, 1], num_samples)
        z_samples = np.linspace(self.domain[2, 0], self.domain[2, 1], num_samples)

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
            
            # Check that the degree is a sequence of length 3 containing ints or a single int, then degree
            # becomes [degree, degree, degree]
            if isinstance(self.degree, int):
                self.degree = np.array([self.degree, self.degree, self.degree])
            elif isinstance(self.degree, (Sequence, np.ndarray)):
                if all(isinstance(d, int) for d in self.degree):
                    self.degree = np.array(self.degree)
                else:
                    raise TypeError("degree must be an int or a sequence of ints.")
            else:
                raise TypeError("degree must be an int or a sequence of ints.")
            
            # Check that the num_knots is a sequence of length 3 containing ints or a single int, then num_knots
            # becomes [num_knots, num_knots, num_knots]
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
                if self.x.unit.is_equivalent(u.um):
                    self.x = self.x.to(u.um).value
                else:
                    raise u.UnitsError("x must be in units of length.")
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
                if self.y.unit.is_equivalent(u.um):
                    self.y = self.y.to(u.um).value
                else:
                    raise u.UnitsError("y must be in units of length.")
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
                if self.z.unit.is_equivalent(u.um):
                    self.z = self.z.to(u.um).value
                else:
                    raise u.UnitsError("z must be in units of length.")
            if isinstance(self.z, (float, int)):
                self.z = np.atleast_1d(self.z)
            elif isinstance(self.z, (Sequence, np.ndarray)):
                if all(isinstance(z_i, (float, int)) for z_i in self.z):
                    self.z = np.asarray(self.z)
                else:
                    raise TypeError("z must be float or sequence of floats.")
            else:
                raise TypeError("z must be float or sequence of floats.")
            
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
            if self.coeffs.shape[0] != self.num_basis_funcs:
                raise ValueError(f"Length of self.coeffs ({self.coeffs.shape[0]}) does not match number of basis functions \
                                ({self.num_basis_funcs}).")
            
            # Check return_sparse
            if not isinstance(self.return_sparse, bool):
                raise TypeError("return_sparse must be a bool.")
            

        if which_method == "get_kinetic_energy_matrix":
            # Check unit
            if isinstance(self.unit, str):
                self.unit = u.Unit(self.unit)
            if not isinstance(self.unit, Unit):
                raise TypeError("unit must be a string or an astropy.units.Unit.")
            if not self.unit.is_equivalent(u.eV):
                raise u.UnitsError("energy unit must be equivalent to eV.")
            
            # Check return_sparse
            if not isinstance(self.return_sparse, bool):
                raise TypeError("return_sparse must be a bool.")
            
        