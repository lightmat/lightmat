import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix, kron
from scipy.spatial import ConvexHull


class BasisSet(ABC):
    def __init__(self, parameters: dict):
        self.parameters = parameters

    @abstractmethod
    def expand(self, coeffs: np.ndarray, position: np.ndarray) -> float:
        pass

    @abstractmethod
    def integral(self, function) -> float:
        pass

    @abstractmethod
    def overlap_matrix(self) -> np.ndarray:
        pass


class BSplineBasis1D:
    """
    Represents a 1D B-spline basis.

    Attributes:
        knots (np.ndarray): Knot positions for the B-spline.
        degree (int): Degree of the B-spline.
        basis_functions (list of BSpline): List of B-spline basis functions.

    Methods:
        evaluate(x, i): Evaluates the i-th B-spline basis function at x.
        evaluate_sparse(x, i): Evaluates the i-th B-spline basis function at x in a sparse format.
    """
    def __init__(
            self, 
            domain: tuple, 
            degree: int, 
            num_knots: int, 
            knots: np.ndarray = None
        ):
        """
        Initializes a 1D B-spline basis.

        Args:
            domain (tuple): The domain range (start, end) for the B-spline.
            degree (int): Degree of the B-spline.
            num_knots (int): Number of knots.
            knots (np.ndarray, optional): Custom array of knots. If None, a uniform knot vector is used.
        """
        self.knots = knots if knots is not None else np.linspace(domain[0], domain[1], num_knots+degree+1)
        self.degree = degree
        self.basis_functions = self._generate_basis()

    def _generate_basis(
            self,
        ) -> list:
        """
        Generates the list of B-spline basis functions.

        This internal method computes the B-spline basis functions based on the provided knot vector 
        and degree. Each basis function is a BSpline object.

        Returns:
            list: List of BSpline objects representing the basis functions.
        """
        basis_functions = []
        n = len(self.knots) - self.degree - 1
        for i in range(n):
            coef = np.zeros(n)
            coef[i] = 1
            spline = BSpline(self.knots, coef, self.degree)
            basis_functions.append(spline)
        return basis_functions

    def evaluate(
            self, 
            x, 
            i: int,
        ):
        """
        Evaluates the i-th B-spline basis function at a given point or points.

        Args:
            x (float or np.ndarray): Point(s) at which to evaluate the basis function.
            i (int): Index of the basis function to evaluate.

        Returns:
            The value(s) of the i-th basis function at x. Returns a scalar if x is a float,
            and a NumPy array if x is a NumPy array.
        """
        if isinstance(x, np.ndarray):
            return np.array([self.basis_functions[i](xi) for xi in x])
        else:
            return self.basis_functions[i](x)

    def evaluate_sparse(
            self, 
            x, 
            i: int,
        ) -> csr_matrix:
        """
        Evaluates the i-th B-spline basis function at a given point or points in a sparse format.

        Args:
            x (float or np.ndarray): Point(s) at which to evaluate the basis function.
            i (int): Index of the basis function to evaluate.

        Returns:
            csr_matrix: Sparse matrix representation of the basis function evaluation.
        """
        if isinstance(x, np.ndarray):
            return csr_matrix([self.basis_functions[i](xi) for xi in x])
        else:
            return csr_matrix([[self.basis_functions[i](x)]])


class Adaptive3DBSplineBasis:
    """
    Represents a 3D B-spline basis adapted to a potential function.

    Attributes:
        x_basis (BSplineBasis1D): B-spline basis for the x dimension.
        y_basis (BSplineBasis1D): B-spline basis for the y dimension.
        z_basis (BSplineBasis1D): B-spline basis for the z dimension.

    Methods:
        evaluate(x, y, z, i, j, k): Evaluates the tensor product of 3D B-spline basis functions at a given point (x, y, z).
        evaluate_sparse(x, y, z, i, j, k): Evaluates the tensor product of 3D B-spline basis functions at a given point (x, y, z) in a sparse format.
        create_adaptive_knots_3d: Creates adaptive knot vectors for each dimension based on a 3D potential function.
    """
    def __init__(
            self, 
            domain: tuple, 
            degrees: tuple, 
            num_knots: tuple, 
            potential_function=None, 
            use_potential: bool = False
        ):
        """
        Initializes a 3D B-spline basis adapted to a potential function.

        Args:
            domain (tuple): Domain ranges for each dimension.
            degrees (tuple): Degrees of the B-splines for each dimension.
            num_knots (tuple): Number of knots for each dimension.
            potential_function (callable, optional): The trap potential function V(x, y, z).
            use_potential (bool, optional): If True, use the potential function to adaptively place knots.
        """
        if use_potential and potential_function:
            knots_x, knots_y, knots_z = self.create_adaptive_knots_3d(potential_function, domain, num_knots, degrees)
        else:
            knots_x = np.linspace(domain[0][0], domain[0][1], num_knots[0] + degrees[0] + 1)
            knots_y = np.linspace(domain[1][0], domain[1][1], num_knots[1] + degrees[1] + 1)
            knots_z = np.linspace(domain[2][0], domain[2][1], num_knots[2] + degrees[2] + 1)
        
        self.x_basis = BSplineBasis1D(domain[0], degrees[0], num_knots[0], knots_x)
        self.y_basis = BSplineBasis1D(domain[1], degrees[1], num_knots[1], knots_y)
        self.z_basis = BSplineBasis1D(domain[2], degrees[2], num_knots[2], knots_z)

    def evaluate(self, x, y, z, i, j, k):
        """
        Evaluates the tensor product of 3D B-spline basis functions at a given point (x, y, z).
        x, y, z can be either floats or arrays.

        Args:
            x, y, z: Coordinates, can be floats or NumPy arrays.
            i, j, k: Indices of the basis function in the x, y, z dimensions.

        Returns:
            The value(s) of the 3D B-spline basis function at (x, y, z).
        """
        x_val = np.array(self.x_basis.evaluate(x, i)).reshape(-1, 1, 1)
        y_val = np.array(self.y_basis.evaluate(y, j)).reshape(1, -1, 1)
        z_val = np.array(self.z_basis.evaluate(z, k)).reshape(1, 1, -1)

        return x_val * y_val * z_val

    def evaluate_sparse(self, x, y, z, i, j, k):
        """
        Evaluates the tensor product of 3D B-spline basis functions at a given point (x, y, z) in a sparse format.
        x, y, z can be either floats or arrays.

        Args:
            x, y, z: Coordinates, can be floats or NumPy arrays.
            i, j, k: Indices of the basis function in the x, y, z dimensions.

        Returns:
            csr_matrix: Sparse matrix representation of the 3D B-spline basis function evaluation.
        """
        x_eval = self.x_basis.evaluate_sparse(x, i)
        y_eval = self.y_basis.evaluate_sparse(y, j)
        z_eval = self.z_basis.evaluate_sparse(z, k)

        # Handling the tensor product in sparse format
        if isinstance(x, np.ndarray):
            x_eval = csr_matrix(x_eval)
        else:
            x_eval = csr_matrix(x_eval).transpose()

        if isinstance(y, np.ndarray):
            y_eval = csr_matrix(y_eval)
        else:
            y_eval = csr_matrix(y_eval).transpose()

        if isinstance(z, np.ndarray):
            z_eval = csr_matrix(z_eval)
        else:
            z_eval = csr_matrix(z_eval).transpose()

        # Compute tensor product using Kronecker product
        xy_eval = kron(x_eval, y_eval)
        xyz_eval = kron(xy_eval, z_eval)

        return xyz_eval

    def create_adaptive_knots_3d(
            self, 
            potential_function, 
            domain: tuple, 
            num_knots: tuple, 
            degrees: tuple, 
            num_samples: int = 50,
        ) -> tuple:
        """
        Creates adaptive knot vectors for each dimension based on a 3D potential function.

        Args:
            potential_function (callable): The trap potential function V(x, y, z).
            domain (tuple): Domain ranges for each dimension.
            num_knots (tuple): Number of knots for each dimension.
            degrees (tuple): Degrees of the B-splines for each dimension.
            num_samples (int, optional): Number of samples for analyzing the potential.

        Returns:
            tuple: A tuple containing three arrays of knots for the x, y, and z dimensions.
        """
        x_samples = np.linspace(domain[0][0], domain[0][1], num_samples)
        y_samples = np.linspace(domain[1][0], domain[1][1], num_samples)
        z_samples = np.linspace(domain[2][0], domain[2][1], num_samples)

        X, Y, Z = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')
        V = potential_function(X, Y, Z)

        grad_x, grad_y, grad_z = np.gradient(V, x_samples, y_samples, z_samples)

        grad_x_agg = np.max(np.abs(grad_x), axis=(1, 2))
        grad_y_agg = np.max(np.abs(grad_y), axis=(0, 2))
        grad_z_agg = np.max(np.abs(grad_z), axis=(0, 1))

        def create_knots_from_gradient(
                gradient: np.ndarray, 
                axis_samples: np.ndarray, 
                num_knots_dim: int, 
                degree: int,
            ) -> np.ndarray:
            """
            Creates a knot vector from a gradient profile of the potential function.

            This function analyzes the gradient along one dimension and determines the placement of knots
            such that more knots are placed in regions with higher gradient magnitude.

            Args:
                gradient (np.ndarray): Gradient of the potential function along one dimension.
                axis_samples (np.ndarray): Sample points along the axis.
                num_knots_dim (int): Number of knots for this dimension.
                degree (int): Degree of the B-spline.

            Returns:
                np.ndarray: Array of knot positions for this dimension.
            """
            points = np.column_stack((axis_samples, gradient))
            hull = ConvexHull(points)
            hull_points = np.unique(points[hull.vertices, 0])
            knots = np.linspace(axis_samples[0], axis_samples[-1], num_knots_dim - len(hull_points))
            knots = np.sort(np.concatenate((knots, hull_points)))
            return np.concatenate(([axis_samples[0]] * degree, knots, [axis_samples[-1]] * degree))

        knots_x = create_knots_from_gradient(grad_x_agg, x_samples, num_knots[0], degrees[0])
        knots_y = create_knots_from_gradient(grad_y_agg, y_samples, num_knots[1], degrees[1])
        knots_z = create_knots_from_gradient(grad_z_agg, z_samples, num_knots[2], degrees[2])

        return knots_x, knots_y, knots_z
