import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.sparse import csr_matrix
from typing import Union, Sequence, Callable
from astropy.units import Quantity
from scipy.interpolate import BSpline

from .spacial_basis_function import SpacialBasisFunction


class BSplineSpacialBasisFunction(SpacialBasisFunction):
    def __init__(
            self,
            spline_x: BSpline,
            spline_y: BSpline,
            spline_z: BSpline,
        ):
        super().__init__()

        if isinstance(spline_x, BSpline):
            self.spline_x = spline_x
        else:
            raise TypeError("spline_x must be a BSpline.")
        if isinstance(spline_y, BSpline):
            self.spline_y = spline_y
        else:
            raise TypeError("spline_y must be a BSpline.")
        if isinstance(spline_z, BSpline):
            self.spline_z = spline_z
        else:
            raise TypeError("spline_z must be a BSpline.")
        

    def eval(
        self, 
        x: Union[float, Sequence[float], np.ndarray, Quantity], 
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        sparse: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        self.x = x
        self.y = y
        self.z = z
        self._check_xyz()
        
        # Evaluate the spline
        xval = np.array([self.spline_x(xi) for xi in self.x])
        yval = np.array([self.spline_y(yi) for yi in self.y])
        zval = np.array([self.spline_z(zi) for zi in self.z])
        result = np.outer(xval, np.outer(yval, zval))

        # Convert to sparse matrix if required
        if sparse:
            result = csr_matrix(result)

        return result

    def calculate_laplacian(
        self,
    ) -> Callable:
        def laplacian(x, y, z):
            # Evaluate the second derivatives of the B-splines
            d2x = self.spline_x.derivative(2)(x)
            d2y = self.spline_y.derivative(2)(y)
            d2z = self.spline_z.derivative(2)(z)

            # Compute the Laplacian
            return d2x * self.spline_y(y) * self.spline_z(z) + \
                self.spline_x(x) * d2y * self.spline_z(z) + \
                self.spline_x(x) * self.spline_y(y) * d2z

        return laplacian


    def eval_laplacian(
        self, 
        x: Union[float, Sequence[float], np.ndarray, Quantity], 
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        sparse: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        self.x = x
        self.y = y
        self.z = z
        self._check_xyz()

        if not hasattr(self, "laplacian"):
            self.laplacian = self.calculate_laplacian()

        # Evaluate the Laplacian
        result = self.laplacian(self.x, self.y, self.z)

        # Convert to sparse matrix if required
        if sparse:
            result = csr_matrix(result)

        return result


    def plot(
        self,
        x: Union[float, Sequence[float], np.ndarray, Quantity], 
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        **kwargs,
    ):
        self.x = x
        self.y = y
        self.z = z
        self._check_xyz()

        plot_dim = sum(len(arr) > 1 for arr in [self.x, self.y, self.z])
        if plot_dim == 0:
            raise ValueError("At least one of x, y, or z must be a sequence of length > 1.")
        elif plot_dim == 1:
            self._plot_1d(**kwargs)
        elif plot_dim == 2:
            self._plot_2d(**kwargs)
        elif plot_dim == 3:
            raise ValueError("Cannot plot a 3D function, at least one of x,y,z have to be a number, not a sequence.")
        

    def plot_laplacian(
            self,
            x: Union[float, Sequence[float], np.ndarray, Quantity],
            y: Union[float, Sequence[float], np.ndarray, Quantity],
            z: Union[float, Sequence[float], np.ndarray, Quantity],
            **kwargs,
        ):
        self.x = x
        self.y = y
        self.z = z
        self._check_xyz()

        plot_dim = sum(len(arr) > 1 for arr in [self.x, self.y, self.z])
        if plot_dim == 0:
            raise ValueError("At least one of x, y, or z must be a sequence of length > 1.")
        elif plot_dim == 1:
            self._plot_1d(**kwargs, laplacian=True)
        elif plot_dim == 2:
            self._plot_2d(**kwargs, laplacian=True)
        elif plot_dim == 3:
            raise ValueError("Cannot plot a 3D function, at least one of x,y,z have to be a number, not a sequence.")
        

    def _plot_1d(
        self,
        laplacian: bool = False,
        **kwargs,
    ):
        fig, ax = plt.subplots(**kwargs)
        if len(self.x) > 1:
            if laplacian:
                ax.plot(self.x, self.eval_laplacian(self.x, self.y, self.z, sparse=False), **kwargs)
            else:
                ax.plot(self.x, self.eval(self.x, self.y, self.z, sparse=False), **kwargs)
            ax.set_xlabel("x", **kwargs)
            ax.set_ylabel("f(x, " + str(self.y[0]) + ", " + str(self.z[0]) + ")", **kwargs)
        elif len(self.y) > 1:
            if laplacian:
                ax.plot(self.y, self.eval_laplacian(self.x, self.y, self.z, sparse=False), **kwargs)
            else:
                ax.plot(self.y, self.eval(self.x, self.y, self.z, sparse=False), **kwargs)
            ax.set_xlabel("y", **kwargs)
            ax.set_ylabel("f(" + str(self.x[0]) + ", y, " + str(self.z[0]) + ")", **kwargs)
        elif len(self.z) > 1:
            if laplacian:
                ax.plot(self.z, self.eval_laplacian(self.x, self.y, self.z, sparse=False), **kwargs)
            else:
                ax.plot(self.z, self.eval(self.x, self.y, self.z, sparse=False), **kwargs)
            ax.set_xlabel("z", **kwargs)
            ax.set_ylabel("f(" + str(self.x[0]) + ", " + str(self.y[0]) + ", z)", **kwargs)


    def _plot_2d(
            self,
            laplacian: bool = False,
            **kwargs,
        ):
        fig, ax = plt.subplots(**kwargs)
        if len(self.x) > 1 and len(self.y) > 1:
            if laplacian:
                ax.imshow(self.eval_laplacian(self.x, self.y, self.z, sparse=False), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], **kwargs)
            else:
                ax.imshow(self.eval(self.x, self.y, self.z, sparse=False), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], **kwargs)
            ax.set_xlabel("x", **kwargs)
            ax.set_ylabel("y", **kwargs)
        elif len(self.x) > 1 and len(self.z) > 1:
            if laplacian:
                ax.imshow(self.eval_laplacian(self.x, self.y, self.z, sparse=False), extent=[self.x[0], self.x[-1], self.z[0], self.z[-1]], **kwargs)
            else:
                ax.imshow(self.eval(self.x, self.y, self.z, sparse=False), extent=[self.x[0], self.x[-1], self.z[0], self.z[-1]], **kwargs)
            ax.set_xlabel("x", **kwargs)
            ax.set_ylabel("z", **kwargs)
        elif len(self.y) > 1 and len(self.z) > 1:
            if laplacian:
                ax.imshow(self.eval_laplacian(self.x, self.y, self.z, sparse=False), extent=[self.y[0], self.y[-1], self.z[0], self.z[-1]], **kwargs)
            else:
                ax.imshow(self.eval(self.x, self.y, self.z, sparse=False), extent=[self.y[0], self.y[-1], self.z[0], self.z[-1]], **kwargs)
            ax.set_xlabel("y", **kwargs)
            ax.set_ylabel("z", **kwargs)


    def _check_xyz(
        self,
    ):
        # Check input
        if isinstance(self.x, Quantity):
            self.x = self.x.to(u.um).value
        if isinstance(self.x, (float, int)):
            self.x = np.array([self.x])
        elif isinstance(self.x, (Sequence, np.ndarray)):
            self.x = np.array(self.x)
        else:
            raise TypeError("x must be a float, int, Sequence, or Quantity.")

        if isinstance(self.y, Quantity):
            self.y = self.y.to(u.um).value
        if isinstance(self.y, (float, int)):
            self.y = np.array([self.y])
        elif isinstance(self.y, (Sequence, np.ndarray)):
            self.y = np.array(self.y)
        else:
            raise TypeError("y must be a float, int, Sequence, or Quantity.")
        
        if isinstance(self.z, Quantity):
            self.z = self.z.to(u.um).value
        if isinstance(self.z, (float, int)):
            self.z = np.array([self.z])
        elif isinstance(self.z, (Sequence, np.ndarray)):
            self.z = np.array(self.z)
        else:
            raise TypeError("z must be a float, int, Sequence, or Quantity.")