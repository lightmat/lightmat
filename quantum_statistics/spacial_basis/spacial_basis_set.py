import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable
from astropy.units import Quantity


class SpacialBasisSet(ABC):
    def __init__(
            self,
            basis_functions: Union[Callable, Sequence[Callable], np.ndarray],
        ):      
        if isinstance(basis_functions, Callable):
            self.basis_functions = np.array([basis_functions])
        elif isinstance(basis_functions, (Sequence, np.ndarray)):
            if any(not isinstance(f, Callable) for f in basis_functions):
                raise TypeError("basis_functions must be a Callable or sequence of Callables.")
            self.basis_functions = np.asarray(basis_functions)

        self.N_basis_functions = len(self.basis_functions)

    @abstractmethod
    def expand(
        self,
        coeffs: Union[float, Sequence[float], np.ndarray],
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ) -> np.ndarray:
        pass

    @abstractmethod
    def overlap_matrix(
        self,
    ) -> np.ndarray:
        pass


