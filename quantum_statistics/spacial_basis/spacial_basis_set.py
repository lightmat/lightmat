import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence
from astropy.units import Quantity
from .spacial_basis_function import SpacialBasisFunction


class SpacialBasisSet(ABC):
    def __init__(
            self,
            basis_functions: Union[SpacialBasisFunction, Sequence[SpacialBasisFunction], np.ndarray],
        ):      
        if isinstance(basis_functions, SpacialBasisFunction):
            self.basis_functions = [basis_functions]
        elif isinstance(basis_functions, (Sequence, np.ndarray)):
            if any(not isinstance(f, SpacialBasisFunction) for f in basis_functions):
                raise TypeError("basis_functions must be a BasisFunction or Sequence of BasisFunctions.")
            self.basis_functions = list(basis_functions)

        self.N_basis_functions = len(self.basis_functions)

    @abstractmethod
    def expand(
        self,
        coeffs: Union[float, Sequence[float], np.ndarray],
        x: Union[float, Sequence[float], np.ndarray, Quantity],
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def overlap_matrix(
        self,
    ) -> np.ndarray:
        pass


