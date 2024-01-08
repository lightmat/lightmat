import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable
from astropy.units import Quantity


class SpacialBasisSet(ABC):
    def __init__(
            self,
        ):      
        self.N_basis_functions = None

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


