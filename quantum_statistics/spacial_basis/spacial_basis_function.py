import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable
from astropy.units import Quantity


class SpacialBasisFunction(ABC):
    def __init__(
            self,
            # everything necessary to implement the eval() method
        ):
        pass

    @abstractmethod
    def eval(
        self, 
        x: Union[float, Sequence[float], np.ndarray, Quantity], 
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        sparse: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def calculate_laplacian(
        self,
    ) -> Callable:
        pass

    @abstractmethod
    def eval_laplacian(
        self, 
        x: Union[float, Sequence[float], np.ndarray, Quantity], 
        y: Union[float, Sequence[float], np.ndarray, Quantity],
        z: Union[float, Sequence[float], np.ndarray, Quantity],
        sparse: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def plot(
        self,
        **kwargs,
    ):
        pass

    