import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable
from astropy.units import Quantity


class SpatialBasisSet(ABC):
    def __init__(
            self,
        ):      
        self.domain = None
        self.num_basis_funcs = None




