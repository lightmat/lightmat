from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence



class Lattice1d():

    def __init(
            self,
            lattice_direction: Sequence[float] = np.array([1, 0, 0]),
            theta: Union[u.Quantity, float] = 0,
    ):
        pass