from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .laser import Laser


class LaserSetup(object):
    
        def __init__(
                self,
                lasers: Union[Laser, Sequence[Laser]],
        ) -> None:
            """Initializes a LaserSetup instance.
    
            Args:
                    lasers: A single or sequence of Laser instances.
    
            Returns:
                    None           
            """
            self.lasers = lasers
            self._check_input('init')

        

        def I(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> u.Quantity:
            """Returns the intensity of the laser setup at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The intensity of the lattice beam at the given position.
            """
            I = 0 * u.mW / u.cm**2
            for laser in self.lasers:
                I = I + laser.I(x, y, z)

            return I        



        def _check_input(
                    self, 
                    method,
        ) -> None:
            """Checks the input for the LaserSetup instance."""
            if method == 'init':
                if not isinstance(self.lasers, (Laser, Sequence)):
                    raise ValueError("The input beams must be a single or sequence of Laser instances.")
                if isinstance(self.lasers, Sequence):
                    for laser in self.lasers:
                        if not isinstance(laser, Laser):
                            raise ValueError("The input beams must be a single or sequence of Laser instances.")
