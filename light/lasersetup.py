from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .gaussianbeam import GaussianBeam
from .lattice1d import Lattice1d
from .bowtie_lattice2d import BowtieLattice2d


class LaserSetup(object):
    
        def __init__(
                self,
                beams: Union[GaussianBeam, Lattice1d, BowtieLattice2d, Sequence[Union[GaussianBeam, Lattice1d, BowtieLattice2d]]],
        ) -> None:
            """Initializes a LaserSetup instance.
    
            Args:
                    beams: A single or sequence of GaussianBeam, Lattice1d or BowtieLattice2d instances.
    
            Returns:
                    None           
            """
            self.beams = beams
            self._check_input('init')



        def E_vec(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> u.Quantity:
            """Returns the electric field vector of the laser setup at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The electric field vector of the lattice beam at the given position.
            """
            Evec = np.zeros((3, len(x), len(y), len(z))) * u.V / u.m
            for beam in self.beams:
                Evec = Evec + beam.E_vec(x, y, z)

            return Evec
        

        def E(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> u.Quantity:
            """Returns the electric field of the laser setup at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The electric field of the lattice beam at the given position.
            """
            return np.linalg.norm(self.E_vec(x, y, z), axis=0)
        

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
            return c*eps0/2 * np.abs(self.E(x, y, z))**2
        



        def _check_input(
                    self, 
                    method,
        ) -> None:
            """Checks the input for the LaserSetup instance."""
            if method == 'init':
                if not isinstance(self.beams, (GaussianBeam, Lattice1d, BowtieLattice2d, Sequence)):
                    raise ValueError("The input beams must be a single or sequence of GaussianBeam, Lattice1d or BowtieLattice2d instances.")
                if isinstance(self.beams, Sequence):
                    for beam in self.beams:
                        if not isinstance(beam, (GaussianBeam, Lattice1d, BowtieLattice2d)):
                            raise ValueError("The input beams must be a single or sequence of GaussianBeam, Lattice1d or BowtieLattice2d instances.")
