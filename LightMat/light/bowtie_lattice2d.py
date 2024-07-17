from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .laser import Laser
from .beams import GaussianBeam
from .lattice1d import Lattice1d


class BowtieLattice2d(Laser):
    
        def __init__(
                self,
                lattice_direction_vec1: Sequence[float],
                lattice_direction_vec2: Sequence[float],
                lambda_: Union[u.Quantity, float, Sequence[float], np.ndarray],
                w0: Union[u.Quantity, float, Sequence[float], np.ndarray],
                P: Union[u.Quantity, float, Sequence[float], np.ndarray],
                pol_Jones_vec: Union[str, Sequence[float], Sequence[str]] = 'linear horizontal',
                z0: Union[u.Quantity, float, Sequence[float], np.ndarray] = 0 * u.um,
                name: str = 'BowtieLattice2d',
        ) -> None:
            """Initializes a BowtieLattice2d instance. The 2d bowtie lattice just consists of two 1d lattices, but all 4 beams
               are from the same laser and interfere with each other.
    
            Args:
                lattice_direction_vec1: 3d vector specifying the first lattice propagation in the global standard Carteesian 
                                        coordinate system.
                lattice_direction_vec2: 3d vector specifying the second lattice propagation in the global standard Carteesian 
                                        coordinate system.
                lambda_: Wavelength of the beams in [nm]. Can either be a scalar or 4d vector specifying different wavelengths
                         for all beams.
                w0: Beam waist diameter in [um]. Either a scalar for the same circular beam waist for all beams or a 2d vector
                    for the same elliptical beam waist for all beams, or a list of four 2d vectors for different beam waists
                    for all lattice beams.
                P: Power of the beam in [W]. Either a scalar for the same power in all beams or a 4d vector for different power
                   in the four lattice beams.
                pol_Jones_vec: 2d vector specifying the polarization of the beam in the local x-y plane. The convention 
                               is that the horizontal polarization is along the local x-direction and the vertical polarization
                               along the local y-direction. Either a 2d vector for the same polarization for all beams or a list
                               of four 2d vectors for different polarizations for the four lattice beams. Alternatively to the 2d
                               vector, a string in ['linear horizontal', 'linear vertical', 'circular right', 'circular left'] 
                               can be passed to specify the polarization. Defaults to 'linear horizontal'.
                z0: Position of the beam waist in [um] along the beam propagation direction. Either a scalar for same beam waist
                    position for all beams or a 4d vector for different beam waist positions for the four lattice beams.
                    Defaults to 0um.
                name: Name of the Lattice1d instance. Defaults to 'BowtieLattice2d'.
    
            Returns:
                    None           
            """
            self.lattice_direction_vec1 = np.asarray(lattice_direction_vec1)
            self.lattice_direction_vec2 = np.asarray(lattice_direction_vec2)
            self.lambda_ = lambda_
            self.w0 = w0
            self.P = P
            self.pol_Jones_vec = pol_Jones_vec
            self.z0 = z0
            self.name = name
            self._check_input('init')

    
            self.beam_forward1 = GaussianBeam(
                beam_direction_vec=self.lattice_direction_vec1,
                pol_Jones_vec=self.pol_Jones_vec if isinstance(self.pol_Jones_vec, str) or \
                                                    (self.pol_Jones_vec.ndim == 1 and not isinstance(self.pol_Jones_vec[0], str)) else self.pol_Jones_vec[0],
                lambda_=self.lambda_ if self.lambda_.isscalar else self.lambda_[0],
                w0=self.w0 if self.w0.ndim == 1 or self.w0.isscalar else self.w0[0],
                P=self.P if self.P.isscalar else self.P[0],
                z0=self.z0 if self.z0.isscalar else self.z0[0],
            )

            self.beam_forward2 = GaussianBeam(
                beam_direction_vec=self.lattice_direction_vec2,
                pol_Jones_vec=self.pol_Jones_vec if isinstance(self.pol_Jones_vec, str) or \
                                                    (self.pol_Jones_vec.ndim == 1 and not isinstance(self.pol_Jones_vec[0], str)) else self.pol_Jones_vec[1],
                lambda_=self.lambda_ if self.lambda_.isscalar else self.lambda_[1],
                w0=self.w0 if self.w0.ndim == 1 or self.w0.isscalar else self.w0[1],
                P=self.P if self.P.isscalar else self.P[1],
                z0=self.z0 if self.z0.isscalar else self.z0[1],
            )

            self.beam_backward1 = GaussianBeam(
                beam_direction_vec=-self.lattice_direction_vec1,
                pol_Jones_vec=self.pol_Jones_vec if isinstance(self.pol_Jones_vec, str) or \
                                                    (self.pol_Jones_vec.ndim == 1 and not isinstance(self.pol_Jones_vec[0], str)) else self.pol_Jones_vec[2],
                lambda_=self.lambda_ if self.lambda_.isscalar else self.lambda_[2],
                w0=self.w0 if self.w0.ndim == 1 or self.w0.isscalar else self.w0[2],
                P=self.P if self.P.isscalar else self.P[2],
                z0=self.z0 if self.z0.isscalar else self.z0[2],
            )

            self.beam_backward2 = GaussianBeam(
                beam_direction_vec=-self.lattice_direction_vec2,
                pol_Jones_vec=self.pol_Jones_vec if isinstance(self.pol_Jones_vec, str) or \
                                                    (self.pol_Jones_vec.ndim == 1 and not isinstance(self.pol_Jones_vec[0], str)) else self.pol_Jones_vec[3],
                lambda_=self.lambda_ if self.lambda_.isscalar else self.lambda_[3],
                w0=self.w0 if self.w0.ndim == 1 or self.w0.isscalar else self.w0[3],
                P=self.P if self.P.isscalar else self.P[3],
                z0=self.z0 if self.z0.isscalar else self.z0[3],
            )


            super().__init__(
                name = self.name,
                beams = [self.beam_forward1, self.beam_backward1, self.beam_forward2, self.beam_backward2], 
            )
        


        def _check_input(
                self, 
                method: str,
        ) -> None:
            """Checks the input of the method ``method``."""
            if method == 'init':
                # Check lattice direction vectors
                for lattice_direction_vec in [self.lattice_direction_vec1, self.lattice_direction_vec2]:
                    if not isinstance(lattice_direction_vec, (Sequence, np.ndarray)): 
                        raise TypeError('The beam_direction_vec must be a sequence.')
                    if not len(lattice_direction_vec) == 3:
                        raise ValueError('The beam_direction_vec must be a 3d vector.')
                self.lattice_direction_vec1 = np.asarray(self.lattice_direction_vec1)
                self.lattice_direction_vec1 = self.lattice_direction_vec1 / np.linalg.norm(self.lattice_direction_vec1)  # Normalize
                self.lattice_direction_vec2 = np.asarray(self.lattice_direction_vec2)
                self.lattice_direction_vec2 = self.lattice_direction_vec2 / np.linalg.norm(self.lattice_direction_vec2)  # Normalize


                # Check wavelength
                if isinstance(self.lambda_, u.Quantity):
                    if not self.lambda_.unit.is_equivalent(u.nm):
                        raise ValueError('The wavelength must be in units equivalent to nm.')
                elif isinstance(self.lambda_, (float, int)):
                    self.lambda_ = self.lambda_ * u.nm
                elif isinstance(self.lambda_, (Sequence, np.ndarray)):
                    if len(self.lambda_) != 4:
                        raise ValueError('The wavelength must be a scalar or a 4d vector.')
                    self.lambda_ = np.asarray(self.lambda_) * u.nm

                
                # Check beam waist diameter
                if isinstance(self.w0, u.Quantity):
                    if not self.w0.unit.is_equivalent(u.um):
                        raise ValueError('The beam waist diameter must be in units equivalent to um.')
                elif isinstance(self.w0, (float, int)):
                    self.w0 = self.w0 * u.um
                elif isinstance(self.w0, (Sequence, np.ndarray)):
                    self.w0 = np.asarray(self.w0)
                    if self.w0.ndim == 1:
                        if len(self.w0) != 2:
                            raise ValueError('The beam waist diameter must be a scalar or a 2d vector or a list of four 2d vectors.')
                    elif self.w0.ndim == 2:
                        if len(self.w0) != 4 or len(self.w0[0]) != 2 or len(self.w0[1]) != 2 or len(self.w0[2]) != 2 or len(self.w0[3]) != 2:
                            raise ValueError('The beam waist diameter must be a scalar or a 2d vector or a list of four 2d vectors.')
                    else:
                        raise ValueError('The beam waist diameter must be a scalar or a 2d vector or a list of four 2d vectors.')
                    self.w0 = self.w0 * u.um

                
                # Check power
                if isinstance(self.P, u.Quantity):
                    if not self.P.unit.is_equivalent(u.W):
                        raise ValueError('The power must be in units equivalent to W.')
                elif isinstance(self.P, (float, int)):
                    self.P = self.P * u.W
                elif isinstance(self.P, (Sequence, np.ndarray)):
                    self.P = np.asarray(self.P)
                    if len(self.P) != 4:
                        raise ValueError('The power must be a scalar or a 4d vector.')
                    self.P = self.P * u.W
                else:
                    raise ValueError('The power must be a scalar or a 4d vector.')
                

                # Check polarization Jones vector
                if isinstance(self.pol_Jones_vec, str):
                    if self.pol_Jones_vec not in ['linear horizontal', 'linear vertical', 'circular right', 'circular left']:
                        raise ValueError('The polarization Jones vector must be one of the following strings: ["linear horizontal", "linear vertical", "circular right", "circular left"].')
                elif isinstance(self.pol_Jones_vec, (Sequence, np.ndarray)):
                    self.pol_Jones_vec = np.asarray(self.pol_Jones_vec)
                    if self.pol_Jones_vec.ndim == 1:
                        if len(self.pol_Jones_vec) != 2:
                            raise ValueError('The polarization Jones vector must be a 2d vector or a list of four 2d vectors.')                    
                    elif self.pol_Jones_vec.ndim == 2:
                        if len(self.pol_Jones_vec) != 4 or len(self.pol_Jones_vec[0]) != 2 or len(self.pol_Jones_vec[1]) != 2 \
                            or len(self.pol_Jones_vec[2]) != 2 or len(self.pol_Jones_vec[3]) != 2:
                            raise ValueError('The polarization Jones vector must be a 2d vector or a list of four 2d vectors.')
                    else:
                        raise ValueError('The polarization Jones vector must be a 2d vector or a list of four 2d vectors.')
                else:
                    raise ValueError('The polarization Jones vector must be a 2d vector or string or a list of four 2d vectors or four strings.')
                
                
                # Check position of the beam waist
                if isinstance(self.z0, u.Quantity):
                    if not self.z0.unit.is_equivalent(u.um):
                        raise ValueError('The position of the beam waist must be in units equivalent to um.')
                elif isinstance(self.z0, (float, int)):
                    self.z0 = self.z0 * u.um
                elif isinstance(self.z0, (Sequence, np.ndarray)):
                    self.z0 = np.asarray(self.z0)
                    if len(self.z0) != 4:
                        raise ValueError('The position of the beam waist must be a scalar or a 4d vector.')
                    self.z0 = self.z0 * u.um
                else:
                    raise ValueError('The position of the beam waist must be a scalar or a 4d vector.')
                
                # Check name
                if not isinstance(self.name, str):
                    raise TypeError('The name must be a string.')