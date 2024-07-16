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
                lambda_: Union[u.Quantity, float],
                w0: Union[u.Quantity, float, Sequence[float], np.ndarray],
                P: Union[u.Quantity, float],
                pol_Jones_vec: Union[str, Sequence[float]] = 'linear vertical',
                z0: Union[u.Quantity, float] = 0 * u.um,
                name: str = 'BowtieLattice2d',
                color: str = None,
        ) -> None:
            """Initializes a BowtieLattice2d instance.
    
            Args:
                    lattice_direction_vec1: 3d vector specifying the first lattice propagation in the global standard Carteesian 
                                            coordinate system.
                    lattice_direction_vec2: 3d vector specifying the second lattice propagation in the global standard Carteesian 
                                            coordinate system.
                    lambda_: Wavelength of the beam in [nm]. 
                    w0: Beam waist diameter in [um]. Either a scalar for circular beams or a sequence of two floats for
                        elliptical beams having different beam waist diameters in local x- and y-direction.
                    P: Power of the beam in [W].
                    pol_Jones_vec: 2d vector specifying the polarization of the beam in the local coordinate system
                               where the beam propagates along the local z-direction. The convention is that the
                               horizontal polarization is along the local x-direction and the vertical polarization
                               along the local y-direction. On can either directly specify the 2d vector or use the
                               strings 'linear horizontal', 'linear vertical', 'circular right', 'circular left' to
                               specify the polarization. Defaults to 'linear vertical'.
                    z0: Position of the beam waist in [um] along the beam propagation direction. Defaults to 0um.
    
            Returns:
                    None           
            """
            self.lattice_direction_vec1 = np.asarray(lattice_direction_vec1)
            self.lattice_direction_vec2 = np.asarray(lattice_direction_vec2)
            self.pol_Jones_vec = pol_Jones_vec
            self.lambda_ = lambda_
            self.w0 = w0
            self.P = P
            self.z0 = z0
            self.name = name
            self.color = color
            self._check_input('init')

            super().__init__(
                self.name,
                [self.lattice_direction_vec1, -self.lattice_direction_vec1, 
                 self.lattice_direction_vec2, -self.lattice_direction_vec2,],
                self.lambda_, 
                self.P,
                self.color,
            )
    
            self.beam_forward1 = GaussianBeam(
                beam_direction_vec=self.lattice_direction_vec1,
                pol_Jones_vec=self.pol_Jones_vec,
                lambda_=self.lambda_,
                w0=self.w0,
                P=self.P,
                z0=self.z0,
            )

            self.beam_forward2 = GaussianBeam(
                beam_direction_vec=self.lattice_direction_vec2,
                pol_Jones_vec=self.pol_Jones_vec,
                lambda_=self.lambda_,
                w0=self.w0,
                P=self.P,
                z0=self.z0,
            )

            self.beam_backward1 = GaussianBeam(
                beam_direction_vec=-self.lattice_direction_vec1,
                pol_Jones_vec=self.pol_Jones_vec,
                lambda_=self.lambda_,
                w0=self.w0,
                P=self.P,
                z0=self.z0,
            )

            self.beam_backward2 = GaussianBeam(
                beam_direction_vec=-self.lattice_direction_vec2,
                pol_Jones_vec=self.pol_Jones_vec,
                lambda_=self.lambda_,
                w0=self.w0,
                P=self.P,
                z0=self.z0,
            )



        def E_vec(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> u.Quantity:
            """Returns the electric field vector of the 2d lattice at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The electric field vector of the lattice beam at the given position.
            """
            E_vec = self.beam_forward1.E_vec(x, y, z) + self.beam_backward1.E_vec(x, y, z) + \
                    self.beam_forward2.E_vec(x, y, z) + self.beam_backward2.E_vec(x, y, z)
            return E_vec.to(u.V/u.m)
        

        def E(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> np.ndarray:
            """Returns the electric field of the 2d lattice at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The electric field of the lattice beam at the given position.
            """
            return np.linalg.norm(self.E_vec(x, y, z), axis=0).to(u.V/u.m)
        

        def I(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> np.ndarray:
            """Returns the intensity of the 2d lattice at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The intensity of the lattice beam at the given position.
            """
            return (c*eps0/2 * np.abs(self.E(x, y, z))**2).to(u.mW/u.cm**2)
        



        def E_vec_sym(
                self,
                x: sp.Symbol,
                y: sp.Symbol,
                z: sp.Symbol,
        ):
            E_vec = self.beam_forward1.E_vec_sym(x, y, z) + self.beam_backward1.E_vec_sym(x, y, z) + \
                    self.beam_forward2.E_vec_sym(x, y, z) + self.beam_backward2.E_vec_sym(x, y, z)
            return E_vec
        

        def E_sym(
                self,
                x: sp.Symbol,
                y: sp.Symbol,
                z: sp.Symbol,
        ):
            return self.E_vec_sym(x, y, z).norm()
        

        def I_sym(
                self,
                x: sp.Symbol,
                y: sp.Symbol,
                z: sp.Symbol,
        ):
            E = self.E_sym(x, y, z)
            I = (c.to(u.m/u.s).value*eps0.value/2 * abs(E)**2)

            return I
        


        def _check_input(
                self, 
                method: str,
        ) -> None:
            """Checks the input of the method ``method``."""
            if method == 'init':
                # Check lattice direction vector 1
                if not isinstance(self.lattice_direction_vec1, (Sequence, np.ndarray)): 
                    raise TypeError('The beam_direction_vec must be a sequence.')
                if not len(self.lattice_direction_vec1) == 3:
                    raise ValueError('The beam_direction_vec must be a 3d vector.')
                self.lattice_direction_vec1 = np.asarray(self.lattice_direction_vec1)
                self.lattice_direction_vec1 = self.lattice_direction_vec1 / np.linalg.norm(self.lattice_direction_vec1)  # Normalize

                # Check lattice direction vector 2
                if not isinstance(self.lattice_direction_vec2, (Sequence, np.ndarray)):
                    raise TypeError('The beam_direction_vec must be a sequence.')
                if not len(self.lattice_direction_vec2) == 3:
                    raise ValueError('The beam_direction_vec must be a 3d vector.')
                self.lattice_direction_vec2 = np.asarray(self.lattice_direction_vec2)
                self.lattice_direction_vec2 = self.lattice_direction_vec2 / np.linalg.norm(self.lattice_direction_vec2)  # Normalize


                # Check wavelength
                # ... will be checked in GaussianBeam()

                
                # Check beam waist diameter
                # ... will be checked in GaussianBeam()

                
                # Check power
                # ... will be checked in GaussianBeam()
                

                # Check polarization Jones vector
                # ... will be checked in GaussianBeam()
                
                
                # Check position of the beam waist
                # ... will be checked in GaussianBeam()

                
                # Check name
                if not isinstance(self.name, str):
                    raise TypeError('The name must be a string.')