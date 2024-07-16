from abc import ABC, abstractmethod
from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence

class Beam(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        beam_direction: np.ndarray,
        lambda_: u.Quantity,
        P: u.Quantity,
        pol_Jones_vec: np.ndarray,
        pol_vec_3d: np.ndarray,
    ) -> None:
        """
        Initializes the abstract Beam() class.

        Args:
            name: Name of the beam.
            beam_direction: 3d vector specifying the direction of the beam in the standard Carteesian coordinate system.
            lambda_: Wavelength of the beam in unit equivalent to [nm].
            P: Power of the beam in unit equivalent to [W].
            pol_Jones_vec: Complex 2d vector specifying the polarization of the beam in the local coordinate system
                            where the beam propagates along the local z-direction. The convention is that the
                            horizontal polarization is along the local x-direction and the vertical polarization
                            along the local y-direction.
            pol_3d_vec: Complex 3d vector specifying the polarization of the beam in the global standard Carteesian
                        coordinate system. It is perpendicular to ``_beam_direction_vec``.
        """
        self.name = name
        self.beam_direction = beam_direction
        self.lambda_ = lambda_
        self.P = P
        self.pol_Jones_vec = pol_Jones_vec
        self.pol_vec_3d = pol_vec_3d

        self.k = (2*np.pi / (self.lambda_)).to(1/u.um)
        self.omega = (2*np.pi*c/self.lambda_).to(u.THz)
        
    

    @abstractmethod
    def E_vec(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field vector of the beam at the position (x,y,z) in [V/m] in the standard Carteesian coordinate system.
           This is just the complex electric field amplitude multiplied by the 3d polarization vector. Here, x, y, z are the global
           standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().
           
           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].
                
           Returns:
                u.Quantity: Complex electric field vector of the beam at the position (x,y,z) in [V/m] in the standard Carteesian coordinate system.
        """
        pass


    @abstractmethod
    def E(
            self, 
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field amplitude of the laser at the position (x,y,z) in [V/m]. 
           Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Complex electric field amplitude of the beam at the position (x,y,z) in [V/m], can be either float or array.
        """
        pass

    @abstractmethod
    def I(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:    
        """Returns the intensity of the laser at the position (x,y,z) in [mW/cm^2]. Here, x, y, z are the global standard Carteesian
           coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Intensity of the beam at the position (x,y,z) in [mW/cm^2], can be either float or array.
        """
        pass
