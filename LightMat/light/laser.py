from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence
import random
import math

random.seed(99)

from .beams import Beam

class Laser(object):
    def __init__(
        self,
        name: str,
        beams: Sequence[Beam],
        pol_vec_3d: Union[Sequence[float], np.ndarray, None] = None,
    ) -> None:
        """
        Initializes the laser. A laser can have several beams. The beams interfere with each other to 
        generate the total electric field of the laser.

        Args:
            name: Name of the laser.
            beams: Sequence of beams in the laser.
            pol_vec_3d: Complex 3d polarization vector of the laser's electric vector field. If it is 
                        None, it is tried to be determined as (E_vec / E) for the laser's electric field, 
                        but only if it's constant over space, else it is left at None. Default is None.
        """
        self.beams = beams
        self.name = name
        self.pol_vec_3d = pol_vec_3d
        self._check_input('init')

        if self.pol_vec_3d is None:
            self.pol_vec_3d = self._calculate_pol_vec_3d()

        if any(self.beams[0].lambda_ != beam.lambda_ for beam in self.beams):
            raise ValueError('All beams must have the same wavelength.')
        else:
            self.lambda_ = self.beams[0].lambda_
            self.omega = (2*np.pi*c/self.lambda_).to(u.THz)

        
    
    def E_vec(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the total complex electric field vector of the laser at the position (x,y,z) in [V/m] in the standard Carteesian 
           coordinate system. All ``beams`` of the laser interfere with each other. Here, x, y, z are the global standard Carteesian 
           coordinates in [um] and can be either float or array obtained from np.meshgrid().
           
           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].
                
           Returns:
                u.Quantity: Complex electric field vector of the laser at the position (x,y,z) in [V/m] in the standard Carteesian coordinate system.
        """
        Evecs = np.array([beam.E_vec(x, y, z).to(u.V/u.m).value for beam in self.beams]) 
        Evec = np.sum(Evecs, axis=0) * u.V/u.m
        return Evec



    def E(
            self, 
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field amplitude of the laser at the position (x,y,z) in [V/m]. All ``beams`` of the laser interfere 
           with each other. Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained 
           from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Complex electric field amplitude of the laser at the position (x,y,z) in [V/m], can be either float or array.
        """
        #return np.linalg.norm(self.E_vec(x, y, z), axis=0).to(u.V/u.m)
        Es = np.array([beam.E(x, y, z).to(u.V/u.m).value for beam in self.beams])
        E = np.sum(Es, axis=0) * u.V/u.m
        return E 
    


    def I(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:    
        """Returns the intensity of the laser at the position (x,y,z) in [mW/cm^2]. All ``beams`` of the laser interfere 
           with each other. Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float 
           or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Intensity of the laser at the position (x,y,z) in [mW/cm^2], can be either float or array.
        """
        return (c*eps0/2 * np.abs(self.E(x, y, z))**2).to(u.mW/u.cm**2)
    

    def _calculate_pol_vec_3d(
            self,
    ) -> np.ndarray:
        """Returns the complex 3D polarization vector of the laser.
        
           Returns:
                np.ndarray: Complex 3D polarization vector of the laser's electric vector field.
        """
        # Find a position where the electric field is non-zero and calculate the polarization vector
        #pol_vec_3d = np.array([0, 0, 0], dtype=np.complex128)
        #pol_vec_3d_alternative = np.array([0, 0, 0], dtype=np.complex128)
        #flag = True
        #maxiter = 1000
#
        #for _ in range(maxiter):
        #    x = random.random() * u.mm 
        #    y = random.random() * u.mm
        #    z = random.random() * u.mm
        #    E = self.E(x, y, z)
        #    if np.abs(E.value) > 1e-10 and np.abs(E.value) < 1e10:
        #        if flag:
        #            pol_vec_3d = (self.E_vec(x, y, z) / E).value
        #            flag = False
        #        else:
        #            pol_vec_3d_alternative = (self.E_vec(x, y, z) / E).value
        #            break
#
        #print('pol_vec_3d             :', pol_vec_3d)
        #print('pol_vec_3d_alternative :', pol_vec_3d_alternative)
        #print('\n')
#
        #if np.allclose(pol_vec_3d, pol_vec_3d_alternative) or np.allclose(pol_vec_3d, -pol_vec_3d_alternative):
        #    return pol_vec_3d
        #else:
        #    print("WARNING: The polarization vector of the laser '" + self.name + "' seems to be not constant over space! It was set to 'None'.")
        #    return None
        pol_vec = np.array([0, 0, 0], dtype=np.complex128)
        for beam in self.beams:
            pol_vec = pol_vec + np.sqrt(beam.P.to(u.mW).value) * beam.pol_vec_3d
        return pol_vec / np.linalg.norm(pol_vec)
        



    def _check_input(
            self, 
            method: str,
    ) -> None:
        """Checks the input of the method ``method``."""
        if method == 'init':
            # Check beams
            if isinstance(self.beams, Beam):
                self.beams = np.array([self.beams])
            elif isinstance(self.beams, (Sequence, np.ndarray)):
                if not all([isinstance(beam, Beam) for beam in self.beams]):
                    raise TypeError('beams must be an instance of Beam or a sequence of Beam instances.')
            else:
                raise TypeError('beams must be an instance of Beam or a sequence of Beam instances.')

            # Check pol_vec_3d
            if self.pol_vec_3d is not None:
                if not isinstance(self.pol_vec_3d, (Sequence, np.ndarray)):
                    raise TypeError('pol_vec_3d must be a sequence or numpy array.')
                if len(self.pol_vec_3d) == 3:
                    if all([isinstance(pol, (int, float, complex)) for pol in self.pol_vec_3d]):
                        self.pol_vec_3d = np.asarray(self.pol_vec_3d, dtype=np.complex128)
                        self.pol_vec_3d = self.pol_vec_3d / np.linalg.norm(self.pol_vec_3d)
                    else:
                        raise TypeError('pol_vec_3d must contain only integers, floats or complex numbers.')
                else:
                    raise ValueError('pol_vec_3d must have length 3.')
                