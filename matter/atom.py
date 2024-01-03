from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence



class Atom(object):
    
    def __init__(
            self,
            name: str,
            state: Union[str, Sequence[str]],
            transition_data: Union[None, pd.DataFrame] = None,
            backend: Union[None, str] = None,
    ) -> None:
        
        self._name = name
        self._state = state
        self._transition_data = transition_data
        self._backend = backend

        self._check_input()



    def _check_input(self,):
        """Check that the input values are valid."""

        # Check name
        if not isinstance(self._name, str):
            raise TypeError("name must be a str.")
        if self._name not in ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',] and self._transition_data == None:
            raise ValueError("The name must be one of ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',] or the transition_data\
                              must be provided as pd.dataframe.")
        
        # Check state
        if not isinstance(self._state, Union[str, Sequence[str]]):
            raise TypeError("state must be a str or Sequence of str.")
        
        
        # Check pol
        if isinstance(self._pol, str):
            if self._pol not in ['linear horizontal', 'linear vertical', 'circular left', 'circular right']:
                raise ValueError("If pol is a str, it must be one of the following: 'linear horizontal', \
                                 'linear vertical', 'circular left', 'circular right'.")
        elif isinstance(self._pol, Sequence):
            if len(self._pol) != 2:
                raise ValueError("If pol is a Sequence, it must be a 2D Jones vector (ux, uy) in the frame where \
                                  beam_direction points along z-dir, e.g.: (1, 0) == 'linear horizontal', \
                                  (0, 1) == 'linear vertical', 1/sqrt(2)*(1, i) == 'circular left', \
                                  1/sqrt(2)*(1, -i) == 'circular right'.")
        else:
            raise TypeError("pol must be a str or an Sequence.")
        
        # Check beam properties
        for attr_name in ['_lambda_', '_nu', '_w', '_k', '_w0', '_z_R', '_theta', '_P', '_I0', '_E0']:
            attr_value = getattr(self, attr_name)
            if attr_value is not None and not isinstance(attr_value, u.Quantity):
                raise TypeError(f"The attribute '{attr_name}' must be an instance of astropy.units.Quantity.")      
        em_properties = sum([getattr(self, attr_name) != None for attr_name in ['_lambda_', '_nu', '_w', '_k']])
        if em_properties != 1:
            raise ValueError("Exactly one of the following must be provided: lambda_, nu, w or k.")
        geometry_properties = sum([getattr(self, attr_name) != None for attr_name in ['_w0', '_z_R', '_theta']])
        if geometry_properties != 1:
            raise ValueError("Exactly one of the following must be provided: w0, z_R or theta.")
        power_properties = sum([getattr(self, attr_name) != None for attr_name in ['_P', '_I0', '_E0']])
        if power_properties != 1:
            raise ValueError("Exactly one of the following must be provided: P, I0 or E0.")
        if self._lambda_ != None and not self._lambda_.unit.is_equivalent(u.m):
            raise ValueError("Wavelength must be of unit equivalent to meter.")
        if self._nu != None and not self._nu.unit.is_equivalent(1/u.s):
            raise ValueError("Frequency must be of unit equivalent to 1/s.")
        if self._w != None and not self._w.unit.is_equivalent(1/u.s):
            raise ValueError("Angular frequency must be of unit equivalent to 1/s.")
        if self._k != None and not self._k.unit.is_equivalent(1/u.m):
            raise ValueError("Wavenumber must be of unit equivalent to 1/m.")
        if self._w0 != None and not self._w0.unit.is_equivalent(u.m):
            raise ValueError("Beam waist diameter must be of unit equivalent to meter.")
        if self._z_R != None and not self._z_R.unit.is_equivalent(u.m):
            raise ValueError("Rayleigh length must be of unit equivalent to meter.")
        if self._theta != None and not self._theta.unit.is_equivalent(u.rad):
            raise ValueError("Beam divergence angle must be of unit equivalent to radian.")
        if self._P != None and not self._P.unit.is_equivalent(u.W):
            raise ValueError("Power must be of unit equivalent to Watt.")
        if self._I0 != None and not self._I0.unit.is_equivalent(u.W/u.m**2):
            raise ValueError("Intensity must be of unit equivalent to Watt per square meter.")
        if self._E0 != None and not self._E0.unit.is_equivalent(u.V/u.m):
            raise ValueError("Electric field strength must be of unit equivalent to Volt per meter.")
        
        # Check w0_zpos
        if not isinstance(self._w0_zpos, u.Quantity):
            raise TypeError("w0_zpos must be an instance of astropy.units.Quantity.")