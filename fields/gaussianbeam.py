from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Iterable



class GaussianBeam(object):
    """
    A class representing a Gaussian laser beam with various initialization parameters.
    
    One of the following must be provided for electromagnetic properties:
    - Wavelength (lambda_)
    - Frequency (nu)
    - Angular frequency (w)
    - Wavenumber (k)

    One of the following must be provided for beam geometry:
    - Beam waist diameter (w0)
    - Rayleigh length (z_R)
    - Beam divergence angle (theta)
    
    One of the following must be provided for beam power attributes:
    - Power (P0)
    - Intensity (I0)
    - Electric field strength (E0)
    """

    def __init__(
            self, 
            lambda_: Union[u.Quantity, None] = None,
            nu: Union[u.Quantity, None] = None,
            w: Union[u.Quantity, None] = None,
            k: Union[u.Quantity, None] = None,
            w0: Union[u.Quantity, None] = None,
            z_R: Union[u.Quantity, None] = None,
            theta: Union[u.Quantity, None] = None,
            P0: Union[u.Quantity, None] = None,
            I0: Union[u.Quantity, None] = None,
            E0: Union[u.Quantity, None] = None,
            w0_zpos: u.Quantity = 0*u.m,
            pol: Union[str, Iterable[float]] = 'linear',
            **kwargs
        ):
        """Initialize a GaussianBeam object.
        
            Parameters:
                lambda_ (astropy.units.Quantity): Wavelength of the beam.
                nu (astropy.units.Quantity): Frequency of the beam.
                w (astropy.units.Quantity): Angular frequency of the beam.
                k (astropy.units.Quantity): Wavenumber of the beam.
                w0 (astropy.units.Quantity): Beam waist diameter.
                z_R (astropy.units.Quantity): Rayleigh length.
                theta (astropy.units.Quantity): Beam divergence angle.
                P0 (astropy.units.Quantity): Peak power.
                I0 (astropy.units.Quantity): Peak intensity.
                E0 (astropy.units.Quantity): Peak electric field strength.
                w0_zpos (astropy.units.Quantity): Position of the beam waist.
                pol (str or Iterable[float]): Polarization of the beam.
                
            Returns:
                GaussianBeam: A GaussianBeam object.
        """

        self._lambda_ = lambda_
        self._nu = nu
        self._w = w
        self._k = k
        self._w0 = w0
        self._z_R = z_R
        self._theta = theta
        self._P0 = P0
        self._I0 = I0
        self._E0 = E0
        self._w0_zpos = w0_zpos
        self._pol = pol

        self._check_input()

        # Electromagnetic properties
        if self._lambda_ != None and self._nu == None and self._w == None and self._k == None:
            self._lambda_ = self._lambda_.to(u.nm)
            self._nu = (c / self._lambda_).to(u.THz)
            self._w = (2*np.pi * self._nu).to(u.THz)
            self._k = (2*np.pi / self._lambda_).to(1/u.m)
        elif self._nu != None and self._lambda_ == None and self._w == None and self._k == None:
            self._lambda_ = (c / self._nu).to(u.nm)
            self._nu = self._nu.to(u.THz)
            self._w = (2*np.pi * self._nu).to(u.THz)
            self._k = (2*np.pi / self._lambda_).to(1/u.m)
        elif self._w != None and self._lambda_ == None and self._nu == None and self._k == None:
            self._nu = (self._w / (2*np.pi)).to(u.THz)
            self._lambda_ = (c / self._nu).to(u.nm)
            self._w = self._w.to(u.THz)
            self._k = (2*np.pi / self._lambda_).to(1/u.m)
        elif self._k != None and self._lambda_ == None and self._nu == None and self._w == None:
            self._lambda_ = (2*np.pi / self._k).to(u.nm)
            self._nu = (c / self._lambda_).to(u.THz)
            self._w = (2*np.pi * self._nu).to(u.THz)
            self._k = self._k.to(1/u.m)
        else:
            raise ValueError("Exactly one of the following must be provided: lambda_, nu, w or k.")

        # Beam geometry
        if self._w0 != None and self._z_R == None and self._theta == None:
            self._w0 = self._w0.to(u.mm)
            self._z_R = (np.pi * self._w0**2 / self._lambda_).to(u.m)
            self._theta = (self._lambda_.to(u.m) / (np.pi * self._w0.to(u.m)))*u.rad
        elif self._z_R != None and self._w0 == None and self._theta == None:
            self._z_R = self._z_R.to(u.m)
            self._w0 = np.sqrt((self._z_R * self._lambda_ / np.pi)).to(u.mm)
            self._theta = (self._lambda_.to(u.m) / (np.pi * self._w0.to(u.m)))*u.rad
        elif self._theta != None and self._w0 == None and self._z_R == None:
            self._theta = self._theta.to(u.rad)
            self._w0 = (self._lambda_.to(u.m) / (np.pi * self._theta.to(u.rad).value)).to(u.mm)
            self._z_R = (np.pi * self._w0**2 / self._lambda_).to(u.m)
        else:
            raise ValueError("Exactly one of the following must be provided: w0, z_R or theta.")
        

        # Beam power attributes
        if self._P0 != None and self._I0 == None and self._E0 == None:
            self._P0 = self._P0.to(u.W)
            self._I0 = (2*self._P0 / (np.pi * self._w0**2)).to(u.mW/u.cm**2)
            self._E0 = (np.sqrt(2*self._I0 / (eps0*c))).to(u.V/u.m)
        elif self._I0 != None and self._P0 == None and self._E0 == None:
            self._I0 = self._I0.to(u.mW/u.cm**2)
            self._P0 = (self._I0 * np.pi * self._w0**2 / 2).to(u.W)
            self._E0 = (np.sqrt(2*self._I0 / (eps0*c))).to(u.V/u.m)
        elif self._E0 != None and self._P0 == None and self._I0 == None:
            self._E0 = self._E0.to(u.V/u.m)
            self._I0 = (eps0*c*self._E0**2 / 2).to(u.mW/u.cm**2)
            self._P0 = (self._I0 * np.pi * self._w0**2 / 2).to(u.W)
        else:
            raise ValueError("Exactly one of the following must be provided: P0, I0 or E0.")      
        
        

    def _check_input(self,):
        """Check that the input values are valid."""

        for attr_name in ['_lambda_', '_nu', '_w', '_k', '_w0', '_z_R', '_theta', '_P0', '_I0', '_E0']:
            attr_value = getattr(self, attr_name)
            if attr_value is not None and not isinstance(attr_value, u.Quantity):
                raise TypeError(f"The attribute '{attr_name}' must be an instance of astropy.units.Quantity.")
        
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
        if self._P0 != None and not self._P0.unit.is_equivalent(u.W):
            raise ValueError("Power must be of unit equivalent to Watt.")
        if self._I0 != None and not self._I0.unit.is_equivalent(u.W/u.m**2):
            raise ValueError("Intensity must be of unit equivalent to Watt per square meter.")
        if self._E0 != None and not self._E0.unit.is_equivalent(u.V/u.m):
            raise ValueError("Electric field strength must be of unit equivalent to Volt per meter.")


    # Electromagnetic properties
    @property
    def lambda_(self):
        return self._lambda_        
    
    @property
    def nu(self):
        return self._nu
    
    @property
    def w(self):
        return self._w

    @property
    def k(self):
        return self._k
        

    # Beam geometry
    @property
    def w0(self):
        return self._w0

    @property
    def z_R(self):
        return self._z_R
        
    @property
    def theta(self):
        return self._theta

            
    # Beam power attributes
    @property
    def P(self):
        return self._P0     

    @property
    def I(self):
        return self._I0
        
    @property
    def E(self):
        return self._E0 
    
    @property
    def pol(self):
        return self._pol
    
    @property
    def w0_zpos(self):
        return self._w0_zpos
    
    def print_beam_data(self,):
        # TODO: Make a matplotlib plot with LaTeX beam data on the left and plot of beam on the right.
        print(r"Wavelength: $\lambda$ = ", self._lambda_)
        print(r"Frequency: $\nu$ = ", self._nu)
        print(r"Angular frequency: $\omega$ = ", self._w)
        print(r"Wave number: $k$ = ", self._k)
        print(r"Beam waist diameter: $w_0$ = ", self._w0)
        print(r"Rayleigh length: $z_R$ = ", self._z_R)
        print(r"Beam divergence angle: $\theta$ = ", self._theta)
        print(r"Peak power: $P_0$ = ", self._P0)
        print(r"Peak intensity: $I_0$ = ", self._I0)
        print(r"Peak electric field strength: $E_0$ = ", self._E0)

    
    def w(self, 
          z: Union[u.Quantity, Iterable[u.Quantity]],
        ) -> Union[u.Quantity, float, Iterable[float]]:
        """Gaussian beam diameter at position z."""
        return self._w0 * np.sqrt(1 + (z / self._z_R)**2)
    
    def I(self, 
          r: Union[u.Quantity, Iterable[u.Quantity]],
          z: Union[u.Quantity, Iterable[u.Quantity]],
        ) -> Union[u.Quantity, Iterable[u.Quantity]]:
        """Gaussian beam intensity I(r,z) for beam propagation in z-direction and linear
           polarization in x-direction."""
        return self._I0*(self._w0 / self.w(z))**2 * np.exp(-2*(r / self.w(z))**2)
