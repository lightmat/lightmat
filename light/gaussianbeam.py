from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from typing import Union
from collections.abc import Sequence



class GaussianBeam(object):
    """
    A class representing a Gaussian laser beam with various initialization parameters.
    In the local coordinate system of this GaussianBeam instance, the beam propagates along the local
    z-direction and the polarization Jones vector is in the local x-y-plane such that horizontal polarization
    is along the local x-direction and the vertical polarization along the local y-direction respectively. 
    The attribute _rotation_matrix transforms from the local coordinate system to the global standard Carteesian
    coordinate system.
    Note, that in principle the rotation matrix is ambiguous because the Jones vector in the 
           orthogonal subspace to beam_direction has an arbitrary 2d basis in general. We choose the 
           following convention for the 3d polarization vector at the 3 beam directions (1,0,0), (0,1,0) 
           and (0,0,1) respectively:
           Beam along x-axis: Linear horizontal polarization -> pol_vec = (0,1,0)
                              Linear vertical polarization -> pol_vec = (0,0,1)
           Beam along y-axis: Linear horizontal polarization -> pol_vec = (1,0,0)
                              Linear vertical polarization -> pol_vec = (0,0,-1)
           Beam along z-axis: Linear horizontal polarization -> pol_vec = (1,0,0)
                              Linear vertical polarization -> pol_vec = (0,1,0)
                              
    The beam is initialized with the following parameters:
    
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
    - Beam power (P)
    - Intensity (I0)
    - Electric field strength (E0)

    The following parameters do not necessarily need to be provided and have default values:
    - beam_direction: Direction of the beam in global coorinate system. Defaults to [0, 0, 1].
    - pol: Polarization of the beam as 2d Jones vector (perpendicular to beam_direction). 
           Defaults to 'linear horizontal'.
    - w0_zpos: Position of the beam waist, i.e. distance from origin along beam_direction. 
               Defaults to 0*astropy.units.meter.

    Example for a Gaussian beam along x-direction with right-hand circular polarization, wavelength 800 nm, 
    beam waist diameter 1 mm and peak power 5 mW:

    >>> import astropy.units as u
    >>> from gaussianbeam import GaussianBeam
    >>> beam = GaussianBeam(
                        beam_direction=[1, 0, 0], 
                        pol='circular right', 
                        lambda_=800*u.nm, 
                        w0=(1*u.mm, 300*u.um), 
                        P=5*u.mW
                    )

    -----------------------------------------------------------------------------------------------------------

    Attributes:
        beam_direction (np.ndarray): Direction of the beam in global coorinate system (3d vector)
        pol (np.ndarray): Polarization of the beam as 2D Jones vector (ux, uy) in the frame where 
                          beam_direction points along z-dir, e.g.: (1, 0) == 'linear horizontal', 
                          (0, 1) == 'linear vertical', 1/sqrt(2)*(1, i) == 'circular left', 
                          1/sqrt(2)*(1, -i) == 'circular right' 
        lambda_ (astropy.units.Quantity): Wavelength of the beam.
        nu (astropy.units.Quantity): Frequency of the beam.
        w (astropy.units.Quantity): Angular frequency of the beam.
        k (astropy.units.Quantity): Wavenumber of the beam.
        w0 (astropy.units.Quantity or Sequence[astropy.units.Quantity]): Beam waist diameter. If it's a 2d Sequence, than an 
                                                                         elliptical beam gets defined, where w0[0] is the beam
                                                                         waist in horizontal direction and w0[1] the beam waist 
                                                                         in vertical direction.
        z_R (astropy.units.Quantity): Rayleigh length.
        theta (astropy.units.Quantity): Beam divergence angle.
        P (astropy.units.Quantity): Beam power.
        I0 (astropy.units.Quantity): Peak intensity.
        E0 (astropy.units.Quantity): Peak electric field strength.
        w0_zpos (astropy.units.Quantity): Position of the beam waist, i.e. distance from origin along 
                                          beam_direction.
        k_vec (astropy.units.Quantity): Wave vector of the beam in global coordinate system.
        rotation_matrix (np.ndarray): Rotation matrix to transform from local to global coordinate system.
        pol_vec (np.ndarray): 3D polarization vector in global coordinate system (of course this is perpendicular
                              to the beam_direction vector in the global coordinate system).

    -----------------------------------------------------------------------------------------------------------

    Methods:

    """

    def __init__(
            self, 
            beam_direction: Sequence[float] = np.array([0, 0, 1]),
            pol: Union[str, Sequence[complex]] = 'linear horizontal', 
            lambda_: Union[u.Quantity, None] = None,
            nu: Union[u.Quantity, None] = None,
            w: Union[u.Quantity, None] = None,
            k: Union[u.Quantity, None] = None,
            w0: Union[u.Quantity, Sequence[u.Quantity], None] = None,
            z_R: Union[u.Quantity, None] = None,
            theta: Union[u.Quantity, None] = None,
            P: Union[u.Quantity, None] = None,
            I0: Union[u.Quantity, None] = None,
            E0: Union[u.Quantity, None] = None,
            w0_zpos: u.Quantity = 0*u.m,
        ):
        """Initialize a GaussianBeam object.
        
            Parameters:
                beam_direction (Sequence[float]): Direction of the beam in global coorinate system. 
                                                  Must be a 3D vector. Defaults to [0, 0, 1].
                pol (str or Sequence[float]): Polarization of the beam. If str, must be one of the following: 
                                              'linear horizontal', 'linear vertical', 'circular left', 
                                              'circular right'. If Sequence[float], must be a 2D Jones vector 
                                              (ux, uy) in the frame where beam_direction points along z-dir, e.g.:
                                              (1, 0) == 'linear horizontal', 
                                              (0, 1) == 'linear vertical', 
                                              1/sqrt(2)*(1, i) == 'circular left', 
                                              1/sqrt(2)*(1, -i) == 'circular right' 
                                              Note, that circular right/left is from the point of view of the 
                                              receiver, not the source. Defaults to 'linear horizontal'.
                lambda_ (astropy.units.Quantity or None): Wavelength of the beam. Defaults to None.
                nu (astropy.units.Quantity or None): Frequency of the beam. Defaults to None.
                w (astropy.units.Quantity or None): Angular frequency of the beam. Defaults to None.
                k (astropy.units.Quantity or None): Wavenumber of the beam. Defaults to None.
                w0 (astropy.units.Quantity or Sequence[astropy.units.Quantity] or None): Beam waist diameter. If a 2d Sequence
                                                                                         is provided, than an elliptical beam
                                                                                         gets defined, where w0[0] is the beam
                                                                                         waist in horizontal direction and w0[1]
                                                                                         the beam waist in vertical direction.
                                                                                         Defaults to None.
                z_R (astropy.units.Quantity or None): Rayleigh length. Defaults to None.
                theta (astropy.units.Quantity or None): Beam divergence angle. Defaults to None.
                P (astropy.units.Quantity or None): Peak power. Defaults to None.
                I0 (astropy.units.Quantity or None): Peak intensity. Defaults to None.
                E0 (astropy.units.Quantity or None): Peak electric field strength. Defaults to None.
                w0_zpos (astropy.units.Quantity): Position of the beam waist, i.e. distance from origin along 
                                                  beam_direction. Defaults to 0*astropy.units.meter.
                
            Returns:
                GaussianBeam: A GaussianBeam object.
        """
        self._beam_direction = beam_direction
        self._pol = pol
        self._lambda_ = lambda_
        self._nu = nu
        self._w = w
        self._k = k
        self._w0 = w0
        self._z_R = z_R
        self._theta = theta
        self._P = P
        self._I0 = I0
        self._E0 = E0
        self._w0_zpos = w0_zpos

        self._check_input()
        self._initialize_beam()

        self._k_vec = self._k * self._beam_direction
        self._rotation_matrix = self._calculate_rotation_matrix() # Rotation matrix to transform from local
                                                                  # to global coordinate system
        self._pol_vec = self._rotation_matrix @ np.array([self._pol[0], self._pol[1], 0]) # 3D polarization vector
        assert self._beam_direction @ self._pol_vec < 1e-12, "pol vector must be perpendicular to beam_direction" 


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

            
    # Beam power 
    @property
    def P(self):
        return self._P     

    @property
    def I0(self):
        return self._I0
        
    @property
    def E0(self):
        return self._E0 
    
    @property
    def pol(self):
        return self._pol
    
    @property
    def w0_zpos(self):
        return self._w0_zpos
    
    # Polarization and beam direction
    @property
    def beam_direction(self):
        return self._beam_direction
    
    @property
    def k_vec(self):
        return self._k_vec
    
    @property
    def rotation_matrix(self):
        return self._rotation_matrix
    
    @property
    def pol_vec(self):
        return self._pol_vec

    
    # Methods

    def E_field_amplitude(
            self,
            x: u.Quantity,
            y: u.Quantity,
            z: u.Quantity,
    ) -> u.Quantity:
        """Calculate the electric field strength of the Gaussian beam at the given position. The complex 
           E-field vector can then be obtained as: Evec(x,y,z) = E(x,y,z) * pol_vec
        
            Parameters:
                x (astropy.units.Quantity): x-coordinate of the position.
                y (astropy.units.Quantity): y-coordinate of the position.
                z (astropy.units.Quantity): z-coordinate of the position.
                
            Returns:
                astropy.units.Quantity]: Electric field strength of the Gaussian beam at the given position.
        """

        # Check input
        for c in [x, y, z]:
            if not isinstance(c, u.Quantity):
                raise TypeError("x, y and z must be an instance of astropy.units.Quantity.")
            else:
                if not c.unit.is_equivalent(u.m):
                    raise ValueError("x, y and z must be of unit equivalent to meter.")
                else:
                    x = x.to(u.mm)
                    y = y.to(u.mm)
                    z = z.to(u.mm)

        # Transform position from global to local coordinate system
        pos = np.array([x.value, y.value, z.value]) 
        pos = np.transpose(self._rotation_matrix) @ pos
        x = pos[0] * u.mm
        y = pos[1] * u.mm
        z = pos[2] * u.mm

        if not isinstance(self.w0, Sequence):
            # Calculate electric field strength (see https://en.wikipedia.org/wiki/Gaussian_beam)
            r = np.sqrt(x**2 + y**2)
            wz = self._w0 * np.sqrt(1 + (z - self._w0_zpos)**2 / self._z_R**2)
            one_over_Rz = z / (z**2 + self._z_R**2)
            Psiz = np.arctan((z - self._w0_zpos) / self._z_R).value # arctan gives u.rad unit, we just need the value
            E = self._E0 * self._w0/wz * np.exp(-r**2 / wz**2) * np.exp(-1j * (self._k*(z-self._w0_zpos)) \
                                                                        + self._k*r**2*one_over_Rz/2 - Psiz)
        else:
            # TODO: Implement elliptical beam shape
            pass
    
        return E
    


    def intensity(
            self,
            x: u.Quantity,
            y: u.Quantity,
            z: u.Quantity,
    ) -> u.Quantity:
        """Calculate the intensity of the Gaussian beam at the given position.
        
            Parameters:
                x (astropy.units.Quantity): x-coordinate of the position.
                y (astropy.units.Quantity): y-coordinate of the position.
                z (astropy.units.Quantity): z-coordinate of the position.
                
            Returns:
                astropy.units.Quantity: Intensity of the Gaussian beam at the given position.
        """

        # Check input
        for c in [x, y, z]:
            if not isinstance(c, u.Quantity):
                raise TypeError("x, y and z must be an instance of astropy.units.Quantity.")
            else:
                if not c.unit.is_equivalent(u.m):
                    raise ValueError("x, y and z must be of unit equivalent to meter.")
                else:
                    x = x.to(u.mm)
                    y = y.to(u.mm)
                    z = z.to(u.mm)

        # Transform position from global to local coordinate system
        pos = np.array([x.value, y.value, z.value]) 
        pos = np.transpose(self._rotation_matrix) @ pos
        x = pos[0] * u.mm
        y = pos[1] * u.mm
        z = pos[2] * u.mm

        if not isinstance(self.w0, Sequence):
            # Calculate intensity (see https://en.wikipedia.org/wiki/Gaussian_beam)
            r = np.sqrt(x**2 + y**2)
            wz = self._w0 * np.sqrt(1 + (z - self._w0_zpos)**2 / self._z_R**2)
            I = self._I0 * (self._w0/wz)**2 * np.exp(-2*r**2 / wz**2)
        else:
            # TODO: Implement elliptical beam shape
            pass

        return I
    



    def _check_input(self,):
        """Check that the input values are valid."""

        # Check beam_direction
        if not isinstance(self._beam_direction, Sequence):
            raise TypeError("beam_direction must be an Sequence.")
        if len(self._beam_direction) != 3:
            raise ValueError("beam_direction must be a 3D vector in the global coordinate system.")
        
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
        
    
    def _initialize_beam(self,):
        """Initialize the beam."""

        # Beam direction and polarization
        self._beam_direction = np.array(self._beam_direction) / np.linalg.norm(self._beam_direction)
        if isinstance(self._pol, str):
            if self._pol == 'linear horizontal':
                self._pol = np.array([1, 0])
            elif self._pol == 'linear vertical':
                self._pol = np.array([0, 1])
            elif self._pol == 'circular left':
                self._pol = 1/np.sqrt(2)*np.array([1, 1j])
            elif self._pol == 'circular right':
                self._pol = 1/np.sqrt(2)*np.array([1, -1j])
        elif isinstance(self._pol, Sequence):
            self._pol = np.array(self._pol) / np.linalg.norm(self._pol)

        # Electromagnetic properties
        if self._lambda_ != None:
            self._lambda_ = self._lambda_.to(u.nm)
            self._nu = (c / self._lambda_).to(u.THz)
            self._w = (2*np.pi * self._nu).to(u.THz)
            self._k = (2*np.pi / self._lambda_).to(1/u.m)
        elif self._nu != None:
            self._lambda_ = (c / self._nu).to(u.nm)
            self._nu = self._nu.to(u.THz)
            self._w = (2*np.pi * self._nu).to(u.THz)
            self._k = (2*np.pi / self._lambda_).to(1/u.m)
        elif self._w != None:
            self._nu = (self._w / (2*np.pi)).to(u.THz)
            self._lambda_ = (c / self._nu).to(u.nm)
            self._w = self._w.to(u.THz)
            self._k = (2*np.pi / self._lambda_).to(1/u.m)
        elif self._k != None:
            self._lambda_ = (2*np.pi / self._k).to(u.nm)
            self._nu = (c / self._lambda_).to(u.THz)
            self._w = (2*np.pi * self._nu).to(u.THz)
            self._k = self._k.to(1/u.m)

        # Beam geometry
        if self._w0 != None:
            self._w0 = self._w0.to(u.mm)
            self._z_R = (np.pi * self._w0**2 / self._lambda_).to(u.m)
            self._theta = (self._lambda_.to(u.m) / (np.pi * self._w0.to(u.m)))*u.rad
        elif self._z_R != None:
            self._z_R = self._z_R.to(u.m)
            self._w0 = np.sqrt((self._z_R * self._lambda_ / np.pi)).to(u.mm)
            self._theta = (self._lambda_.to(u.m) / (np.pi * self._w0.to(u.m)))*u.rad
        elif self._theta != None:
            self._theta = self._theta.to(u.rad)
            self._w0 = (self._lambda_.to(u.m) / (np.pi * self._theta.to(u.rad).value)).to(u.mm)
            self._z_R = (np.pi * self._w0**2 / self._lambda_).to(u.m)
        
        # Beam power attributes
        if self._P != None:
            self._P = self._P.to(u.W)
            self._I0 = (2*self._P / (np.pi * self._w0**2)).to(u.mW/u.cm**2)
            self._E0 = (np.sqrt(2*self._I0 / (eps0*c))).to(u.V/u.m)
        elif self._I0 != None:
            self._I0 = self._I0.to(u.mW/u.cm**2)
            self._P = (self._I0 * np.pi * self._w0**2 / 2).to(u.W)
            self._E0 = (np.sqrt(2*self._I0 / (eps0*c))).to(u.V/u.m)
        elif self._E0 != None:
            self._E0 = self._E0.to(u.V/u.m)
            self._I0 = (eps0*c*self._E0**2 / 2).to(u.mW/u.cm**2)
            self._P = (self._I0 * np.pi * self._w0**2 / 2).to(u.W)

        


    def _calculate_rotation_matrix(self,):
        """Calculate the rotation matrix to transform from the local coordinate system to the global 
           coordinate system. 
           
           In the local coordinate system of this GaussianBeam instance, the 3d polarization vector 
           and the beam_direction vector are given by:
           pol_vec = (pol[0], pol[1], 0)
           beam_direction = (0, 0, 1) 

           In the global coordinate system, the 3d polarization vector and the beam_direction vector 
           are given by:
           pol_vec = R * (pol[0], pol[1], 0)   # R is the rotation matrix
           beam_direction = (beam_direction[0], beam_direction[1], beam_direction[2])

           Note, that in principle the rotation matrix is ambiguous because the Jones vector in the 
           orthogonal subspace to beam_direction has an arbitrary 2d basis in general. We choose the 
           following convention for the 3d polarization vector at the 3 beam directions (1,0,0), (0,1,0) 
           and (0,0,1) respectively:
           Beam along x-axis: Linear horizontal polarization -> pol_vec = (0,1,0)
                              Linear vertical polarization -> pol_vec = (0,0,1)
           Beam along y-axis: Linear horizontal polarization -> pol_vec = (1,0,0)
                              Linear vertical polarization -> pol_vec = (0,0,-1)
           Beam along z-axis: Linear horizontal polarization -> pol_vec = (1,0,0)
                              Linear vertical polarization -> pol_vec = (0,1,0)
           
           Note, that regardless of the coordinate system, the polarization vector is always
           perpendicular to the beam_direction vector!           
        """

        beam_direction = np.array(self.beam_direction) / np.linalg.norm(self.beam_direction) # Normalize beam direction
        z_axis = np.array([0, 0, 1])

        # Choosing local x-axis based on the beam direction
        if beam_direction[0] != 0:
            local_x_axis = np.array([0, 1, 0])
        else:
            local_x_axis = np.array([1, 0, 0])

        # Computing local y-axis
        local_y_axis = np.cross(beam_direction, local_x_axis)
        local_y_axis /= np.linalg.norm(local_y_axis)  # Normalize

        # Adjust local x-axis to ensure orthogonality
        local_x_axis = np.cross(local_y_axis, beam_direction)

        # Constructing rotation matrix
        R = np.column_stack((local_x_axis, local_y_axis, beam_direction))

        return R
