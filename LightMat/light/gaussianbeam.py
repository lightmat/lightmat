from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Union
from collections.abc import Sequence

from .laser import Laser



class GaussianBeam(Laser):
    """
    A class representing a Gaussian laser beam.
    In the local coordinate system of this GaussianBeam instance, the beam propagates along the local
    z-direction and the polarization Jones vector is in the local x-y-plane with the convention that horizontal 
    polarization is along the local x-direction and the vertical polarization along the local y-direction respectively. 
    The attribute ``_rotation_matrix`` transforms from the local coordinate system to the global standard Carteesian
    coordinate system given by the basis (1,0,0), (0,1,0), (0,0,1).

    Note, that in principle the rotation matrix is ambiguous because the Jones vector in the orthogonal subspace to 
    ``beam_direction_vec`` has an arbitrary 2d basis in general. As mentioned above, we choose the following convention 
    in the local coordinate system of the GaussianBeam instance:
    Beam along local z-axis: Linear horizontal polarization -> ``pol_Jones_vec`` = (1,0) and ``pol_3d_vec`` = (1,0,0)
                             Linear vertical polarization -> ``pol_Jones_vec`` = (0,1) and ``pol_3d_vec`` = (0,1,0)

    Of course, in any case ``pol_3d_vec`` is perpendicular to ``beam_direction_vec``, both in the local coordinate
    system of the GaussianBeam instance and the global standard Carteesian coordiante system.

    ----------------------------------------------------------------------------------------------------------------------

    Parameters:
        beam_direction_vec (array_like): 3d vector specifying the beam propagation in the global standard Carteesian 
                                         coordinate system.
        pol_Jones_vec (array_like): 2d vector specifying the polarization of the beam in the local coordinate system
                                    where the beam propagates along the local z-direction. The convention is that the
                                    horizontal polarization is along the local x-direction and the vertical polarization
                                    along the local y-direction.
        lambda_ (astropy.Quantity, float): Wavelength of the beam in [nm]. 
        w0 (astropy.Quantity, float): Beam waist diameter in [um]. Either a scalar for circular beams or a sequence of two floats for
                                      elliptical beams having different beam waist diameters in local x- and y-direction.
        P (astropy.Quantity, float): Power of the beam in [W].
        z0 (astropy.Quantity, float): Position of the beam waist in [um] along the beam propagation direction. 
                                      Defaults to 0um.       

    ----------------------------------------------------------------------------------------------------------------------

    Attributes:
        beam_direction_vec (np.ndarray): 3d vector specifying the beam propagation in the global standard Carteesian 
                                         coordinate system.
        pol_Jones_vec (np.ndarray): 2d vector specifying the polarization of the beam in the 2d plane perpedicular to
                                    ``beam_direction_vec``. The convention is that the horizontal polarization is along the 
                                    local x-direction and the vertical polarization along the local y-direction.
        pol_3d_vec (np.ndarray): 3d vector specifying the polarization of the beam in the global standard Carteesian
                                 coordinate system. It is perpendicular to ``_beam_direction_vec``.
        lambda_ (astropy.Quantity): Wavelength of the beam in [nm]. 
        w0 (astropy.Quantity): Beam waist diameter in [um]. Either a scalar for circular beams or a sequence of two floats for
                               elliptical beams having different beam waist diameters in local x- and y-direction.
        P (astropy.Quantity): Power of the beam in [W].
        z0 (astropy.Quantity): Position of the beam waist in [um] along the beam propagation direction. 
                                Defaults to 0um.
        k (astropy.Quantity): Wave number of the beam in [1/um].
        k_vec (astrophy.Quantity): 3d wave vector of the beam in [1/um] in the global standard Carteesian coordinate system. 
                                   It is parallel to ``_beam_direction_vec`` and perpendicular to ``pol_3d_vec``.
        omega (astropy.Quantity): Angular frequency of the beam in [THz].
        nu (astropy.Quantity): Frequency of the beam in [THz].
        z_R (astropy.Quantity): Rayleigh range of the beam in [um].
        theta (astropy.Quantity): Divergence angle of the beam in [rad].
        I0 (astropy.Quantity): Peak intensity of the beam in [mW/cm^2].
        E0 (astropy.Quantity): Peak electric field strength of the beam in [V/m].
        _rotation_matrix (array_like): 3x3 matrix that transforms from the local coordinate system of the GaussianBeam 
                                       instance where propagation is along local z-direction to the global standard Carteesian 
                                       coordinate system. 

    ----------------------------------------------------------------------------------------------------------------------

    Methods:
        w(z_local): Calculate the beam diameter at the position(s) ``z_local`` in the local coordinate system of the GaussianBeam instance.
        R(z_local): Calculate the radius of curvature of the beam at the position(s) ``z_local`` in the local coordinate system.
        Psi(z_local): Calculate the Gouy phase of the beam at the position(s) ``z_local`` in the local coordinate system.
        E(x,y,z): Returns the electric field strength of the beam at the position (x,y,z) in [V/m]. Here, x, y, z are 
                  the global standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().
        E_vec(x,y,z): Returns the electric field vector of the beam at the position (x,y,z) in [V/m] in the standard
                      Carteesian coordinate system. This is just the electric field strength multiplied by ``pol_3d_vec``. 
                      Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array
                      obtained from np.meshgrid().
        I(x,y,z): Returns the intensity of the beam at the position (x,y,z) in [mW/cm^2]. Here, x, y, z are the global
                  standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().
    """
    def __init__(
            self,
            beam_direction_vec: Sequence[float],
            pol_Jones_vec: Sequence[float],
            lambda_: Union[u.Quantity, float],
            w0: Union[u.Quantity, float, Sequence[float], np.ndarray],
            P: Union[u.Quantity, float],
            z0: Union[u.Quantity, float] = 0 * u.um,
            name: str = "GaussianBeam",
            color: str = None,
    ) -> None:
        """Initializes a GaussianBeam instance.
        
           Args:
                beam_direction_vec: 3d vector specifying the beam propagation in the global standard Carteesian 
                                    coordinate system.
                pol_Jones_vec: 2d vector specifying the polarization of the beam in the local coordinate system
                               where the beam propagates along the local z-direction. The convention is that the
                               horizontal polarization is along the local x-direction and the vertical polarization
                               along the local y-direction.
                lambda_: Wavelength of the beam in [nm]. 
                w0: Beam waist diameter in [um]. Either a scalar for circular beams or a sequence of two floats for
                    elliptical beams having different beam waist diameters in local x- and y-direction.
                P: Power of the beam in [W].
                z0: Position of the beam waist in [um] along the beam propagation direction. Defaults to 0um.

           Returns:
                None           
        """
        # Define attributes and check input
        self.beam_direction_vec = np.array(beam_direction_vec)
        self.pol_Jones_vec = np.array(pol_Jones_vec)
        self.lambda_ = lambda_
        self.w0 = w0 # either a scalar for circular beams or a vector of two floats for elliptical beams
        self.P = P
        self.z0 = z0
        self.name = name
        self.color = color
        self._check_input('init')

        super().__init__(self.name, self.beam_direction_vec, self.lambda_, self.P, self.color)

        # Calculate derived attributes
        self.k = (2*np.pi / (self.lambda_)).to(1/u.um)
        self.k_vec = self.k * self.beam_direction_vec
        self.omega = (2*np.pi * c / self.lambda_).to(u.THz)
        self.nu = (self.omega / (2*np.pi)).to(u.THz)
        self.z_R = (np.pi * self.w0**2 / self.lambda_).to(u.um) # either a scalar for circular beams or a vector of two floats for elliptical beams
        self.theta = (self.lambda_ / (np.pi * self.w0)).value # either a scalar for circular beams or a vector of two floats for elliptical beams
        if np.isscalar(self.w0.value): # Circular beam
            self.I0 = (2 * self.P / (np.pi * self.w0**2)).to(u.mW/u.cm**2)
        else: # Elliptical beam
            self.I0 = (self.P / (np.pi * self.w0[0] * self.w0[1])).to(u.mW/u.cm**2)
        self.E0 = (np.sqrt(2 * self.I0 / (eps0 * c))).to(u.V/u.m)

        self._rotation_matrix = self._calculate_rotation_matrix() 
        self._rotation_matrix_sym = self._calculate_rotation_matrix_sym()
        self.pol_3d_vec = self._rotation_matrix @ np.array([self.pol_Jones_vec[0], self.pol_Jones_vec[1], 0])
        self.pol_3d_vec = self.pol_3d_vec / np.linalg.norm(self.pol_3d_vec)  # Normalize

        # Sanity checks for vector directions
        assert np.isclose(self.beam_direction_vec @ self.pol_3d_vec, 0), "The beam direction and the 3d polarization vector must be perpendicular."
        assert np.isclose(self.beam_direction_vec @ self.k_vec.value, self.k.value), "The beam direction and the wave vector must be parallel."
        


    def w(
            self,
            z_local: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Calculate the beam diameter at the position(s) ``z_local`` in the local coordinate system of the GaussianBeam instance, i.e.
           at distance ``z_local``-``z0`` from the beam waist diameter ``w0`` along the beam propagation direction in the global
           standard Carteesian coordinate system. For circular beams, the beam diameter is a scalar, for elliptical beams
           it comprises two scalars for the local x- and y-direction respectively.

           Args:
                z_local: z position(s) in local coordinate system. ``z_local``-``z0`` is the distance from beam waist diameter ``w0`` in [um].

           Returns:
                u.Quantity: Beam diameter w(``z_local``) at the local position ``z_local`` in [um], can be either float for circular beam or 
                            array for elliptical beam.
        """
        self.z_local = z_local
        self._check_input('w')

        # Calculate beam diameter at given local z positions
        if np.isscalar(self.w0.value): # Circular beam
            w = self.w0 * np.sqrt(1 + (self.z_local - self.z0)**2 / self.z_R**2)
        else: # Elliptical beam
            w = np.array([
                (self.w0[0] * np.sqrt(1 + (self.z_local - self.z0)**2 / self.z_R[0]**2)).to(self.w0.unit),
                (self.w0[1] * np.sqrt(1 + (self.z_local - self.z0)**2 / self.z_R[1]**2)).to(self.w0.unit),
            ]) * self.w0.unit

        return w.to(u.um)
    

    def R(
            self,
            z_local: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Calculate the radius of wavefront curvature at the position(s) ``z_local`` in the local coordinate system of the GaussianBeam 
           instance, i.e. at distance ``z_local``-``z0`` from the beam waist diameter ``w0`` along the beam propagation direction in 
           the global standard Carteesian coordinate system. For circular beams, R(``z_local``) is a scalar, for elliptical beams
           it comprises two scalars for the local x- and y-direction respectively.

           Args:
                z_local: z position(s) in local coordinate system. ``z_local``-``z0`` is the distance from beam waist diameter ``w0`` in [um].

           Returns:
                u.Quantity: R(``z_local``) at the local position ``z_local`` in [um], can be either float for circular beam or array for 
                            elliptical beam.
        """
        self.z_local = z_local
        self._check_input('R')

        # Calculate radius of wavefront curvature at given local z positions
        if np.isscalar(self.w0.value): # Circular beam
            R = self.z_local + self.z_R**2 / (self.z_local - self.z0)
        else: # Elliptical beam
            R = np.array([
                (self.z_local + self.z_R[0]**2 / (self.z_local - self.z0)).to(self.z_local.unit),
                (self.z_local + self.z_R[1]**2 / (self.z_local - self.z0)).to(self.z_local.unit),
            ]) * self.z_local.unit

        return R.to(u.um)
    

    def Psi(
            self,
            z_local: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Calculate the Gouy phase at the position(s) ``z_local`` in the local coordinate system of the GaussianBeam instance, i.e.
           at distance ``z_local``-``z0`` from the beam waist diameter ``w0`` along the beam propagation direction in the global
           standard Carteesian coordinate system. For circular beams, the Gouy phase is a scalar, for elliptical beams
           it comprises two scalars for the local x- and y-direction respectively.

           Args:
                z_local: z position(s) in local coordinate system. ``z_local``-``z0`` is the distance from beam waist diameter ``w0`` in [um].

           Returns:
                u.Quantity: Gouy phase Psi(``z_local``) at the local position ``z_local``, can be either float for circular beam or 
                            array for elliptical beam.
        """
        self.z_local = z_local
        self._check_input('Psi')

        # Calculate Gouy phase at given local z positions
        if np.isscalar(self.w0.value): # Circular beam
            Psi = np.arctan((self.z_local - self.z0) / self.z_R).to(u.rad).value # get rid of unit because [rad] is dimensionless 
                                                                                 # and astropy doesn't handle it well in exponential
        else: # Elliptical beam
            Psi = np.array([
                np.arctan((self.z_local - self.z0) / self.z_R[0]).to(u.rad).value, 
                np.arctan((self.z_local - self.z0) / self.z_R[1]).to(u.rad).value, 
            ])

        return Psi


    def E(
            self, 
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the complex electric field amplitude of the beam at the position (x,y,z) in [V/m]. 
           Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Complex electric field amplitude of the beam at the position (x,y,z) in [V/m], can be either float or array.
        """
        self.x = x
        self.y = y
        self.z = z
        self._check_input('E')

        # Store the original shape for reshaping later
        original_shape = self.x.shape  

        # Flatten x, y, z if they are meshgrids (if x has shape (n,n,n), then x_flat has shape (n**3,))
        x_flat = self.x.ravel()
        y_flat = self.y.ravel()
        z_flat = self.z.ravel()

        # Stack and transform position(s) from global to local coordinate system
        pos_flat = np.stack([x_flat, y_flat, z_flat], axis=-1).T   # Shape: (3, n**3)
        pos_local_flat = (self._rotation_matrix.T @ pos_flat)      # Note that the transpose of the rotation matrix is its inverse
        pos_local = pos_local_flat.reshape((3,) + original_shape)  # Reshape back to original shape

        # Unpack the local positions
        x_local, y_local, z_local = pos_local[0], pos_local[1], pos_local[2]

        # Calculate beam propagation parameters at given local z positions
        wz = self.w(z_local)
        Rz = self.R(z_local)
        Psiz = self.Psi(z_local)

        # Calculate electric field strength
        if np.isscalar(self.w0.value): # Circular beam, see https://en.wikipedia.org/wiki/Gaussian_beam, but of course it is also the special case
                                       # of the elliptical beam defined below
            E = self.E0 * self.w0 / wz * np.exp(-(x_local**2 + y_local**2) / wz**2 \
                                                - 1j * self.k * (x_local**2 + y_local**2) / (2*Rz) \
                                                + 1j * Psiz \
                                                - 1j * self.k * z_local)

        else: # Elliptical beam, see equations (62) in chap 16 with c00=E0*sqrt(wx0*wy0), (58) in chap. 16 with (5) in chap. 17 and (49) in chap. 16
            E = self.E0 * np.sqrt(self.w0[0] / wz[0]) * np.sqrt(self.w0[1] / wz[1]) \
                        * np.exp(-(x_local**2 / wz[0]**2 + y_local**2 / wz[1]**2) \
                                 - 1j * self.k * (x_local**2 / (2*Rz[0]) + y_local**2 / (2*Rz[1])) \
                                 + 1j * (Psiz[0]/2 + Psiz[1]/2) \
                                 - 1j * self.k * z_local)
            

        # Mask to handle calculation differently at the beam waist location z=z0 in order to avoid division by zero in the exponential
        # due to zero wavefront curvature at the beam waist
        # mask = ~np.isclose(z_local.value, self.z0.value)  
        # E = np.zeros_like(x_local.value, dtype=complex) * self.E0.unit  # Initialize complex electric field amplitude, must have same shape as x_local, y_local 
                                                                        # and z_local, which all have the same shape (as ensured in _check_input())

        #if np.isscalar(self.w0.value): # Circular beam, see https://en.wikipedia.org/wiki/Gaussian_beam, but of course it is also the special case
        #                               # of the elliptical beam defined below
        #    # First at positions z != z0
        #    E[mask] = self.E0 * self.w0 / wz[mask] * np.exp(-(x_local[mask]**2 + y_local[mask]**2) / wz[mask]**2 \
        #                                                    - 1j * self.k * (x_local[mask]**2 + y_local[mask]**2) / (2*Rz[mask]) \
        #                                                    + 1j * Psiz[mask] \
        #                                                    - 1j * self.k * z_local[mask])
        #    # Treat position z=z0 separately
        #    E[~mask] = self.E0 * np.exp(-(x_local[~mask]**2 + y_local[~mask]**2) / wz[~mask]**2)
#
        #else: # Elliptical beam, see equations (62) in chap 16 with c00=E0*sqrt(wx0*wy0), (58) in chap. 16 with (5) in chap. 17 and (49) in chap. 16
        #    # First at positions z != z0
        #    E[mask] = self.E0 * np.sqrt(self.w0[0] / wz[0][mask]) * np.sqrt(self.w0[1] / wz[1][mask]) \
        #              * np.exp(x_local[mask]**2 / wz[0][mask]**2 + y_local[mask]**2 / wz[1][mask]**2 \
        #                       - 1j * self.k * (x_local[mask]**2 / (2*Rz[0][mask]) + y_local[mask]**2 / (2*Rz[1][mask])) \
        #                       + 1j * (Psiz[0][mask]/2 + Psiz[1][mask]/2) \
        #                       - 1j * self.k * z_local[mask]),
        #    # Treat position z=z0 separately
        #    E[~mask] = self.E0 * np.exp(x_local[~mask]**2 / wz[0][~mask]**2 + y_local[~mask]**2 / wz[1][~mask]**2)


        return np.squeeze(E).to(u.V/u.m) # if E is scalar, return E instead of np.array([E])


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
        E = self.E(x, y, z)
        Evec = np.array([
            E * self.pol_3d_vec[0],
            E * self.pol_3d_vec[1],
            E * self.pol_3d_vec[2],
        ]) * E.unit
        
        return Evec
    

    def I(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity],
            y: Union[float, Sequence[float], np.ndarray, u.Quantity],
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:    
        """Returns the intensity of the beam at the position (x,y,z) in [mW/cm^2]. Here, x, y, z are the global standard Carteesian
           coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Intensity of the beam at the position (x,y,z) in [mW/cm^2], can be either float or array.
        """
        E = self.E(x, y, z)
        I = (c*eps0/2 * np.abs(E)**2).to(u.mW/u.cm**2)

        return I.to(u.mW/u.cm**2)



    def _calculate_rotation_matrix(
            self, 
    ) -> np.ndarray:
        """Calculate the rotation matrix to transform from the local coordinate system to the global 
           coordinate system. 
           
           In the local coordinate system of this GaussianBeam instance, the 3d polarization vector 
           and the beam direction vector are given by:
           3d polarization = (pol_Jones_vec[0], pol_Jones_vec[1], 0)
           beam direction = (0, 0, 1) 

           In the global coordinate system, the 3d polarization vector and the beam direction vector 
           are given by:
           3d polarization = R * (pol_Jones_vec[0], pol_Jones_vec[1], 0)   # R is the rotation matrix
           beam direction = (beam_direction[0], beam_direction[1], beam_direction[2])

           Args:
                None

           Returns:
                np.ndarray: 3x3 rotation matrix to transform from the local coordinate system to the global standard 
                             Carteesian coordinate system.       
        """
        assert np.isclose(np.linalg.norm(self.beam_direction_vec), 1), "beam_direction must be a unit vector"

        # Choosing local x-axis based on the beam direction (this follows the convention described in the class docstring)
        if self.beam_direction_vec[0] != 0:
            local_x_axis = np.array([0, 1, 0])
        else:
            local_x_axis = np.array([1, 0, 0])

        # Computing local y-axis
        local_y_axis = np.cross(self.beam_direction_vec, local_x_axis)
        local_y_axis = local_y_axis / np.linalg.norm(local_y_axis)  # Normalize

        # Adjust local x-axis to ensure orthogonality
        local_x_axis = np.cross(local_y_axis, self.beam_direction_vec)

        # Constructing rotation matrix
        R = np.column_stack((local_x_axis, local_y_axis, self.beam_direction_vec))

        return R




    def _check_input(
            self, 
            method: str,
    ) -> None:
        """Checks the input of the method ``method``."""
        if method == 'init':
            # Check beam direction vector
            if not isinstance(self.beam_direction_vec, (Sequence, np.ndarray)): 
                raise TypeError('The beam_direction_vec must be a sequence.')
            if not len(self.beam_direction_vec) == 3:
                raise ValueError('The beam_direction_vec must be a 3d vector.')
            self.beam_direction_vec = np.asarray(self.beam_direction_vec)
            self.beam_direction_vec = self.beam_direction_vec / np.linalg.norm(self.beam_direction_vec)  # Normalize

            # Check polarization Jones vector
            if not isinstance(self.pol_Jones_vec, (Sequence, np.ndarray)):
                raise TypeError('The pol_Jones_vec must be a sequence.')
            if not len(self.pol_Jones_vec) == 2:
                raise ValueError('The pol_Jones_vec must be a 2d vector.')
            self.pol_Jones_vec = np.asarray(self.pol_Jones_vec)
            self.pol_Jones_vec = self.pol_Jones_vec / np.linalg.norm(self.pol_Jones_vec)  # Normalize

            # Check wavelength
            if isinstance(self.lambda_, (float, int)):
                self.lambda_ = self.lambda_ * u.nm
            elif isinstance(self.lambda_, u.Quantity) and self.lambda_.unit.is_equivalent(u.nm):
                if np.isscalar(self.lambda_.value):
                    self.lambda_ = self.lambda_.to(u.nm)
                else:
                    raise TypeError('The wavelength lambda_ must be an astropy.Quantity or float.')
            else:
                raise TypeError('The wavelength lambda_ must be an astropy.Quantity or float.')

            
            # Check beam waist diameter
            if isinstance(self.w0, (float, int)):
                self.w0 = self.w0 * u.um
            elif isinstance(self.w0, (Sequence, np.ndarray)) and not isinstance(self.w0, u.Quantity) and len(self.w0) == 2:
                self.w0 = np.asarray(self.w0) * u.um
            elif isinstance(self.w0, u.Quantity) and self.w0.unit.is_equivalent(u.um):
                if np.isscalar(self.w0.value):
                    self.w0 = self.w0.to(u.um)
                elif len(self.w0.value) == 2:
                    self.w0 = self.w0.to(u.um)
                else:
                    raise ValueError('The beam waist diameter w0 must be a scalar or sequence of two floats.')
            else:
                raise TypeError('The beam waist diameter w0 must be an astropy.Quantity or float or sequence of floats.')

            
            # Check power
            if isinstance(self.P, (float, int)):
                self.P = self.P * u.W
            elif isinstance(self.P, u.Quantity) and self.P.unit.is_equivalent(u.W):
                if np.isscalar(self.P.value):
                    self.P = self.P.to(u.W)
                else:
                    raise TypeError('The power P must be an astropy.Quantity or float.')
            else:
                raise TypeError('The power P must be an astropy.Quantity or float.')
            
            # Check position of the beam waist
            if isinstance(self.z0, (float, int)):
                self.z0 = self.z0 * u.um
            elif isinstance(self.z0, u.Quantity) and self.z0.unit.is_equivalent(u.um):
                if np.isscalar(self.z0.value):
                    self.z0 = self.z0.to(u.um)
                else:
                    raise TypeError('The position of the beam waist z0 must be an astropy.Quantity or float.')
            else:
                raise TypeError('The position of the beam waist z0 must be an astropy.Quantity or float.')
            
            # Check name
            if not isinstance(self.name, str):
                raise TypeError('The name must be a string.')


        elif method == 'w' or method == 'R' or method == 'Psi':
            # Check z
            if isinstance(self.z_local, (float, int)):
                self.z_local = np.array([self.z_local]) * u.um
            elif isinstance(self.z_local, (Sequence, np.ndarray)) and not isinstance(self.z_local, u.Quantity):
                self.z_local = np.asarray(self.z_local) * u.um
            elif isinstance(self.z_local, u.Quantity) and self.z_local.unit.is_equivalent(u.um):
                self.z_local = (np.atleast_1d(self.z_local.value) * self.z_local.unit).to(u.um)
            else:
                raise TypeError('The z-coordinate must be an astropy.Quantity or float or sequence of floats.')
            
            

        elif method == 'E' or method == 'E_vec' or method == 'I':
            # Check x
            if isinstance(self.x, (float, int)):
                self.x = np.array([self.x]) * u.um
            elif isinstance(self.x, (Sequence, np.ndarray)) and not isinstance(self.x, u.Quantity):
                self.x = np.asarray(self.x) * u.um
            elif isinstance(self.x, u.Quantity) and self.x.unit.is_equivalent(u.um):
                self.x = (np.atleast_1d(self.x.value) * self.x.unit).to(u.um)
            else:
                raise TypeError('The x-coordinate must be an astropy.Quantity or float or sequence of floats.')
            
            # Check y
            if isinstance(self.y, (float, int)):
                self.y = np.array([self.y]) * u.um
            elif isinstance(self.y, (Sequence, np.ndarray)) and not isinstance(self.y, u.Quantity):
                self.y = np.asarray(self.y) * u.um
            elif isinstance(self.y, u.Quantity) and self.y.unit.is_equivalent(u.um):
                self.y = (np.atleast_1d(self.y.value) * self.y.unit).to(u.um)
            else:
                raise TypeError('The y-coordinate must be an astropy.Quantity or float or sequence of floats.')
            
            # Check z
            if isinstance(self.z, (float, int)):
                self.z = np.array([self.z]) * u.um
            elif isinstance(self.z, (Sequence, np.ndarray)) and not isinstance(self.z, u.Quantity):
                self.z = np.asarray(self.z) * u.um
            elif isinstance(self.z, u.Quantity) and self.z.unit.is_equivalent(u.um):
                self.z = (np.atleast_1d(self.z.value) * self.z.unit).to(u.um)
            else:
                raise TypeError('The z-coordinate must be an astropy.Quantity or float or sequence of floats.')
            

        elif method == 'E_sym':
            # Check x
            if not isinstance(self.x, sp.Symbol):
                raise TypeError('The x-coordinate must be a sympy.Symbol.')
            if not self.x.is_real:
                raise TypeError('The x-coordinate must be a real sympy.Symbol.')
            
            # Check y
            if not isinstance(self.y, sp.Symbol):
                raise TypeError('The y-coordinate must be a sympy.Symbol.')
            if not self.y.is_real:
                raise TypeError('The y-coordinate must be a real sympy.Symbol.')
            
            # Check z
            if not isinstance(self.z, sp.Symbol):
                raise TypeError('The z-coordinate must be a sympy.Symbol.')
            if not self.z.is_real:
                raise TypeError('The z-coordinate must be a real sympy.Symbol.')


        
    def w_sym(
            self,
            z_local: sp.Symbol,
    ): 
        w0 = self.w0.to(u.um).value
        z0 = self.z0.to(u.um).value
        z_R = self.z_R.to(u.um).value

        # Calculate beam diameter at given local z positions
        if np.isscalar(self.w0.value): # Circular beam
            w = w0 * sp.sqrt(1 + (z_local - z0)**2 / z_R**2)
        else: # Elliptical beam
            w = sp.Array([
                w0[0] * sp.sqrt(1 + (z_local - z0)**2 / z_R[0]**2),
                w0[1] * sp.sqrt(1 + (z_local - z0)**2 / z_R[1]**2),
            ]) 

        return w 



    def R_sym(
            self,
            z_local: sp.Symbol,
    ): 
        z_R = self.z_R.to(u.um).value
        z0 = self.z0.to(u.um).value

        # Calculate radius of wavefront curvature at given local z positions
        if np.isscalar(self.w0.value): # Circular beam
            R = z_local + z_R**2 / (z_local - z0)
        else: # Elliptical beam
            R = sp.Array([
                z_local + z_R[0]**2 / (z_local - z0),
                z_local + z_R[1]**2 / (z_local - z0),
            ]) 

        return R
    


    def Psi_sym(
            self,
            z_local: sp.Symbol,
    ): 
        z_R = self.z_R.to(u.um).value
        z0 = self.z0.to(u.um).value

        # Calculate Gouy phase at given local z positions
        if np.isscalar(self.w0.value): # Circular beam
            Psi = sp.atan((z_local - z0) / z_R)
        else: # Elliptical beam
            Psi = sp.Array([
                sp.atan((z_local - z0) / z_R[0]), 
                sp.atan((z_local - z0) / z_R[1]), 
            ])

        return Psi



    def E_sym(
            self, 
            x: sp.Symbol, 
            y: sp.Symbol, 
            z: sp.Symbol,
    ):
        self.x = x
        self.y = y
        self.z = z
        self._check_input('E_sym')

        # Transform gobal coordinates to local coordinates
        r = sp.Matrix([self.x, self.y, self.z])
        r_local = self._rotation_matrix_sym.T * r # inverse of rotation matrix is transpose
        x_local, y_local, z_local = r_local[0], r_local[1], r_local[2]

        # Calculate beam propagation parameters at given local z positions
        wz = self.w_sym(z_local)
        Rz = self.R_sym(z_local)
        Psiz = self.Psi_sym(z_local)

        E0 = self.E0.to(u.V/u.m).value
        w0 = self.w0.to(u.um).value
        k = self.k.to(1/u.um).value

        # Calculate electric field strength
        if np.isscalar(self.w0.value): # Circular beam, see https://en.wikipedia.org/wiki/Gaussian_beam, but of course it is also the special case
                                       # of the elliptical beam defined below
            E = E0 * w0 / wz * sp.exp(-(x_local**2 + y_local**2) / wz**2 \
                                      - 1j * k * (x_local**2 + y_local**2) / (2*Rz) \
                                      + 1j * Psiz \
                                      - 1j * k * z_local)

        else: # Elliptical beam, see equations (62) in chap 16 with c00=E0*sqrt(wx0*wy0), (58) in chap. 16 with (5) in chap. 17 and (49) in chap. 16
            E = E0 * sp.sqrt(w0[0] / wz[0]) * sp.sqrt(w0[1] / wz[1]) \
                        * sp.exp(-(x_local**2 / wz[0]**2 + y_local**2 / wz[1]**2) \
                                 - 1j * k * (x_local**2 / (2*Rz[0]) + y_local**2 / (2*Rz[1])) \
                                 + 1j * (Psiz[0]/2 + Psiz[1]/2) \
                                 - 1j * k * z_local)

        return E 
    


    def E_vec_sym(
            self,
            x: sp.Symbol, 
            y: sp.Symbol, 
            z: sp.Symbol,
    ):
        E = self.E_sym(x, y, z)
        Evec = E * sp.Matrix(self.pol_3d_vec)
        
        return Evec
    

    def I_sym(
            self,
            x: sp.Symbol, 
            y: sp.Symbol, 
            z: sp.Symbol,
    ):
        E = self.E_sym(x, y, z)
        I = (c.to(u.m/u.s).value*eps0.value/2 * abs(E)**2)

        return I



    def _calculate_rotation_matrix_sym(
        self,
    ):
        beam_direction_vec = sp.Matrix(self.beam_direction_vec)
        beam_direction_vec = beam_direction_vec / beam_direction_vec.norm()  # Normalize
        #assert beam_direction_vec.norm() == 1, "beam_direction_vec must be a unit vector"
        
        # Choosing local x-axis based on the beam direction
        if beam_direction_vec[0] != 0:
            local_x_axis = sp.Matrix([0, 1, 0])
        else:
            local_x_axis = sp.Matrix([1, 0, 0])
        
        # Computing local y-axis
        local_y_axis = beam_direction_vec.cross(local_x_axis)
        local_y_axis = local_y_axis / local_y_axis.norm()  # Normalize
        
        # Adjust local x-axis to ensure orthogonality
        local_x_axis = local_y_axis.cross(beam_direction_vec)
        
        # Constructing rotation matrix
        R = sp.Matrix.hstack(local_x_axis, local_y_axis, beam_direction_vec)
        
        return R