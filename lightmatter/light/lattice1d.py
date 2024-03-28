from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .laser import Laser
from .gaussianbeam import GaussianBeam


class Lattice1d(Laser):

    def __init__(
            self,
            lattice_direction_vec: Sequence[float],
            intersect_angle: float,
            pol_Jones_vec: Sequence[float],
            lambda_: Union[u.Quantity, float],
            w0: Union[u.Quantity, float, Sequence[float], np.ndarray],
            P: Union[u.Quantity, float],
            z0: Union[u.Quantity, float] = 0 * u.um,
    ) -> None:
        """Initializes a Lattice1d instance.
        
           Args:
                beam_direction_vec: 3d vector specifying the beam propagation in the global standard Carteesian 
                                    coordinate system.
                intersect_angle: Angle in [deg] between the two counterpropagating beams making up the lattice.
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
        self.lattice_direction_vec = lattice_direction_vec
        self.intersect_angle = intersect_angle
        self.pol_Jones_vec = pol_Jones_vec
        self.lambda_ = lambda_
        self.w0 = w0
        self.P = P
        self.z0 = z0
        self._check_input('init')

        super().__init__(self.lambda_, self.P)

        # Calculate the beam directions
        self.beam_direction_forward_vec, self.beam_direction_backward_vec = self._calculate_beam_directions()

        # Create the two GaussianBeam instances for the two counterpropagating lattice beams
        self.beam_forward = GaussianBeam(
            beam_direction_vec=self.beam_direction_forward_vec,
            pol_Jones_vec=self.pol_Jones_vec,
            lambda_=self.lambda_,
            w0=self.w0,
            P=self.P,
            z0=self.z0,
        )

        self.beam_backward = GaussianBeam(
            beam_direction_vec=self.beam_direction_backward_vec,
            pol_Jones_vec=self.pol_Jones_vec,
            lambda_=self.lambda_,
            w0=self.w0,
            P=self.P,
            z0=self.z0,
        )




    def E_vec_sym(
            self,
            x: sp.Symbol,
            y: sp.Symbol,
            z: sp.Symbol,
    ):
        E_forward = self.beam_forward.E_vec_sym(x, y, z)
        E_backward = self.beam_backward.E_vec_sym(x, y, z)
        return E_forward + E_backward

        
        
    def E_vec(
            self,
            x: Union[u.Quantity, float, Sequence[float], np.ndarray],
            y: Union[u.Quantity, float, Sequence[float], np.ndarray],
            z: Union[u.Quantity, float, Sequence[float], np.ndarray],
    ) -> u.Quantity:
        """Returns the electric field vector of the 1d lattice at the given position.
        
           Args:
                x: x-coordinate of the position in [um].
                y: y-coordinate of the position in [um].
                z: z-coordinate of the position in [um].

           Returns:
                np.ndarray: The electric field vector of the lattice beam at the given position.
        """
        E_forward = self.beam_forward.E_vec(x, y, z)
        E_backward = self.beam_backward.E_vec(x, y, z)
        return E_forward + E_backward
    

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
    


    def E(
            self,
            x: Union[u.Quantity, float, Sequence[float], np.ndarray],
            y: Union[u.Quantity, float, Sequence[float], np.ndarray],
            z: Union[u.Quantity, float, Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Returns the electric field of the 1d lattice at the given position.
        
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
    ) -> np.ndarray:
        """Returns the intensity of the 1d lattice at the given position.
        
           Args:
                x: x-coordinate of the position in [um].
                y: y-coordinate of the position in [um].
                z: z-coordinate of the position in [um].

           Returns:
                np.ndarray: The intensity of the lattice beam at the given position.
        """
        return c*eps0/2 * np.abs(self.E(x, y, z))**2
    



    def _calculate_beam_directions(
            self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the direction vectors of the two counterpropagating lattice beams.
        
           Returns:
                Tuple[np.ndarray, np.ndarray]: The direction vectors of the two counterpropagating lattice beams.
        """
        # Ensure lattice_direction is a unit vector
        self.lattice_direction_vec = self.lattice_direction_vec / np.linalg.norm(self.lattice_direction_vec)

        # Calculate the rotation axis as perpendicular to the lattice direction.
        # This can be any vector not parallel to lattice_direction, but a common choice
        # is to use the cross product with a standard basis vector that is not aligned with it.
        # For simplicity, choose the z-axis or any other axis that is not parallel to lattice_direction.
        z_axis = np.array([0, 0, 1])
        if np.allclose(self.lattice_direction_vec, z_axis):
            # If lattice_direction is parallel to the z-axis, choose a different axis for the cross product
            z_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(self.lattice_direction_vec, z_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the rotation axis

        # Create rotation objects for rotating around the calculated axis by ±half_angle
        rotation_forward = R.from_rotvec(rotation_axis * np.deg2rad(self.intersect_angle / 2))
        rotation_backward = R.from_rotvec(rotation_axis * np.deg2rad(-self.intersect_angle / 2))

        # Apply rotations to the lattice_direction to get the beam directions
        beam_direction_forward_vec = rotation_forward.apply(self.lattice_direction_vec)
        beam_direction_backward_vec = - rotation_backward.apply(self.lattice_direction_vec)

        return beam_direction_forward_vec, beam_direction_backward_vec
    


    def _check_input(
            self, 
            method: str,
    ) -> None:
        """Checks the input of the method ``method``."""
        if method == 'init':
            # Check lattice direction vector
            if not isinstance(self.lattice_direction_vec, (Sequence, np.ndarray)): 
                raise TypeError('The beam_direction_vec must be a sequence.')
            if not len(self.lattice_direction_vec) == 3:
                raise ValueError('The beam_direction_vec must be a 3d vector.')
            self.lattice_direction_vec = np.asarray(self.lattice_direction_vec)
            self.lattice_direction_vec = self.lattice_direction_vec / np.linalg.norm(self.lattice_direction_vec)  # Normalize

            # Check intersect angle
            if not isinstance(self.intersect_angle, (float, int)):
                raise TypeError('The intersect_angle must be a float.')
            if self.intersect_angle > 180 or self.intersect_angle < 0:
                raise ValueError('The intersect_angle must be in the range [0, 180].')

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
    