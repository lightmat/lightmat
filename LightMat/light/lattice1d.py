from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .laser import Laser
from .beams import GaussianBeam


class Lattice1d(Laser):

    def __init__(
            self,
            lattice_direction_vec: Sequence[float],
            intersect_angle: float,
            lambda_: Union[u.Quantity, float],
            w0: Union[u.Quantity, float, Sequence[float], np.ndarray],
            P: Union[u.Quantity, float, Sequence[float], np.ndarray],
            intersect_axis: Union[Sequence[float], np.ndarray, None] = None,
            pol_Jones_vec: Union[str, Sequence[float], Sequence[str]] = 'linear horizontal',
            z0: Union[u.Quantity, float, Sequence[float], np.ndarray] = 0 * u.um,
            name: str = 'Lattice1d',
    ) -> None:
        """Initializes a Lattice1d Laser instance. It hosts two beams interfering with each other to form a 1d lattice.
        
           Args:
                lattice_direction_vec: 3d vector specifying the 1d lattice direction in the global standard Carteesian 
                                       coordinate system.
                intersect_angle: Angle in [deg] between the two counterpropagating beams making up the lattice.
                lambda_: Wavelength of the two beams in [nm]. 
                w0: Beam waist diameter in [um]. Either a scalar for the same circular beam waist for both beams or a 2d vector
                    for the same elliptical beam waist for both beams, or a list of two 2d vectors for different beam waists
                    for the two lattice beams.
                P: Power of the beam in [W]. Either a scalar for the same power in both beams or a 2d vector for different power
                   in the two lattice beams.
                intersect_axis: 3d vector specifying the axis around which the two counterpropagating beams are 
                                rotated apart by ``intersect_angle``. If None, than the z-axis is chosen, unless the
                                ``lattice_direction_vec`` is along z-axis, then the y-axis. Defaults to None.
                pol_Jones_vec: 2d vector specifying the polarization of the beam in the local x-y plane. The convention of
                               is that the horizontal polarization is along the local x-direction and the vertical polarization
                               along the local y-direction. Either a 2d vector for the same polarization for both beams or a list
                               of two 2d vectors for different polarizations for the two lattice beams. Alternatively to the 2d
                               vector, a string in ['linear horizontal', 'linear vertical', 'circular right', 'circular left'] 
                               can be passed to specify the polarization. Defaults to 'linear horizontal'.
                z0: Position of the beam waist in [um] along the beam propagation direction. either a scalar for same beam waist
                    position for both beams or a 2d vector for different beam waist positions for the two lattice beams.
                    Defaults to 0um.
                name: Name of the Lattice1d instance. Defaults to 'Lattice1d'.

           Returns:
                None           
        """
        self.lattice_direction_vec = lattice_direction_vec
        self.intersect_angle = intersect_angle
        self.lambda_ = lambda_
        self.w0 = w0
        self.P = P
        self.intersect_axis = intersect_axis
        self.pol_Jones_vec = pol_Jones_vec
        self.z0 = z0
        self.name = name
        self._check_input('init')

        # Calculate the beam directions
        self.beam_direction_forward_vec, self.beam_direction_backward_vec = self._calculate_beam_directions()

        # Create the two GaussianBeam instances for the two counterpropagating lattice beams
        self.beam_forward = GaussianBeam(
            beam_direction_vec=self.beam_direction_forward_vec,
            pol_Jones_vec=self.pol_Jones_vec if isinstance(self.pol_Jones_vec, str) or \
                                                (self.pol_Jones_vec.ndim == 1 and not isinstance(self.pol_Jones_vec[0], str)) else self.pol_Jones_vec[0],
            lambda_=self.lambda_ if self.lambda_.isscalar else self.lambda_[0],
            w0=self.w0 if self.w0.ndim == 1 or self.w0.isscalar else self.w0[0],
            P=self.P if self.P.isscalar else self.P[0],
            z0=self.z0 if self.z0.isscalar else self.z0[0],
        )

        self.beam_backward = GaussianBeam(
            beam_direction_vec=self.beam_direction_backward_vec,
            pol_Jones_vec=self.pol_Jones_vec if isinstance(self.pol_Jones_vec, str) or \
                                                (self.pol_Jones_vec.ndim == 1 and not isinstance(self.pol_Jones_vec[1], str)) else self.pol_Jones_vec[1],
            lambda_=self.lambda_ if self.lambda_.isscalar else self.lambda_[1],
            w0=self.w0 if self.w0.ndim == 1 or self.w0.isscalar else self.w0[1],
            P=self.P if self.P.isscalar else self.P[1],
            z0=self.z0 if self.z0.isscalar else self.z0[1],
        )

        super().__init__(
            name = self.name,
            beams = [self.beam_forward, self.beam_backward], 
        )   
    


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
        rotation_axis = self.intersect_axis
        if rotation_axis is None:
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
                raise ValueError('The intersect_angle must be in the range [0°, 180°].')

            # Check wavelength
            if isinstance(self.lambda_, u.Quantity):
                if not self.lambda_.unit.is_equivalent(u.nm):
                    raise ValueError('The wavelength must be in units equivalent to nm.')
            elif isinstance(self.lambda_, (float, int)):
                self.lambda_ = self.lambda_ * u.nm
            else:
                raise ValueError('The wavelength must be an astropy Quantity or a float.')

            
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
                        raise ValueError('The beam waist diameter must be a scalar or a 2d vector or a list of two 2d vectors.')
                elif self.w0.ndim == 2:
                    if len(self.w0) != 2 or len(self.w0[0]) != 2 or len(self.w0[1]) != 2:
                        raise ValueError('The beam waist diameter must be a scalar or a 2d vector or a list of two 2d vectors.')
                else:
                    raise ValueError('The beam waist diameter must be a scalar or a 2d vector or a list of two 2d vectors.')
                self.w0 = self.w0 * u.um

            
            # Check power
            if isinstance(self.P, u.Quantity):
                if not self.P.unit.is_equivalent(u.W):
                    raise ValueError('The power must be in units equivalent to W.')
            elif isinstance(self.P, (float, int)):
                self.P = self.P * u.W
            elif isinstance(self.P, (Sequence, np.ndarray)):
                self.P = np.asarray(self.P)
                if len(self.P) != 2:
                    raise ValueError('The power must be a scalar or a 2d vector.')
                self.P = self.P * u.W
            else:
                raise ValueError('The power must be a scalar or a 2d vector.')
            

            # Check intersect axis
            if self.intersect_axis is not None:
                if not isinstance(self.intersect_axis, (Sequence, np.ndarray)):
                    raise TypeError('The intersect_axis must be a sequence.')
                if not len(self.intersect_axis) == 3:
                    raise ValueError('The intersect_axis must be a 3d vector.')
                self.intersect_axis = np.asarray(self.intersect_axis)
                self.intersect_axis = self.intersect_axis / np.linalg.norm(self.intersect_axis)
                if np.allclose(self.intersect_axis, self.lattice_direction_vec):
                    raise ValueError('The intersect_axis must not be parallel to the lattice direction vector.')


            # Check polarization Jones vector
            if isinstance(self.pol_Jones_vec, str):
                if self.pol_Jones_vec not in ['linear horizontal', 'linear vertical', 'circular right', 'circular left']:
                    raise ValueError('The polarization Jones vector must be one of the following strings: ["linear horizontal", "linear vertical", "circular right", "circular left"].')
            elif isinstance(self.pol_Jones_vec, (Sequence, np.ndarray)):
                self.pol_Jones_vec = np.asarray(self.pol_Jones_vec) 
                if self.pol_Jones_vec.ndim == 1:
                    if len(self.pol_Jones_vec) != 2:
                        raise ValueError('The polarization Jones vector must be a 2d vector or a list of two 2d vectors.')                
                elif self.pol_Jones_vec.ndim == 2:
                    if len(self.pol_Jones_vec) != 2 or len(self.pol_Jones_vec[0]) != 2 or len(self.pol_Jones_vec[1]) != 2:
                        raise ValueError('The polarization Jones vector must be a 2d vector or a list of two 2d vectors.')
                else:
                    raise ValueError('The polarization Jones vector must be a 2d vector or a list of two 2d vectors.')
            else:
                raise ValueError('The polarization Jones vector must be a 2d vector or string or a list of two 2d vectors or two strings.')
            
            
            # Check position of the beam waist
            if isinstance(self.z0, u.Quantity):
                if not self.z0.unit.is_equivalent(u.um):
                    raise ValueError('The position of the beam waist must be in units equivalent to um.')
            elif isinstance(self.z0, (float, int)):
                self.z0 = self.z0 * u.um
            elif isinstance(self.z0, (Sequence, np.ndarray)):
                self.z0 = np.asarray(self.z0)
                if len(self.z0) != 2:
                    raise ValueError('The position of the beam waist must be a scalar or a 2d vector.')
                self.z0 = self.z0 * u.um
            else:
                raise ValueError('The position of the beam waist must be a scalar or a 2d vector.')
            
            # Check name
            if not isinstance(self.name, str):
                raise TypeError('The name must be a string.')
    