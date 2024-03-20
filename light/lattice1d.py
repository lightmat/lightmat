from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .gaussianbeam import GaussianBeam


class Lattice1d(object):

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
        self.lattice_direction_vec = np.asarray(lattice_direction_vec)
        self.intersect_angle = intersect_angle
        self.pol_Jones_vec = pol_Jones_vec
        self.lambda_ = lambda_
        self.w0 = w0
        self.P = P
        self.z0 = z0

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
    