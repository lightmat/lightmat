from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sympy as sp
from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple
from collections.abc import Sequence

from .laser import Laser


class LaserSetup(object):
    
        def __init__(
                self,
                lasers: Union[Laser, Sequence[Laser]],
        ) -> None:
            """Initializes a LaserSetup instance.
    
            Args:
                    lasers: A single or sequence of Laser instances.
    
            Returns:
                    None           
            """
            self.lasers = lasers
            self._check_input('init')

        

        def I_sym(
                self,
                x: sp.Symbol,
                y: sp.Symbol,
                z: sp.Symbol,
        ):
            I = 0
            for laser in self.lasers:
                I = I + laser.I_sym(x, y, z)

            return I
        


        def I(
                self,
                x: Union[u.Quantity, float, Sequence[float], np.ndarray],
                y: Union[u.Quantity, float, Sequence[float], np.ndarray],
                z: Union[u.Quantity, float, Sequence[float], np.ndarray],
        ) -> u.Quantity:
            """Returns the intensity of the laser setup at the given position.
    
            Args:
                    x: x-coordinate of the position in [um].
                    y: y-coordinate of the position in [um].
                    z: z-coordinate of the position in [um].
    
            Returns:
                    np.ndarray: The intensity of the lattice beam at the given position.
            """
            I = 0 * u.mW / u.cm**2
            for laser in self.lasers:
                I = I + laser.I(x, y, z)

            return I.to(u.mW/u.cm**2)



        def define_arrowhead(self, direction, start_point=np.array([0, 0, 0]), length=0.15, radius=0.03):
            """Generates coordinates for a half-cone arrowhead."""            
            # Normalize the direction vector
            direction = np.array(direction) / np.linalg.norm(direction)
            
            # Check if the direction is parallel or anti-parallel to the z-axis
            if np.allclose(direction, [0, 0, 1]):
                # Direction is parallel to the z-axis
                rotation_matrix = np.eye(3)  # Identity matrix, no rotation needed
            elif np.allclose(direction, [0, 0, -1]):
                # Direction is anti-parallel to the z-axis
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, -1]])  # 180-degree rotation around the x-axis
            else:
                # General case, create a rotation matrix that aligns the z-axis with the direction vector
                # Using the Rodrigues' rotation formula
                k = np.cross([0, 0, 1], direction)
                k = k / np.linalg.norm(k)
                theta = np.arccos(direction[2])  # Angle between direction and z-axis
                
                K = np.array([[0, -k[2], k[1]],
                            [k[2], 0, -k[0]],
                            [-k[1], k[0], 0]])
                rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

            # Define a semi-circle in the xy-plane for the base of the cone
            theta = np.linspace(0, np.pi, 30)  # Semi-circle for the top half
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.zeros_like(theta)
            
            # Create the points of the cone
            cone_base = np.vstack((x, y, z)).T
            cone_tip = np.array([0, 0, length]) # Tip is along the positive z-axis

            # Apply the rotation matrix to the base and the tip
            cone_base_rotated = cone_base @ rotation_matrix.T + start_point
            cone_tip_rotated = cone_tip @ rotation_matrix.T + start_point

            return cone_base_rotated, cone_tip_rotated



        def plot_beams(self):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('X', color='white', fontsize=16, labelpad=-10)
            ax.set_ylabel('Y', color='white', fontsize=16, labelpad=-10)
            ax.set_zlabel('Z', color='white', fontsize=16, labelpad=-10)
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            fig.tight_layout()
            
            colors = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue', 'orange', 'purple']
            
            for i, laser in enumerate(self.lasers):
                color = colors[i % len(colors)] if laser.color is None else laser.color
                for j, direction in enumerate(laser.beam_directions):
                    # Normalize the direction
                    direction = np.array(direction) / np.linalg.norm(direction)

                    # The beam should pass through the origin and extend to the ends of the axes
                    # Determine the scale factor based on the axis limits
                    scale_factor = max(abs(ax.get_xlim()[0]), abs(ax.get_ylim()[0]), abs(ax.get_zlim()[0]))

                    # Extend the beam in both positive and negative direction from the origin
                    end_point_positive = direction * scale_factor
                    end_point_negative = direction * -scale_factor

                    # Plotting the beam line with shading effect
                    for k in range(1, 15):
                        ax.plot(*np.column_stack((end_point_negative, end_point_positive)), color=color,
                                linewidth=1 + k * 0.5, alpha=1 / k,
                                label=laser.name if j == 0 and k == 1 else None)
                    
                    # Plotting the half-cone (arrowhead) only in the positive direction for simplicity
                    cone_base, cone_tip = self.define_arrowhead(direction, end_point_positive)
                    if cone_base is not None and cone_tip is not None:
                        vertices = [list(zip(cone_base[:,0], cone_base[:,1], cone_base[:,2])) + [cone_tip.tolist()]]
                        ax.add_collection3d(Poly3DCollection(vertices, color=color))

            fig.legend()
            plt.show()





        def _check_input(
                    self, 
                    method,
        ) -> None:
            """Checks the input for the LaserSetup instance."""
            if method == 'init':
                if not isinstance(self.lasers, (Laser, Sequence)):
                    raise ValueError("The input beams must be a single or sequence of Laser instances.")
                if isinstance(self.lasers, Laser):
                    self.lasers = np.array([self.lasers])
                if isinstance(self.lasers, Sequence):
                    for laser in self.lasers:
                        if not isinstance(laser, Laser):
                            raise ValueError("The input beams must be a single or sequence of Laser instances.")
