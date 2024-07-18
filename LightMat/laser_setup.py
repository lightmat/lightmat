import astropy.units as u
from astropy.constants import hbar, c, eps0, e, a0, h, hbar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sympy.physics.wigner import wigner_3j, wigner_6j
from typing import Union
from collections.abc import Sequence
from typing import Union


from .light import Laser, Beam
from .matter import Atom

class LaserSetup(object):
    
    def __init__(
            self,
            lasers: Union[Laser, Beam, Sequence[Union[Laser, Beam]]],
            atom: Union[Atom, None] = None, 
    ) -> None:
        """Initialise the LaserSetup object. Either provide a single laser or a sequence of lasers and if an atom 
           object is provided, the potential of the atom in its hfs state can be calculated with self.V().
        
           Args:
               lasers: Laser or sequence of lasers.
               atom: Atom object representing the atom for which the potential is to be generated. Default is None.
        """
        self.lasers = lasers
        self.atom = atom
        self._check_input("init")



    def V(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
    ) -> u.Quantity:
        """Returns the potential of the `atom` in its hfs state given the light field of the `lasers` at the position (x,y,z) in [h x MHz]. 
           Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].

           Returns:
                u.Quantity: Potential of the `atom` in its hfs state given the light field of the `lasers` at the position (x,y,z) in [h x MHz].
        """
        F = self.atom.hfs_state['F']
        mF = self.atom.hfs_state['mF']

        # Add the potential from each laser to the total potential
        V = 0 * u.MHz # in [h x MHz]

        for laser in self.lasers:
            # Calculate the electric field amplitude of the laser at the position (x,y,z)
            E_squared = np.real(laser.E(x, y, z) * np.conj(laser.E(x, y, z))) # this is real anyways, just get rid of complex cast warnings

            # Calculate the polarizabilities of the atom in the hfs state for the laser frequency
            alpha_s = self.atom.scalar_hfs_polarizability(laser.omega)
            alpha_v = self.atom.vector_hfs_polarizability(laser.omega)
            alpha_t = self.atom.tensor_hfs_polarizability(laser.omega)

            # Calculate the coefficients, equation (20) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
            if laser.pol_vec_3d is not None:
                C = 2 * np.imag(np.conj(laser.pol_vec_3d[0]) * laser.pol_vec_3d[1])
                D = 1 - 3*np.conj(laser.pol_vec_3d[2]) * laser.pol_vec_3d[2]
            else:
                raise ValueError("The potential can only be calculated if all lasers have a specified pol_vec_3d.")
            
            # Calculate the potential, equation (19) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
            V += (-1/4 * E_squared * (alpha_s + C*alpha_v*mF/(2*F) - D*alpha_t*(3*mF**2 - F*(F+1)) / (2*F*(2*F-1)))).to(u.MHz)

        return V



    def plot_beams(
            self,
    ) -> None:
        """Plots the beams of the lasers in the LaserSetup object in 3D."""
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
            
        colors = ['cyan', 'green', 'yellow', 'magenta', 'red', 'blue', 'orange', 'purple']
            
        for i, laser in enumerate(self.lasers):
            color = colors[i % len(colors)] 
            for j, beam in enumerate(laser.beams):
                # Normalize the direction
                direction = np.asarray(beam.beam_direction) 
                direction = direction / np.linalg.norm(direction)

                # The beam should pass through the origin and extend to the ends of the axes
                # Determine the scale factor based on the axis limits
                scale_factor = max(abs(ax.get_xlim()[0]), abs(ax.get_ylim()[0]), abs(ax.get_zlim()[0]))

                # Extend the beam in both positive and negative direction from the origin
                end_point_positive = direction * scale_factor
                end_point_negative = direction * -scale_factor

                # Plot transparent xy plane
                x = np.linspace(-1, 1, 2)
                y = np.linspace(-1, 1, 2)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)  
                ax.plot_surface(X, Y, Z, color='white', alpha=0.03) 

                # Plotting the beam line with shading effect
                for k in range(1, 5):
                    if end_point_positive[2] >= 0 and end_point_negative[2] < 0:
                        ax.plot(*np.column_stack((end_point_negative, [0, 0, 0])), color=color,
                                linewidth=1 + k * 0.5, alpha=0.2 / k,
                                label=None)
                        ax.plot(*np.column_stack(([0, 0, 0], end_point_positive)), color=color,
                                linewidth=1 + k * 0.5, alpha=1 / k,
                                label=None)
                        if k == 1:
                            ax.quiver(0, 0, 0, end_point_positive[0], end_point_positive[1], end_point_positive[2], 
                                    color=color, length=scale_factor, arrow_length_ratio=0.1, linewidth=2, alpha=1,)
                    elif end_point_positive[2] < 0 and end_point_negative[2] >= 0:
                        ax.plot(*np.column_stack((end_point_positive, [0, 0, 0])), color=color,
                                linewidth=1 + k * 0.5, alpha=0.2 / k,
                                label=None)
                        ax.plot(*np.column_stack(([0, 0, 0], end_point_negative)), color=color,
                                linewidth=1 + k * 0.5, alpha=1 / k,
                                label=None)
                        if k == 1:
                            ax.quiver(0, 0, 0, end_point_positive[0], end_point_positive[1], end_point_positive[2], 
                                    color=color, length=scale_factor, arrow_length_ratio=0.1, linewidth=2, alpha=0.2,)
                    elif end_point_positive[2] < 0 and end_point_negative[2] < 0:
                        ax.plot(*np.column_stack((end_point_positive, end_point_negative)), color=color,
                                linewidth=1 + k * 0.5, alpha=0.2 / k,
                                label=None)
                        if k == 1:
                            ax.quiver(0, 0, 0, end_point_positive[0], end_point_positive[1], end_point_positive[2], 
                                    color=color, length=scale_factor, arrow_length_ratio=0.1, linewidth=2, alpha=0.2,)
                    elif end_point_positive[2] >= 0 and end_point_negative[2] >= 0:
                        ax.plot(*np.column_stack((end_point_positive, end_point_negative)), color=color,
                                linewidth=1 + k * 0.5, alpha=1 / k,
                                label=None)
                        if k == 1:
                            ax.quiver(0, 0, 0, end_point_positive[0], end_point_positive[1], end_point_positive[2], 
                                    color=color, length=scale_factor, arrow_length_ratio=0.1, linewidth=2, alpha=1,)
                            
                    
                # Adding an invisible plot for the legend
                if j == 0:
                    ax.plot([], [], color=color, linewidth=2, alpha=1, label=laser.name)                 

        fig.legend()
        plt.show()




    def _check_input(
            self, 
            method: str,
    ) -> None:
        if method == "init":
            # Check lasers
            if isinstance(self.lasers, Sequence):
                for i, laser in enumerate(self.lasers):
                    if isinstance(laser, Beam):
                        self.lasers[i] = Laser(name=laser.name, beams=[laser], pol_vec_3d=laser.pol_vec_3d)
                    elif not isinstance(laser, Laser):
                        raise TypeError(f"lasers must be an instance of Laser or Beam, not {type(laser)}")
            elif isinstance(self.lasers, Beam):
                self.lasers = [Laser(name=self.lasers.name, beams=[self.lasers], pol_vec_3d=self.lasers.pol_vec_3d)]
            elif isinstance(self.lasers, Laser):
                self.lasers = [self.lasers]
            else:
                raise TypeError(f"lasers must be an instance of Laser or Beam, not {type(self.lasers)}")


            # Check atom
            if self.atom is not None and not isinstance(self.atom, Atom):
                raise TypeError(f"atom must be an instance of Atom, not {type(self.atom)}")
            


