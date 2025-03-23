import astropy.units as u
from astropy.constants import hbar, c, eps0, e, a0, h, hbar, k_B
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sympy.physics.wigner import wigner_3j, wigner_6j
from typing import Union
from collections.abc import Sequence
from typing import Union, Tuple


from .light import Laser, Beam
from .matter import Atom
from .many_body.spatial_basis import SpatialBasisSet, GridSpatialBasisSet

class LaserSetup(object):
    
    def __init__(
            self,
            lasers: Union[Laser, Beam, Sequence[Union[Laser, Beam]]],
            atoms: Union[Atom, Sequence[Atom], None] = None, 
    ) -> None:
        """Initialise the LaserSetup object. Either provide a single laser or a sequence of lasers. If an atom or 
           sequence of atoms is provided, the potential of the laser setup in the atom's hfs state can be calculated 
           with self.V() (sequence of potentials if atoms is a sequence of atoms).
        
           Args:
               lasers: Laser or sequence of lasers.
               atoms: Atom object representing the atom for which the potential is to be generated. Can either be
                      single atom or sequence of atoms. Default is None.
        """
        self.lasers = lasers
        self.atoms = atoms
        self._check_input("init")

        self.x = None
        self.y = None
        self.z = None
        self.Es = [None]*len(self.lasers) # E-field amplitudes for each laser at the position (x,y,z)


    def V(
            self,
            x: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            y: Union[float, Sequence[float], np.ndarray, u.Quantity], 
            z: Union[float, Sequence[float], np.ndarray, u.Quantity],
            unit: u.Unit = u.MHz,
    ) -> u.Quantity:
        """Returns the potential of the `atoms` in their hfs state given the light field of the `lasers` at the position (x,y,z) in [h x MHz],
           or in [kB x nK] or in [eV] depending on ``unit``. 
           If `atoms` just has a single atom, the potential is returned as a single u.Quantity of same shape as (x,y,z). If `atoms` is a sequence 
           of atoms, then a list of u.Quantity is returned, each element of the list corresponding to the potential of the respective atom in the
           sequence.
           Here, x, y, z are the global standard Carteesian coordinates in [um] and can be either float or array obtained from np.meshgrid().

           Args:
                x: Global standard Carteesian coordinate in [um].
                y: Global standard Carteesian coordinate in [um].
                z: Global standard Carteesian coordinate in [um].
                unit: Unit of the potential. Must be one of [u.MHz, u.uK, u.eV]. Default is u.MHz.

           Returns:
                u.Quantity: Potential of the `atom` in its hfs state given the light field of the `lasers` at the position (x,y,z) in [h x MHz],
                            or in [kB x nK] or in [eV] depending on ``unit``. .
        """
        self.x_tmp = x
        self.y_tmp = y
        self.z_tmp = z
        self.unit = unit
        self._check_input('V')

        # List of potentials for each atom in the sequence of atoms
        Vs = []

        for atom in self.atoms:
            F = atom.hfs_state['F']
            mF = atom.hfs_state['mF']

            # Add the potential from each laser to the total potential of the atom
            V = 0 * u.MHz # in [h x MHz]
            for i, laser in enumerate(self.lasers):
                # Calculate the electric field amplitude of the laser at the position (x,y,z):
                    # print(f'Calculating electric field amplitude of laser {laser.name}...')
                E = laser.E(self.x_tmp, self.y_tmp, self.z_tmp)
                self.Es[i] = E
                E_squared = np.real(self.Es[i] * np.conj(self.Es[i])) # this is real anyways, just get rid of complex cast warnings

                # Calculate the polarizabilities of the atom in the hfs state for the laser frequency
                # print(f'Calculating polarizability of hfs state {atom.hfs_state} at λ={laser.lambda_}...')
                alpha_s = atom.scalar_hfs_polarizability(omega_laser=laser.omega)
                alpha_v = atom.vector_hfs_polarizability(omega_laser=laser.omega)
                alpha_t = atom.tensor_hfs_polarizability(omega_laser=laser.omega)

                # Calculate the coefficients, equation (20) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
                if laser.pol_vec_3d is not None:
                    C = np.real(2 * np.imag(np.conj(laser.pol_vec_3d[0]) * laser.pol_vec_3d[1])) # this is real anyways, just get rid of complex cast warnings
                    D = np.real(1 - 3*np.conj(laser.pol_vec_3d[2]) * laser.pol_vec_3d[2]) # this is real anyways, just get rid of complex cast warnings
                else:
                    raise ValueError("The potential can only be calculated if all lasers have a specified pol_vec_3d.")
                
                # Calculate the potential, equation (19) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
                V = V + (-1/4 * E_squared * (alpha_s + C*alpha_v*mF/(2*F) - D*alpha_t*(3*mF**2 - F*(F+1)) / (2*F*(2*F-1)))).to(u.MHz)

            # Append the potential of the atom and set the position, such that the electric field is not recalculated
            Vs.append(V.value) # in [h x MHz]
            self.x = self.x_tmp
            self.y = self.y_tmp
            self.z = self.z_tmp

        Vs = np.asarray(Vs) * u.MHz
        if unit == u.nK:
            Vs = (Vs * h / k_B).to(u.nK)
        elif unit == u.eV:
            Vs = (Vs * h).to(u.eV)

        return np.squeeze(Vs).real # in [h x MHz], [kB x nK] or [eV]. Return as scalar if only one atom and as array if sequence of atoms



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
            if self.atoms is not None:
                if isinstance(self.atoms, (Atom, Sequence, np.ndarray)):
                    if isinstance(self.atoms, (Sequence, np.ndarray)):
                        for i, atom in enumerate(self.atoms):
                            if not isinstance(atom, Atom):
                                raise TypeError(f"atoms must be an instance of Atom, not {type(atom)}")
                    elif isinstance(self.atoms, Atom):
                        self.atoms = [self.atoms]
                else:
                    raise TypeError(f"atom must be an instance of Atom or sequence of Atoms, not {type(self.atom)}")
            

        if method == 'V':
            # Check x
            if isinstance(self.x_tmp, u.Quantity) and self.x_tmp.unit.is_equivalent(u.um):
                self.x_tmp = self.x_tmp.to(u.um).value
            if isinstance(self.x_tmp, (float, int)):
                self.x_tmp = np.array([self.x_tmp]) * u.um
            elif isinstance(self.x_tmp, (Sequence, np.ndarray)) and not isinstance(self.x_tmp, u.Quantity):
                self.x_tmp = np.asarray(self.x_tmp) * u.um
            else:
                raise TypeError('The x-coordinate must be an astropy.Quantity or float or sequence of floats.')
            
            # Check y
            if isinstance(self.y_tmp, u.Quantity) and self.y_tmp.unit.is_equivalent(u.um):
                self.y_tmp = self.y_tmp.to(u.um).value
            if isinstance(self.y_tmp, (float, int)):
                self.y_tmp = np.array([self.y_tmp]) * u.um
            elif isinstance(self.y_tmp, (Sequence, np.ndarray)) and not isinstance(self.y_tmp, u.Quantity):
                self.y_tmp = np.asarray(self.y_tmp) * u.um
            else:
                raise TypeError('The y-coordinate must be an astropy.Quantity or float or sequence of floats.')
            
            # Check z
            if isinstance(self.z_tmp, u.Quantity) and self.z_tmp.unit.is_equivalent(u.um):
                self.z_tmp = self.z_tmp.to(u.um).value
            if isinstance(self.z_tmp, (float, int)):
                self.z_tmp = np.array([self.z_tmp]) * u.um
            elif isinstance(self.z_tmp, (Sequence, np.ndarray)) and not isinstance(self.z_tmp, u.Quantity):
                self.z_tmp = np.asarray(self.z_tmp) * u.um
            else:
                raise TypeError('The z-coordinate must be an astropy.Quantity or float or sequence of floats.')
            

            # Check unit
            if not isinstance(self.unit, u.Unit):
                raise TypeError('unit must be an astropy.Unit.')
            elif not self.unit.is_equivalent(u.MHz) and not self.unit.is_equivalent(u.uK) and not self.unit.is_equivalent(u.eV):
                raise ValueError('unit must be one of [u.MHz, u.uK, u.eV].')
            elif self.unit.is_equivalent(u.MHz):
                self.unit = u.MHz
            elif self.unit.is_equivalent(u.nK):
                self.unit = u.nK
            elif self.unit.is_equivalent(u.eV):
                self.unit = u.eV


