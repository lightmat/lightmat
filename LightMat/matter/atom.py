import os
import astropy.units as u
from astropy.constants import hbar, c, eps0, e, a0, h, hbar
import numpy as np
import pandas as pd
from sympy.physics.wigner import wigner_3j, wigner_6j
from fractions import Fraction
from typing import Union
from collections.abc import Sequence



class Atom(object):
    
    def __init__(
            self,
            name: str,
            hfs_state: Union[dict[str, Union[int, str, float]], Sequence[dict[str, Union[int, str, float]]]],
            fs_transition_data: Union[pd.DataFrame, Sequence[pd.DataFrame]] = None,
    ) -> None:
        """Initialise the Atom object.
        
           Args:
               name: The name of the atom. If it is in the database, the transition data will be loaded automatically.
               hfs_state: The hfs_state of the atom, should be dict of form 
                          {'n': int, 'L': str, 'J': float, 'F': int, 'mF': int, 'I': int} 
                          for a hyperfine-structure state. 
               fs_transition_data: Dataframe containing the transition data of the fines tructure state, should have columns 
                                   ['transition', 'wavelength', 'reduced_dipole_element', 'linewidth'], where transition is 
                                   a dict to a fs_state {'n': int, 'L': str, 'J': float}, wavelength is in [nm], reduced dipole
                                   element is in atomic units [e*a0] and linewidth is in [hbar x MHz]. Defaults to None.
        """
        self.name = name
        self.hfs_state = hfs_state
        self.fs_transition_data = fs_transition_data
        self._check_input('init')

        self.fs_state = {'n': self.hfs_state['n'], 'L': self.hfs_state['L'], 'J': self.hfs_state['J']}
        self.fs_transition_data_pretty = self._make_transition_data_pretty()

        

    def scalar_hfs_polarizability(
            self,
            omega_laser: Union[u.Quantity, float, Sequence[float], np.ndarray, None] = None,
            lambda_laser: Union[u.Quantity, float, Sequence[float], np.ndarray, None] = None,
    ) -> u.Quantity:
        """Calculate the scalar polarizability of the provided hfs state of the atom (equation (18) in 
           http://dx.doi.org/10.1140/epjd/e2013-30729-x).
        
           Args:
               omega_laser: Laser frequency for which the the polarizability is to be calculated. If float, it
                            is assumed to be in [hbar x THz], if u.Quantity, it must be in unit equivalent to [Hz].
                            Can either be scalar or sequence. Defaults to None.
               lambda_laser: Alternatively, the laser wavelength can be provided. If float, it is assumed to be in [nm],
                             if u.Quantity, it must be in unit equivalent to [nm]. Can either be scalar or sequence.
                             Defaults to None.
                       
           Returns:
               u.Quantity: The scalar polarizability of the atomic hfs state in [h x Hz / (V/m)^2] in same shape as omega_laser
                           or lambda_laser.
        """
        self.omega_laser = omega_laser
        self.lambda_laser = lambda_laser
        self._check_input('polarizability')

        J = self.hfs_state['J']
            
        # Calculate the scalar polarizability, see equation (18) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
        alpha_s = 1 / (np.sqrt(3*(2*J+1))) * self._reduced_dynamical_polarizability(
                                                                K=0, 
                                                                omega_laser=self.omega_laser,
                                                            )
        return alpha_s # in [h x Hz / (V/m)^2]



    def vector_hfs_polarizability(
            self,
            omega_laser: Union[u.Quantity, float, Sequence[float], np.ndarray, None] = None,
            lambda_laser: Union[u.Quantity, float, Sequence[float], np.ndarray, None] = None,
    ) -> u.Quantity:
        """Calculate the vector polarizability of the provided hfs state of the atom (equation (18) in 
           http://dx.doi.org/10.1140/epjd/e2013-30729-x).
        
           Args:
               omega_laser: Laser frequency for which the the polarizability is to be calculated. If float, it
                            is assumed to be in [hbar x THz], if u.Quantity, it must be in unit equivalent to [Hz].
                            Can either be scalar or sequence. Defaults to None.
               lambda_laser: Alternatively, the laser wavelength can be provided. If float, it is assumed to be in [nm],
                             if u.Quantity, it must be in unit equivalent to [nm]. Can either be scalar or sequence.
                             Defaults to None.
                       
           Returns:
               u.Quantity: The vector polarizability of the atomic hfs state in [h x Hz / (V/m)^2] in same shape as omega_laser
                           or lambda_laser.
        """
        self.omega_laser = omega_laser
        self.lambda_laser = lambda_laser
        self._check_input('polarizability')
        
        J = self.hfs_state['J']
        F = self.hfs_state['F']
        I = self.hfs_state['I']

        # Calculate the vector polarizability, see equation (18) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
        wigner = float(wigner_6j(F, 1, F, J, I, J).evalf())
        alpha_v = (-1)**(J*I*F) * np.sqrt(2*F*(2*F+1) / (F+1)) * wigner * self._reduced_dynamical_polarizability(
                                                                K=1, 
                                                                omega_laser=self.omega_laser,
                                                            )
        return alpha_v # in [h x Hz / (V/m)^2]



    def tensor_hfs_polarizability(
            self,
            omega_laser: Union[u.Quantity, float, Sequence[float], np.ndarray, None] = None,
            lambda_laser: Union[u.Quantity, float, Sequence[float], np.ndarray, None] = None,
    ) -> u.Quantity:
        """Calculate the tensor polarizability of the provided hfs state of the atom (equation (18) in 
           http://dx.doi.org/10.1140/epjd/e2013-30729-x).
        
           Args:
               omega_laser: Laser frequency for which the the polarizability is to be calculated. If float, it
                            is assumed to be in [hbar x THz], if u.Quantity, it must be in unit equivalent to [Hz].
                            Can either be scalar or sequence. Defaults to None.
               lambda_laser: Alternatively, the laser wavelength can be provided. If float, it is assumed to be in [nm],
                             if u.Quantity, it must be in unit equivalent to [nm]. Can either be scalar or sequence.
                             Defaults to None.
                       
           Returns:
               u.Quantity: The tensor polarizability of the atomic hfs state in [h x Hz / (V/m)^2] in same shape as omega_laser
                           or lambda_laser.
        """
        self.omega_laser = omega_laser
        self.lambda_laser = lambda_laser
        self._check_input('polarizability')

        J = self.hfs_state['J']
        F = self.hfs_state['F']
        I = self.hfs_state['I']

        # Calculate the tensor polarizability, see equation (18) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
        wigner = float(wigner_6j(F, 2, F, J, I, J).evalf())
        alpha_t = -(-1)**(J*I*F) * np.sqrt(2*F*(2*F-1)*(2*F+1) / (3*(F+1)*(2*F+3))) * wigner * self._reduced_dynamical_polarizability(
                                                                K=2, 
                                                                omega_laser=self.omega_laser,
                                                            )
        return alpha_t # in [h x Hz / (V/m)^2]
    



    def _reduced_dynamical_polarizability(
            self,
            K: int,
            omega_laser: Union[u.Quantity, float],
    ) -> u.Quantity:
        """Calculate the scalar (K=0), vector (K=1) or tensor (K=2) reduced dynamical polarizability of the 
           atomic fine structure state given by equation (11) in http://dx.doi.org/10.1140/epjd/e2013-30729-x.

           Args:
                K: Order of the polarizability. Must be 0, 1 or 2.
                omega_laser: Laser frequency for which the the polarizability is to be calculated. If float, it
                             is assumed to be in [hbar x THz], if u.Quantity, it must be in unit equivalent to [Hz].

            Returns:
                u.Quantity: The polarizability of the atomic fine structure state in [h x Hz / (V/m)^2].
        """
        self.K = K
        self.omega_laser = omega_laser
        self._check_input('polarizability_reduced')
        
        J = self.fs_state['J']

        # Calculate the fs polarizability, see equation (11) in http://dx.doi.org/10.1140/epjd/e2013-30729-x
        alpha = 0
        for _, row in self.fs_transition_data.iterrows():
            J_prime = row['transition']['J']
            reduced_dipole_element = row['reduced_dipole_element'] * e.si*a0
            omega_transition = ((2*np.pi * c) / (row['wavelength']*u.nm)).to(u.MHz)
            gamma_transition = row['linewidth'] * u.MHz
            wigner = float(wigner_6j(1, self.K, 1, J, J_prime, J).evalf())

            alpha += (-1)**(self.K+J+1+J_prime) * wigner * reduced_dipole_element**2 * 1/hbar *\
                     np.real(1/(omega_transition - omega_laser - 1j*gamma_transition/2) + \
                              (-1)**K/(omega_transition + omega_laser + 1j*gamma_transition/2))

        return (np.sqrt(2*self.K + 1) * alpha / h).to(u.Hz / (u.V/u.m)**2) # in [h x Hz / (V/m)^2]



    def _load_transition_data_from_Savronova(
            self,
    ) -> None:
        """Load the transition data from the database found in 
           Parinaz Barakhshan, Adam Marrs, Akshay Bhosale, Bindiya Arora, Rudolf Eigenmann, Marianna S. Safronova, 
           Portal for High-Precision Atomic Data and Computation (version 2.0). University of Delaware, Newark, DE, USA. 
           URL: https://www.udel.edu/atom/ [February 2022].
        """
        # Specify fs_state based on hfs_state
        fs_state = {'n': self.hfs_state['n'], 'L': self.hfs_state['L'], 'J': self.hfs_state['J']}

        # Read atomic spectra data from csv file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file_path = os.path.join(script_dir, 'atomic_data/' + self.name + '1TransitionRates.csv')
        df = pd.read_csv(csv_file_path, usecols=[0, 1, 5, 8,])

        # Rename columns
        df = df.rename(columns={
            'Initial': 'initial', 
            'Final': 'final', 
            'Wavelength (nm)': 'wavelength', 
            'Transition rate (s-1)': 'transition_rate',
            })

        # Convert initial and final states from string to dict {'n': int, 'J': float}
        def fraction_to_float(fraction_str):
            numerator, denominator = fraction_str.split('/')
            return float(numerator) / float(denominator)

        pattern = r'(\d+)([a-zA-Z]+)(\d+/\d+)'
        df['initial'] = df['initial'].str.extract(pattern).apply(lambda x: {
            'n': int(x[0]), 
            'L': x[1],
            'J': fraction_to_float(x[2]), 
            }, axis=1)
        df['final'] = df['final'].str.extract(pattern).apply(lambda x: {
            'n': int(x[0]), 
            'L': x[1],
            'J': fraction_to_float(x[2]), 
            }, axis=1)


        # Create a new DataFrame with 'transitions' column
        filtered_df = df[df['initial'].apply(lambda x: x == fs_state) | df['final'].apply(lambda x: x == fs_state)]
        
        transitions = []
        adjusted_wavelengths = []
        for index, row in filtered_df.iterrows():
            if row['final'] == fs_state:
                transitions.append(row['initial'])
                adjusted_wavelengths.append(row['wavelength'])  # Positive if fs_state is 'final'
            else:
                transitions.append(row['final'])
                adjusted_wavelengths.append(-row['wavelength'])  # Negative if fs_state is 'initial'

        fs_transitions_df = filtered_df.copy()
        fs_transitions_df['transition'] = transitions
        fs_transitions_df['wavelength'] = adjusted_wavelengths  # Assign the adjusted wavelengths
        fs_transitions_df = fs_transitions_df.drop(columns=['initial', 'final'])
        fs_transitions_df = fs_transitions_df[['transition', 'wavelength', 'transition_rate']]
        fs_transitions_df.reset_index(drop=True, inplace=True)


        # Calculate the reduced dipole element 
        J_primes = np.asarray(fs_transitions_df['transition'].apply(lambda x: x['J']))
        wavelengths = np.asarray(fs_transitions_df['wavelength'].values) * u.nm
        omegas = 2 * np.pi * c / np.abs(wavelengths)
        transition_rates = np.asarray(fs_transitions_df['transition_rate'].values) * 1/u.s

        reduced_dipole_elements = np.sqrt(
            (transition_rates * 3*np.pi * eps0 * hbar * c**3 * (2 * J_primes + 1)) / (omegas**3)
        ).to(u.C * u.m)
        reduced_dipole_elements_au = reduced_dipole_elements / (e.si * a0)

        fs_transitions_df['reduced_dipole_element'] = reduced_dipole_elements_au.value


        # Calculate the transition linewidths
        linewidth_fs_state = (df[df['initial'].apply(lambda x: x == fs_state)]['transition_rate'].sum() * u.Hz).to(u.MHz).value

        linewidths = []
        for state in fs_transitions_df['transition']:
            linewidth_state = (df[df['initial'].apply(lambda x: x == state)]['transition_rate'].sum() * u.Hz).to(u.MHz).value
            linewidths.append(linewidth_state + linewidth_fs_state)

        fs_transitions_df['linewidth'] = linewidths
        fs_transitions_df = fs_transitions_df.drop(columns=['transition_rate'])

        # Save the transition data
        self.fs_transition_data = fs_transitions_df



    def _make_transition_data_pretty(
            self,
    ) -> pd.DataFrame:
        """Create a pretty version of the transition data dataframe.
               
            Returns:
                pd.DataFrame: A pretty version of the transition data dataframe.
        """            
        fs_transition_data_pretty = self.fs_transition_data.copy().rename(columns={
            'transition': 'Transition',
            'wavelength': 'Wavelength [nm]',
            'reduced_dipole_element': 'Reduced dipole element [e*a0]',
            'linewidth': 'Linewidth [hbar x MHz]',
        })

        def transition_to_str(state):
            fraction = Fraction(state['J']).limit_denominator()
            fraction_str = f"{fraction.numerator}/{fraction.denominator}"
            return str(state['n']) + state['L'] + fraction_str 

        fs_transition_data_pretty['Transition'] = fs_transition_data_pretty['Transition'].apply(
                                                      lambda x: transition_to_str(self.fs_state) + ' -> ' + transition_to_str(x)
                                                  )
        #fs_transition_data_pretty.set_caption("Fine structure transition data of " + transition_to_str(fs_state))

        return fs_transition_data_pretty



    def _check_input(
            self,
            method: str,
        ) -> None:
        """Check that the input values are valid."""
        if method == 'init':
            # Check name
            name_list = ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',]
            if not isinstance(self.name, str):
                raise TypeError("name must be a str.")
            if self.name not in name_list:
                if not isinstance(self.fs_transition_data, pd.DataFrame):
                    raise ValueError("If the name is not in " + str(name_list) + ", the transition_data must be provided as pd.dataframe.")
                else:
                    if not all(col in self.fs_transition_data.columns for col in ['transition', 'wavelength', 'reduced_dipole_element', 'linewidth']):
                        raise ValueError("transition_data must have columns ['transition', 'wavelength', 'reduced_dipole_element', 'linewidth'].")
                    if not isinstance(self.fs_transition_data['transition'].iloc[0], dict):
                        raise TypeError("The 'transition' column must be of type dict.")
                    if not all(isinstance(self.fs_transition_data[col].iloc[0], float) for col in ['wavelength', 'reduced_dipole_element', 'linewidth']):
                        raise TypeError("The 'wavelength', 'reduced_dipole_element', and 'linewidth' columns must be of type float.")
                    print("WARNING: The name is not in the data base. The provided transition_data will be used.")
            else:
                if self.fs_transition_data is not None:
                    print("WARNING: The name is in the data base. The provided transition_data will be ignored. Just change the name if you want to actually use the provided transition_data.")
                self._load_transition_data_from_Savronova()
                

            # Check hfs_state
            if isinstance(self.hfs_state, dict):
                if not all(key in self.hfs_state for key in ['n', 'L', 'J', 'mF', 'I']):
                    raise ValueError("hfs_state must be a dict of form {'n': int, 'L': str, 'J': float, 'F': float, 'mF': int, 'I': int}.")
                if not isinstance(self.hfs_state['n'], int):
                    raise TypeError("n must be an int.")
                if not self.hfs_state['L'] in ['s', 'p', 'd', 'f', 'g', 'h', 'i']:
                    raise TypeError("L must be a str in ['s', 'p', 'd', 'f', 'g', 'h', 'i'].")
                if not isinstance(self.hfs_state['J'], (float, int)):
                    raise TypeError("J must be a float.")
                if not isinstance(self.hfs_state['F'], (float, int)):
                    raise TypeError("F must be a float.")
                if not isinstance(self.hfs_state['mF'], (float, int)):
                    raise TypeError("mF must be an float.")
                if not isinstance(self.hfs_state['I'], (float, int)):
                    raise TypeError("I must be an float.")
            else:
                raise TypeError("hfs_state must be a dict of form {'n': int, 'L': str, 'J': float, 'F': float, 'mF': int, 'I': int}.")
            

        if method == 'polarizability' or method == 'polarizability_reduced':
            # Check omega_laser
            if isinstance(self.omega_laser, u.Quantity):
                if not self.omega_laser.unit.is_equivalent(u.Hz):
                    raise ValueError("omega_laser must be in unit equivalent to [Hz].")
            elif isinstance(self.omega_laser, float):
                self.omega_laser = self.omega_laser * u.THz
            elif isinstance(self.omega_laser, (Sequence, np.ndarray)):
                self.omega_laser = np.asarray(self.omega_laser) * u.THz
            elif self.omega_laser is not None:
                raise TypeError("omega_laser must be a float or u.Quantity.")
            
            # Check lambda_laser
            if isinstance(self.lambda_laser, u.Quantity):
                if not self.lambda_laser.unit.is_equivalent(u.nm):
                    raise ValueError("lambda_laser must be in unit equivalent to [nm].")
            elif isinstance(self.lambda_laser, float):
                self.lambda_laser = self.lambda_laser * u.nm
            elif isinstance(self.lambda_laser, (Sequence, np.ndarray)):
                self.lambda_laser = np.asarray(self.lambda_laser) * u.nm
            elif self.lambda_laser is not None:
                raise TypeError("lambda_laser must be a float or u.Quantity.")
            
            self.omega_laser = self.omega_laser if self.omega_laser is not None else 2 * np.pi * c / self.lambda_laser
            
            # Check K
            if method == 'polarizability_reduced':
                if self.K not in [0, 1, 2]:
                    raise ValueError("K must be 0, 1 or 2.")
            

                
