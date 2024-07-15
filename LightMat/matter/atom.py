import os
import astropy.units as u
from astropy.constants import hbar, c, eps0, e, a0, h, hbar
import numpy as np
import pandas as pd
import pywigxjpf as wig
from typing import Union
from collections.abc import Sequence



class Atom(object):
    
    def __init__(
            self,
            name: str,
            hfs_state: Union[dict[str, Union[int, str, float]], Sequence[dict[str, Union[int, str, float]]]],
            I: float,
            fs_transition_data: Union[pd.DataFrame, Sequence[pd.DataFrame]] = None,
    ) -> None:
        """Initialise the Atom object.
        
           Args:
               name: The name of the atom. If it is in the database, the transition data will be loaded automatically.
               hfs_state: The hfs_state of the atom, should be dict of form {'n': int, 'L': str, 'J': float, 'F': int, 'mF': int} 
                          for a hyperfine-structure state. It is also possible to provide a list of hfs_states. 
               I: The nuclear spin of the atom. 
               fs_transition_data: Dataframe containing the transition data of the fines tructure state, should have columns 
                                   ['transition', 'wavelength', 'reduced_dipole_element', 'linewidth'], where transition is 
                                   a dict to a fs_state {'n': int, 'L': str, 'J': float}, wavelength is in [nm], reduced dipole
                                   element is in atomic units [ea0] and linewidth is in [MHz]. If `hfs_state` is a list of states
                                   then `transition_data` is a list of dataframes. Defaults to None.
        """
        self.name = name
        self.hfs_state = hfs_state
        self.I = I
        self.fs_transition_data = fs_transition_data
        self._check_input('init')




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
        transitions = filtered_df.apply(lambda row: row['initial'] if row['final'] == fs_state else row['final'], axis=1)
        fs_transitions_df = filtered_df.copy()
        fs_transitions_df['transition'] = transitions
        fs_transitions_df = fs_transitions_df.drop(columns=['initial', 'final'])
        fs_transitions_df = fs_transitions_df[['transition', 'wavelength', 'transition_rate']]
        fs_transitions_df.reset_index(drop=True, inplace=True)


        # Calculate the reduced dipole element 
        J_primes = np.asarray(fs_transitions_df['transition'].apply(lambda x: x['J']))
        wavelengths = np.asarray(fs_transitions_df['wavelength'].values) * u.nm
        omegas = 2 * np.pi * c / wavelengths
        transition_rates = np.asarray(fs_transitions_df['transition_rate'].values) * 1/u.s

        reduced_dipole_elements = np.sqrt(
            (transition_rates * 3 * np.pi * eps0 * hbar * c**3 * (2 * J_primes + 1)) / (omegas**3)
        ).to(u.C * u.m)
        reduced_dipole_elements_au = reduced_dipole_elements / (e.si * a0)

        fs_transitions_df['reduced_dipole_element'] = reduced_dipole_elements_au.value


        # Calculate the transition linewidths
        linewidth_fs_state = (df[df['initial'].apply(lambda x: x == fs_state)]['transition_rate'].sum() / (2*np.pi) * u.Hz).to(u.MHz).value

        linewidths = []
        for state in fs_transitions_df['transition']:
            linewidth_state = (df[df['initial'].apply(lambda x: x == state)]['transition_rate'].sum() / (2*np.pi) * u.Hz).to(u.MHz).value
            linewidths.append(linewidth_state + linewidth_fs_state)

        fs_transitions_df['linewidth'] = linewidths
        fs_transitions_df = fs_transitions_df.drop(columns=['transition_rate'])

        # Save the transition data
        self.fs_transition_data = fs_transitions_df



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
                    print("WARNING: The name is in the data base. The provided transition_data will be ignored.")
                self._load_transition_data_from_Savronova()
                

            # Check hfs_state
            if isinstance(self.hfs_state, dict):
                if not all(key in self.hfs_state for key in ['n', 'L', 'J', 'mF']):
                    raise ValueError("hfs_state must be a dict of form {'n': int, 'L': str, 'J': float, 'F': float, 'mF': int}.")
                if not isinstance(self.hfs_state['n'], int):
                    raise TypeError("n must be an int.")
                if not self.hfs_state['L'] in ['s', 'p', 'd', 'f', 'g', 'h', 'i']:
                    raise TypeError("L must be a str in ['s', 'p', 'd', 'f', 'g', 'h', 'i'].")
                if not isinstance(self.hfs_state['J'], float):
                    raise TypeError("J must be a float.")
                if not isinstance(self.hfs_state['F'], float):
                    raise TypeError("F must be a float.")
                if not isinstance(self.hfs_state['mF'], int):
                    raise TypeError("mF must be an int.")
            elif isinstance(self.hfs_state, Sequence):
                for hfs_state in self.hfs_state:
                    if not isinstance(hfs_state, dict):
                        raise TypeError("hfs_state must be a dict or list of dicts of form {'n': int, 'L': str, 'J': float, 'F': float, 'mF': int}.")
                    if not all(key in hfs_state for key in ['n', 'L', 'J', 'mF']):
                        raise ValueError("hfs_state must be a dict of form {'n': int, 'L': str, 'J': float, 'F': float, 'mF': int}.")
                    if not isinstance(hfs_state['n'], int):
                        raise TypeError("n must be an int.")
                    if not hfs_state['L'] in ['s', 'p', 'd', 'f', 'g', 'h', 'i']:
                        raise TypeError("L must be a str in ['s', 'p', 'd', 'f', 'g', 'h', 'i'].")
                    if not isinstance(hfs_state['J'], float):
                        raise TypeError("J must be a float.")
                    if not isinstance(hfs_state['F'], float):
                        raise TypeError("F must be a float.")
                    if not isinstance(hfs_state['mF'], int):
                        raise TypeError("mF must be an int.")
            else:
                raise TypeError("hfs_state must be a dict or list of dicts of form {'n': int, 'L': str, 'J': float, 'F': float, 'mF': int}.")
            
                    
            # Check I
            if not isinstance(self.I, int):
                raise TypeError("I must be an int.")
                
