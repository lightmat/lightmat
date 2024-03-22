from astropy import units as u
from astropy.constants import c, eps0, h
import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence



class Atom(object):
    
    def __init__(
            self,
            name: str,
            fs_state: Union[dict[str, float], Sequence[dict[str, float]]],
            I: float = None,
            transition_data: pd.DataFrame = None,
            a_hfs: Union[float, Sequence[float]] = None,
            b_hfs: Union[float, Sequence[float]] = None,
    ) -> None:
        """Initialise the Atom object.
        
           Args:
               name: The name of the atom. If it is in the database, the transition data and nuclear spin will be loaded 
                     automatically.
               fs_state: The fs_state of the atom, should be dict of form {'n': int, 'l': str, 'j': float} for a fine-structure
                         state. It is also possible to provide a list of fs_states. If it is in the database, the hyperfine 
                         structure constants a_hfs and b_hfs of the fs_state will be loaded automatically.
               I: The nuclear spin of the atom. Defaults to None.
               transition_data: Dataframe containing the transition data, should have columns ['initial', 'final', 'wavelength',
                                'A'], where A is the lifetime of the transition in [s^-1] and wavelength is in [nm]. Here, 'initial' 
                                and 'final' should be strings of the form 'nlj' where n is int, l is in [s,p,d,f,...], j is float.
                                Defaults to None.
               a_hfs: The a hyperfine structure constant of the specified fine-structure fs_state(s). Defaults to None.
               b_hfs: The b hyperfine structure constant of the specified fine-structure fs_state(s). Defaults to None.
        """
        self.name = name
        self.fs_state = fs_state
        self.I = I
        self.transition_data = transition_data
        self.a_hfs = a_hfs
        self.b_hfs = b_hfs
        self._check_input('init')



    def hfs_energy_split(
            self,
            F: Union[float, Sequence[float]],
    ) -> Union[float, Sequence[float]]:
        self.F = F
        self._check_input('hfs_energy_split')

        Es = []
        for F in self.F:
            I = self.I
            J = self.state['J']
            G = F*(F+1) - I*(I+1) - J*(J+1)

            E = (0.5 * self.a_hfs * G / h).to(u.MHz)
            if self.b_hfs != None:
                E = E + ((self.b_hfs * (1.5*G*(G+1) - 2*I*(I+1)*J*(J+1)) / (2*I*(2*I-1)*2*J*(2*J-1))) / h).to(u.MHz)
            
            Es.append(E)
        
        Es = np.array(Es)

        return np.squeeze(Es)


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
                if self.I is None:
                    raise ValueError("If the name is not in " + str(name_list) + ", the nuclear spin I must be provided.")
                if self.transition_data is None:
                    raise ValueError("If the name is not in " + str(name_list) + ", the transition_data must be provided as pd.dataframe.")
                print("WARNING: The name is not in the data base. The provided transition_data and nuclear spin I will be used.")
            
            # Check fs_state
            if not isinstance(self.fs_state, (dict, Sequence)):
                raise TypeError("fs_state must be a str or Sequence of str.")
            if isinstance(self.fs_state, dict):
                if not all(key in self.fs_state for key in ['n', 'l', 'j']):
                    raise ValueError("fs_state must be a dict of form {'n': int, 'l': str, 'j': float}.")
                if not isinstance(self.fs_state['n'], int):
                    raise TypeError("n must be an int.")
                if not isinstance(self.fs_state['l'], str):
                    raise TypeError("l must be a str.")
                if not isinstance(self.fs_state['j'], float):
                    raise TypeError("j must be a float.")
            if isinstance(self.fs_state, Sequence):
                for fs_state in self.fs_state:
                    if not all(key in fs_state for key in ['n', 'l', 'j']):
                        raise ValueError("fs_state must be a dict of form {'n': int, 'l': str, 'j': float}.")
                    if not isinstance(fs_state['n'], int):
                        raise TypeError("n must be an int.")
                    if not isinstance(fs_state['l'], str):
                        raise TypeError("l must be a str.")
                    if not isinstance(fs_state['j'], float):
                        raise TypeError("j must be a float.")
            self.fs_state = np.atleast_1d(self.fs_state)
                    
            # Check I
            if self.I is not None:
                if not isinstance(self.I, int):
                    raise TypeError("I must be an int.")
                if self.name in name_list:
                    print("WARNING: The name is in the data base, but the provided nuclear spin I will be used.")
            else:
                pass #TODO: get I from database
                
                
            # Check transition_data
            if self.transition_data is not None:
                if not isinstance(self.transition_data, pd.DataFrame):
                    raise TypeError("transition_data must be a pd.DataFrame.")
                if not all(col in self.transition_data.columns for col in ['initial', 'final', 'wavelength', 'A']):
                    raise ValueError("transition_data must have columns ['initial', 'final', 'wavelength', 'A'].")
                if not all(isinstance(self.transition_data[col].iloc[0], str) for col in ['initial', 'final']):
                    raise TypeError("The 'initial' and 'final' columns must be of type str.")
                if not all(isinstance(self.transition_data[col].iloc[0], float) for col in ['wavelength', 'A']):
                    raise TypeError("The 'wavelength' and 'A' columns must be of type float.")
                if self.name in name_list:
                    print("WARNING: The name is in the data base, but the provided transition_data will be used.")
            else:
                pass #TODO: get transition_data from database


            # Check a_hfs
            if self.a_hfs is not None:
                if not isinstance(self.a_hfs, (float, Sequence)):
                    raise TypeError("a_hfs must be a float or Sequence of float.")
                if isinstance(self.a_hfs, Sequence):
                    if len(self.a_hfs) != len(self.b_hfs) or len(self.a_hfs) != len(self.fs_state):
                        raise ValueError("The length of a_hfs must be the same as b_hfs and fs_state.")
                    for a_hfs in self.a_hfs:
                        if not isinstance(a_hfs, float):
                            raise TypeError("a_hfs must be a float.")
                self.a_hfs = np.atleast_1d(self.a_hfs)
            else:
                self.a_hfs = []
                for fs_state in self.fs_state:
                    pass #TODO: get a_hfs from database

            # Check b_hfs
            if self.b_hfs is not None:
                if not isinstance(self.b_hfs, (float, Sequence)):
                    raise TypeError("b_hfs must be a float or Sequence of float.")
                if isinstance(self.b_hfs, Sequence):
                    if len(self.b_hfs) != len(self.a_hfs) or len(self.b_hfs) != len(self.fs_state):
                        raise ValueError("The length of b_hfs must be the same as a_hfs and fs_state.")
                    for b_hfs in self.b_hfs:
                        if not isinstance(b_hfs, float):
                            raise TypeError("b_hfs must be a float.")
                self.b_hfs = np.atleast_1d(self.b_hfs)
            else:
                self.b_hfs = []
                for fs_state in self.fs_state:
                    pass #TODO: get b_hfs from database
                

        
