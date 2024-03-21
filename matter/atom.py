from astropy import units as u
from astropy.constants import c, eps0
import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence



class Atom(object):
    
    def __init__(
            self,
            name: str,
            state: Union[str, Sequence[str]],
            transition_data: Union[None, pd.DataFrame] = None,
            backend: Union[None, str] = None,
    ) -> None:
        
        self._name = name
        self._state = state
        self._transition_data = transition_data
        self._backend = backend

        self._check_input()



    def _check_input(self,):
        """Check that the input values are valid."""

        # Check name
        if not isinstance(self._name, str):
            raise TypeError("name must be a str.")
        if self._name not in ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',] and self._transition_data == None:
            raise ValueError("The name must be one of ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',] or the transition_data\
                              must be provided as pd.dataframe.")
        
        # Check state
        if not isinstance(self._state, Union[str, Sequence[str]]):
            raise TypeError("state must be a str or Sequence of str.")
        
