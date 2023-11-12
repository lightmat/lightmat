import numpy as np
from abc import ABC, abstractmethod

class Lattice(ABC):
    def __init__(self,):
        self.beams = []

    @abstractmethod
    def create_lattice(self):
        pass

    @abstractmethod
    def electric_field_amplitude(self, x, y, z):
        pass

    @abstractmethod
    def depth(self,):
        pass

    @abstractmethod
    def lamb_dicke(self,):
        pass

    @abstractmethod
    def plot(self, x, y, z):
        pass