import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Quantity
from typing import Union, Sequence, Callable


class SpacialPlot():
    def __init__(
            self,
            func: Union[Callable, Sequence[Callable], np.ndarray],
            func_name: Union[str, Sequence[str], np.ndarray] = 'f',
        ):
        # Check that func is a callable or sequence of callables
        if isinstance(func, Callable):
            self.funcs = [func]
        elif isinstance(func, (Sequence, np.ndarray)):
            if any(not isinstance(f, Callable) for f in func):
                raise TypeError("func must be a Callable or Sequence of Callables.")
            self.funcs = list(func)
        else:
            raise TypeError("func must be a Callable or Sequence of Callables.")
        
        # Check that func_name is a string or sequence of strings
        if isinstance(func_name, str):
            self.func_names = [func_name]
        elif isinstance(func_name, (Sequence, np.ndarray)):
            if any(not isinstance(f, str) for f in func_name):
                raise TypeError("func_name must be a str or Sequence of str.")
            self.func_names = list(func_name)


    def plot_1d_overview(
            self, 
            xpos: Quantity,
            ypos: Quantity,
            zpos: Quantity,
            **kwargs,
        ):

        # Check that xpos is of the form (x,y,z) where x is a sequence and y,z are numbers
        if not isinstance(xpos, Quantity) or not isinstance(xpos, (Sequence, np.ndarray)):
            raise TypeError("xpos must be a Quantity of the form (x,y,z) where x is a Sequence and y,z are numbers.")
        if len(xpos) != 3 or len(xpos[0]) == 1:
            raise ValueError("xpos must be a Quantity of the form (x,y,z) where x is a Sequence and y,z are numbers.")

        # Check that ypos is of the form (x,y,z) where y is a sequence and x,z are numbers
        if not isinstance(ypos, Quantity) or not isinstance(ypos, (Sequence, np.ndarray)):
            raise TypeError("ypos must be a Quantity of the form (x,y,z) where y is a Sequence and x,z are numbers.")
        if len(ypos) != 3 or len(ypos[1]) == 1:
            raise ValueError("ypos must be a Quantity of the form (x,y,z) where y is a Sequence and x,z are numbers.")
        
        # Check that zpos is of the form (x,y,z) where z is a sequence and x,y are numbers
        if not isinstance(zpos, Quantity) or not isinstance(zpos, (Sequence, np.ndarray)):
            raise TypeError("zpos must be a Quantity of the form (x,y,z) where z is a Sequence and x,y are numbers.")
        if len(zpos) != 3 or len(zpos[2]) == 1:
            raise ValueError("zpos must be a Quantity of the form (x,y,z) where z is a Sequence and x,y are numbers.")

        suptitle = kwargs.get('suptitle', None)  
        filename = kwargs.get('filename', None) 

        fig, axs = plt.subplots(1, 3, figsize=(14, 6))
        plt.subplots_adjust(top=0.85)
        fig.suptitle(suptitle, fontsize=24)

        axs[0].set_xlabel('x [' + str(xpos[0].unit) + ']', fontsize=14)
        axs[0].set_title(self.func_names[0] + '(x, ' + str(xpos[1]) + ', ' + str(xpos[2]) + ')', fontsize=18)

        axs[1].set_xlabel('y [' + str(ypos[1].unit) + ']', fontsize=14)
        axs[1].set_title(self.func_names[0] + '(' + str(ypos[0]) + ', y, ' + str(ypos[2]) + ')', fontsize=18)

        axs[2].set_xlabel('z [' + str(zpos[2].unit) + ']', fontsize=14)
        axs[2].set_title(self.func_names[0] + '(' + str(zpos[0]) + ', ' + str(zpos[1]) + ', z)', fontsize=18)

        for i, func in enumerate(self.funcs):
            axs[0].plot(xpos[0], func(xpos[0], xpos[1], xpos[2]), label=self.func_names[0])
            axs[1].plot(ypos[1], func(ypos[0], ypos[1], ypos[2]), label=self.func_names[0])
            axs[2].plot(zpos[2], func(zpos[0], zpos[1], zpos[2]), label=self.func_names[0])

        for ax in axs:
            ax.grid(True)
            ax.legend(loc='upper right', fontsize=14, fancybox=True, framealpha=0.9)

        #for i in range(3):
        #    ax2 = axs[i].twinx()  # Create a secondary y-axis for potential
        #    if i == 0:
        #        line1 = ax2.plot(self.x, self.V_trap_array[:, self.num_grid_points[1]//2, self.num_grid_points[2]//2], 'r--', label=r'$V_{trap}$')  
        #    elif i == 1:
        #        ax2.plot(self.y, self.V_trap_array[self.num_grid_points[0]//2, :, self.num_grid_points[2]//2], 'r--', label=r'$V_{trap}$')
        #    elif i == 2:
        #        ax2.plot(self.z, self.V_trap_array[self.num_grid_points[0]//2, self.num_grid_points[1]//2, :], 'r--', label=r'$V_{trap}$')
        #        
        #    ax2.set_ylabel(r'$V_{trap} \; \left[ nK \right]$', color='r', fontsize=14)  
        #    ax2.tick_params(axis='y', labelcolor='r')  
        #    axs[i].grid(True)
        #        
        #h, _ = axs[0].get_legend_handles_labels()  
        #labels = [r'$n_{total}$', r'$n_0$', r'$n_{ex}$', '$V_{trap}$']
        #fig.legend(h+line1, labels, loc='upper right', fontsize=14, fancybox=True, framealpha=0.9, bbox_to_anchor=(1, 1))  

        fig.tight_layout(rect=[0, 0, 0.95, 1])
        if filename != None:
            fig.savefig(filename, dpi=300, bbox_inches='tight')

        return axs
        
