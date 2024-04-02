import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
from ipywidgets import Dropdown, widgets, HBox, fixed
from IPython.display import display
from collections.abc import Sequence

def interactive_plot(data, x, y, z, title=None, cbar_title=None):
    x = x.to(u.um)
    y = y.to(u.um)
    z = z.to(u.um)

    def update_slider_options(*args):
        options = {
            'xy': [(f'{z[i]:.2f}', i) for i in range(len(z))],
            'xz': [(f'{y[i]:.2f}', i) for i in range(len(y))],
            'yz': [(f'{x[i]:.2f}', i) for i in range(len(x))]
        }
        descriptions = {
        'xy': 'z:',
        'xz': 'y:',
        'yz': 'x:'
        }
        midpoints = {
        'xy': len(z) // 2,
        'xz': len(y) // 2,
        'yz': len(x) // 2
        }
        
        index_slider.options = options[plane_selector.value]
        index_slider.value = midpoints[plane_selector.value]
        index_slider.description = descriptions[plane_selector.value]

    
    def plot_data(plane, index, plot_type, data, x, y, z):
        data_list = data if isinstance(data, list) else [data]
        num_plots = len(data_list)
        cols = 2  # Max number of columns if there are multiple plots
        rows = max((num_plots + cols - 1) // cols, 1)  # Calculate rows, ensure at least 1

        if plot_type == '3d':
            fig = plt.figure(figsize=(8*cols, 6*rows))
            for i, data_array in enumerate(data_list):
                ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                X, Y = np.meshgrid(x.value, y.value) if plane == 'xy' else (np.meshgrid(x.value, z.value) if plane == 'xz' else np.meshgrid(y.value, z.value))
                Z = data_array.value[:,:,index].T if plane == 'xy' else (data_array.value[:,index,:].T if plane == 'xz' else data_array.value[index,:,:].T)
                surf = ax.plot_surface(X, Y, Z, cmap='viridis')
                fig.colorbar(surf, ax=ax, shrink=0.75,).ax.set_title(cbar_title, fontsize=14, pad=10)
                ax.set_xlabel(f'{plane[0]} [um]', fontsize=14)
                ax.set_ylabel(f'{plane[1]} [um]', fontsize=14)
                if title:
                    ax.set_title(title[i] if not isinstance(title, str) else title, fontsize=20)
        elif plot_type == 'pcolormesh':
            if num_plots == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                axs = [[ax]]  # Ensure axs is array-like for consistency
            else:
                fig, axs = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), squeeze=False)
            
            for i, data_array in enumerate(data_list):
                ax = axs[i // cols, i % cols] if num_plots > 1 else axs[0][0]
                mesh = ax.pcolormesh(x.value, y.value, data_array.value[:,:,index].T, shading='auto') if plane == 'xy' \
                    else ax.pcolormesh(x.value, z.value, data_array.value[:,index,:].T, shading='auto') if plane == 'xz' \
                    else ax.pcolormesh(y.value, z.value, data_array.value[index,:,:].T, shading='auto')
                ax.set_xlabel(f'{plane[0]} [um]', fontsize=14)
                ax.set_ylabel(f'{plane[1]} [um]', fontsize=14)
                if title:
                    ax.set_title(title[i] if not isinstance(title, str) else title, fontsize=20)
                fig.colorbar(mesh, ax=ax).ax.set_title(cbar_title, fontsize=14, pad=10)

        plt.tight_layout()
        plt.show()


    plane_selector = Dropdown(options=['xy', 'xz', 'yz'], value='xy', description='Plane:')
    plot_type_selector = Dropdown(options=['pcolormesh', '3d'], value='pcolormesh', description='Plot Type:')
    index_slider = widgets.SelectionSlider(options=[(f'{z[i]}', i) for i in range(len(z))], value=len(z)//2, description='z:', continuous_update=False)
    
    plane_selector.observe(update_slider_options, 'value')
    ui = HBox([plane_selector, plot_type_selector, index_slider,])
    out = widgets.interactive_output(plot_data, {'plane': plane_selector, 'index': index_slider, 'plot_type': plot_type_selector, 'data': fixed(data), 'x': fixed(x), 'y': fixed(y), 'z': fixed(z)})

    display(ui, out)
