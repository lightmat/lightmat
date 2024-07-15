import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection
from ipywidgets import Dropdown, widgets, HBox, fixed, Layout
from IPython.display import display
from collections.abc import Sequence

def interactive_plot(data, x, y, z, title=None, cbar_title=None):
    x = x.to(u.um)
    y = y.to(u.um)
    z = z.to(u.um)

    def update_ui_elements(*args):
        # Adjust visibility and options based on plane selection
        plane = plane_selector.value
        if plane in ['x', 'y', 'z']:
            plot_type_selector.options = ['1d plot']
            index_slider2.layout.visibility = 'visible'
        else:
            plot_type_selector.options = ['pcolormesh', '3d']
            index_slider2.layout.visibility = 'hidden'   
        update_slider_options(manual_update=True)

    def update_slider_options(manual_update=False):
        # This function now has a manual_update flag to avoid unnecessary updates
        plane = plane_selector.value
        if manual_update:  # Only update options when necessary
            if plane in ['x', 'y', 'z']:
                options1 = {
                    'x': [(f'{y[i]:.2f}', i) for i in range(len(y))],
                    'y': [(f'{x[i]:.2f}', i) for i in range(len(x))],
                    'z': [(f'{x[i]:.2f}', i) for i in range(len(x))]
                }
                options2 = {
                    'x': [(f'{z[i]:.2f}', i) for i in range(len(z))],
                    'y': [(f'{z[i]:.2f}', i) for i in range(len(z))],
                    'z': [(f'{y[i]:.2f}', i) for i in range(len(y))]
                }
                descriptions1 = {
                    'x': 'y:',
                    'y': 'x:',
                    'z': 'x:'
                }
                descriptions2 = {
                    'x': 'z:',
                    'y': 'z:',
                    'z': 'y:'
                }
                midpoints1 = {
                    'x': len(y) // 2,
                    'y': len(x) // 2,
                    'z': len(x) // 2
                }
                midpoints2 = {
                    'x': len(z) // 2,
                    'y': len(z) // 2,
                    'z': len(y) // 2
                }
                index_slider1.options = options1[plane]
                index_slider2.options = options2[plane]
                index_slider1.description = descriptions1[plane]
                index_slider2.description = descriptions2[plane]
                index_slider1.value = midpoints1[plane]
                index_slider2.value = midpoints2[plane]
            else:
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
                index_slider1.options = options[plane_selector.value]
                index_slider1.value = midpoints[plane_selector.value]
                index_slider1.description = descriptions[plane_selector.value]

    
    def plot_data(plane, index1, index2, plot_type, data, x, y, z):
        data_list = data if isinstance(data, list) else [data]
        num_plots = len(data_list)
        cols = 2  # Max number of columns if there are multiple plots
        rows = max((num_plots + cols - 1) // cols, 1)  # Calculate rows, ensure at least 1

        if plot_type == '1d plot':
            if num_plots == 1:
                fig, ax = plt.subplots(figsize=(8, 6))
                axs = [[ax]]  # Ensure axs is array-like for consistency
            else:
                fig, axs = plt.subplots(rows, cols, figsize=(8*cols, 6*rows), squeeze=False)
                
            for i, data_array in enumerate(data_list):
                ax = axs[i // cols, i % cols] if num_plots > 1 else axs[0][0]
                plot = ax.plot(x.value, data_array.value[:,index1,index2].T) if plane =='x' \
                       else ax.plot(y.value, data_array.value[index1,:,index2].T) if plane == 'y' \
                       else ax.plot(z.value, data_array.value[index1,index2,:].T)
                ax.set_xlabel(f'{plane} [μm]', fontsize=14)
                ax.set_ylabel(cbar_title, fontsize=14)
                ax.grid(True)
                if title:
                    ax.set_title(title[i] if not isinstance(title, str) else title, fontsize=20)    
        else:
            if plot_type == '3d':
                fig = plt.figure(figsize=(8*cols, 6*rows))
                for i, data_array in enumerate(data_list):
                    ax = fig.add_subplot(rows, cols, i+1, projection='3d')
                    X, Y = np.meshgrid(x.value, y.value) if plane == 'xy' else (np.meshgrid(x.value, z.value) if plane == 'xz' else np.meshgrid(y.value, z.value))
                    Z = data_array.value[:,:,index1].T if plane == 'xy' else (data_array.value[:,index1,:].T if plane == 'xz' else data_array.value[index1,:,:].T)
                    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
                    fig.colorbar(surf, ax=ax, shrink=0.75,).ax.set_title(cbar_title, fontsize=14, pad=10)
                    ax.set_xlabel(f'{plane[0]} [μm]', fontsize=14)
                    ax.set_ylabel(f'{plane[1]} [μm]', fontsize=14)
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
                    mesh = ax.pcolormesh(x.value, y.value, data_array.value[:,:,index1].T, shading='auto') if plane == 'xy' \
                        else ax.pcolormesh(x.value, z.value, data_array.value[:,index1,:].T, shading='auto') if plane == 'xz' \
                        else ax.pcolormesh(y.value, z.value, data_array.value[index1,:,:].T, shading='auto')
                    ax.set_xlabel(f'{plane[0]} [μm]', fontsize=14)
                    ax.set_ylabel(f'{plane[1]} [μm]', fontsize=14)
                    if title:
                        ax.set_title(title[i] if not isinstance(title, str) else title, fontsize=20)
                    fig.colorbar(mesh, ax=ax).ax.set_title(cbar_title, fontsize=14, pad=10)

            plt.tight_layout()
            plt.show()

    plane_selector = Dropdown(options=['xy', 'xz', 'yz', 'x', 'y', 'z'], description='Plane:')
    plot_type_selector = Dropdown(options=['pcolormesh', '3d'], description='Plot Type:')
    index_slider1 = widgets.SelectionSlider(options=[('Select a value', 0)], value=0, description='Index 1:', continuous_update=False)
    index_slider2 = widgets.SelectionSlider(options=[('Select a value', 0)], value=0, description='Index 2:', continuous_update=False, layout=Layout(visibility='hidden'))

    # Adjustments to observer setup to streamline updates
    plane_selector.observe(update_ui_elements, 'value')
    plot_type_selector.observe(lambda change: update_slider_options(manual_update=True), 'value')

    controls = HBox([plane_selector, plot_type_selector, index_slider1, index_slider2])
    out = widgets.interactive_output(plot_data, {'plane': plane_selector, 'index1': index_slider1, 'index2': index_slider2, 'plot_type': plot_type_selector, 'data': fixed(data), \
                                                 'x': fixed(x), 'y': fixed(y), 'z': fixed(z)})

    display(controls, out)
    update_ui_elements()  # Initial UI update