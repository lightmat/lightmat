import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u
from typing import Union, Sequence

from .bec import BEC


def harmonic_trap(
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        trap_depth: float = 200,
        waist: float = 50,
        inhomogenity: float = 0.01
    ) -> Union[float, np.ndarray]:
    """Return a harmonic trap potential in [k_B x nK] for given position(s) in [μm].
    
        Args:
            x, y, z: position in [μm]
            trap_depth: trap depth in [k_B x nK]. Defaults to 200nK.
            waist: region in which we can model the trap as harmonic is [-waist, waist] in [μm]. Defaults to 50μm.
            inhomogenity: This number times `trap_depth` gives the noise magnitude.

        Returns:
            V(x, y, z): harmonic trap potential in [k_B x nK], array of same shape as x, y, z
    """
    # Create Gaussian profile
    r_squared = x**2 + y**2 + z**2
    gaussian_profile = np.exp(-2 * r_squared / waist**2)

    # Apply the potential: -trap_depth at the center, 0 at the edges
    perfect_harmonic_trap = -trap_depth * gaussian_profile

    # Add noise
    noise = np.random.rand(*perfect_harmonic_trap.shape) * inhomogenity * trap_depth
    return perfect_harmonic_trap + noise


def box_trap(
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        trap_depth: float = 200.,
        box_width: Union[float, Sequence[float]] = (100., 100., 100.),
        inhomogenity: float = 0.01
) -> Union[float, np.ndarray]:
    """Return a perfect box potential of depth `trap_depth` in [k_B x nK] for given position(s) in [μm].
       The size of the box in each direction around 0 is given by `box_width` in [μm].
    
        Args:
            x, y, z: position in [μm]
            trap_depth: box depth in [k_B x nK]. Defaults to 200nK.
            box_width: region [-box_width/2 μm, box_width/2 μm] is -`trap_depth`nK, else potential is 0nK.
                       Defaults to (100, 100, 100)μm.
            inhomogenity: This number times `trap_depth` gives the noise magnitude.

        Returns:
            V(x, y, z): box potential in [k_B x nK], array of same shape as x, y, z
    """
    if isinstance(box_width, (int, float)):
        box_width = (box_width, box_width, box_width)
    # Check if the positions are within the box
    in_box = (np.abs(x) <= box_width[0] / 2) & \
             (np.abs(y) <= box_width[1] / 2) & \
             (np.abs(z) <= box_width[2] / 2)

    # Apply the potential: -trap_depth inside the box, 0 outside
    perfect_box = np.where(in_box, -trap_depth, 0.)

    # Add noise
    noise = np.random.rand(*perfect_box.shape) * inhomogenity * trap_depth
    return perfect_box + noise


def box_2d_harmonic_1d_trap(
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray],
        trap_depth: float = 200.,
        box_width: Sequence[float] = (100., 100.),
        waist: float = 50,
        inhomogenity: float = 0.01
) -> Union[float, np.ndarray]:
    """Return a potential of depth `trap_depth` in [k_B x nK] for given position(s) in [μm].
       The potential is a box in the x-y plane and harmonic in the z direction.
    
        Args:
            x, y, z: position in [μm]
            trap_depth: box depth in [k_B x nK]. Defaults to 200nK.
            box_width: region [-box_width/2 μm, box_width/2 μm] is -`trap_depth`nK, else potential is 0nK.
                       Defaults to (100, 100)μm.
            waist: region in which we can model the trap as harmonic is [-waist, waist] in [μm]. Defaults to 50μm.
            inhomogenity: This number times `trap_depth` gives the noise magnitude.

        Returns:
            V(x, y, z): box potential in [k_B x nK], array of same shape as x, y, z
    """
    # Box in x-y plane
    in_box = (np.abs(x) <= box_width[0] / 2) & \
             (np.abs(y) <= box_width[1] / 2)
    perfect_box = np.where(in_box, 1, 0.)

    # Gaussian profile in z direction
    gaussian_profile = np.exp(-2 * z**2 / waist**2)

    # Combine the two
    perfect_trap = - trap_depth * perfect_box * gaussian_profile

    # Add noise
    noise = np.random.rand(*perfect_trap.shape) * inhomogenity * trap_depth
    return perfect_trap + noise



def analyze_bec(Ts, particle_props, mu_change_rate=0.01, init_with_zero_T=False):
    becs = []
    mu = None
    for T in Ts:
        pp = particle_props.copy()
        pp.T = T
        bec = BEC(pp, init_with_zero_T=init_with_zero_T)
        if mu is not None:
            bec.mu = mu # initialize mu with previous converged value
        bec.eval_density(mu_change_rate=mu_change_rate)
        mu = bec.mu
        becs.append(bec)

    return becs


def plot_condens_frac(Ts, becs):
    # Naive formula for condensate fraction (in case of non-interacting bosons)
    def condens_frac(T, T_c, alpha):
        return 1 - (T / T_c)**alpha

    initial_guess_fit_params = [95, 3]
    condens_fracs = [bec.condensate_fraction for bec in becs]
    popt, pcov = curve_fit(condens_frac, Ts.value, condens_fracs, p0=initial_guess_fit_params)
    perr = np.sqrt(np.diag(pcov))
    x_fit = np.linspace(Ts[0].value, Ts[-1].value, 1000)
    y_fit = condens_frac(x_fit, *popt)

    fig, axs = plt.subplots(1, 1, figsize=(9, 7))
    fig.suptitle('Condensate fraction', fontsize=24)
    axs.scatter(Ts, condens_fracs, c='k', marker='o', label='Data')
    axs.plot(x_fit, y_fit, c='k', linestyle='--', label=r'Fit [$T_c=$%.3fnK $\pm$ %.3fnK, $\alpha=$%.3f $\pm$ %.3f]'\
                                                        % (popt[0], perr[0], popt[1], perr[1]))
    axs.set_xlabel(r'$T \; \left[ nK \right]$', fontsize=14)
    axs.set_ylabel(r'Condensate fraction', fontsize=14)
    axs.set_title(r'$N_{total}=200000$, harmonic trap ($d=200nK$, $\omega=(50, 50, 50) \mu m$)', fontsize=16)
    axs.legend(fontsize=10)
    axs.grid(True)
    t = axs.text(0.1, 0.1, r'$\frac{N_{condensed}}{N_{total}} = 1 - \left( \frac{T}{T_c} \right)^{\alpha}$', transform=plt.gca().transAxes,
            fontsize=24, color='black', bbox=dict(facecolor='white', alpha=0.8))