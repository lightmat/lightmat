# Data taken from:
# [1] https://journals.aps.org/pra/pdf/10.1103/PhysRevA.86.052517

# We are interested in the 4S_1/2, 4P_1/2 and 4P_3/2 states of potassium. For each of these states,
# there is a dictionary named data_4S_1_2, data_4P_1_2 and data_4P_3_2, respectively, where the keys
# are tuples (n, L, J) specifying the most important states, which have a dipole-allowed transition
# to the state of interest (note that always S=1/2 in alkali atoms, so we drop S from the state specifier). 
# The values of the dictionaries are dictionaries themselves with the keys 'E1', 'lambda' and 'Gamma'
# specifying the E1 matrix element, the transition wavelength and the transition linewidth, respectively. 

# The E1 matrix elements |<Psi||D||Psi_k>| for the states Psi equal to 4S_1/2, 4P_1/2 and 4P_3/2 
# are given in atomic units [a_0*e] with Bohr radius a0 = 5.29177210903e-11 m and 
# elementary charge e = 1.602176634e-19 C.

# The transition wavelengths lambda are in [nm].

# The linewidts of transitions k->i, Gamma_ki, are called transition probability A_ki 
# in [1] and are converted in [hbar x MHz].

data_4S_1_2 = {
    (4, 'P', 1/2): {'E1': 4.13120, 'lambda': 770.108, 'Gamma': 37.85,},
    (4, 'P', 3/2): {'E1': 5.84120, 'lambda': 766.699, 'Gamma': 38.34,},
    (5, 'P', 1/2): {'E1': 0.2826, 'lambda': 404.836, 'Gamma': 1.214,}, # Typo in [1] in table II, 5P_1/2 decays to 4S_1/2, not 4S_5/2
    (5, 'P', 3/2): {'E1': 0.4166 , 'lambda': 404.528, 'Gamma': 1.324,}, 
    (6, 'P', 1/2): {'E1': 0.0875, 'lambda': 344.836, 'Gamma': 0.1871,},
    (6, 'P', 3/2): {'E1': 0.1326, 'lambda': 344.736, 'Gamma': 0.2154,},
    (7, 'P', 1/2): {'E1': 0.0415, 'lambda': 321.855 , 'Gamma': 0.05108,},
    (7, 'P', 3/2): {'E1': 0.0645, 'lambda': 321.808, 'Gamma': 0.06187,},
    #(8, 'P', 1/2): {'E1':0.0233,}, # no lambda and Gamma in [1]
    #(8, 'P', 3/2): {'E1':0.0383,}, # no lambda and Gamma in [1]
    #(9, 'P', 1/2): {'E1':0.0163,}, # no lambda and Gamma in [1]
    #(9, 'P', 3/2): {'E1':0.0273,}, # no lambda and Gamma in [1]
}

data_4P_1_2 = {
    (3, 'D', 3/2): {'E1': 7.98840, 'lambda': 1169.344, 'Gamma': 20.21,},
    (4, 'S', 1/2): {'E1': 4.13120, 'lambda': 770.108, 'Gamma': 37.85,},
    (4, 'D', 3/2): {'E1': 0.2205, 'lambda': 693.820, 'Gamma': 0.07340,},
    (5, 'S', 1/2): {'E1': 3.87610, 'lambda': 1243.570, 'Gamma': 7.914,},
    (5, 'D', 3/2): {'E1': 0.2645, 'lambda': 581.376, 'Gamma': 0.1796,},
    (6, 'S', 1/2): {'E1': 0.90910, 'lambda': 691.299, 'Gamma': 2.534,},
    (6, 'D', 3/2): {'E1': 0.2935, 'lambda': 534.445, 'Gamma': 0.2849,},
    (7, 'S', 1/2): {'E1': 0.4795 , 'lambda': 578.400, 'Gamma': 1.201,},
    #(7, 'D', 3/2): {'E1': 0.2614,}, # no lambda and Gamma in [1]
    (8, 'S', 1/2): {'E1': 0.3165, 'lambda': 532.481, 'Gamma': 0.6700,}, # Typo in [1], should be lambda = 532.481nm, not 53.2481nm 
    #(8, 'D', 3/2): {'E1': 0.2214,}, # no lambda and Gamma in [1]
    #(9, 'S', 1/2): {'E1': 0.2253,}, # no lambda and Gamma in [1]
    #(10, 'S', 1/2): {'E1': 0.1713,}, # no lambda and Gamma in [1]
}

data_4P_3_2 = {
    (3, 'D', 3/2): {'E1': 3.58320, 'lambda': 1177.289 , 'Gamma': 0.3985,},
    (3, 'D', 5/2): {'E1': 10.74950, 'lambda': 1177.610, 'Gamma': 23.90,},
    (4, 'S', 1/2): {'E1': 5.84120, 'lambda': 766.699, 'Gamma': 38.34,},
    (4, 'D', 3/2): {'E1': 0.0885, 'lambda': 696.609, 'Gamma': 0.01160,},
    (4, 'D', 5/2): {'E1': 0.2605, 'lambda': 696.661, 'Gamma': 0.06751,},
    (5, 'S', 1/2): {'E1': 5.52420, 'lambda': 1252.558, 'Gamma': 15.73,},
    (5, 'D', 3/2): {'E1': 0.1245, 'lambda': 583.333, 'Gamma': 0.03924,},
    (5, 'D', 5/2): {'E1': 0.3745, 'lambda': 583.351, 'Gamma': 0.2379,},
    (6, 'S', 1/2): {'E1': 1.28710, 'lambda': 694.068, 'Gamma': 5.019,},
    (6, 'D', 3/2): {'E1': 0.1355 , 'lambda': 536.099, 'Gamma': 0.05991,}, # Typo in [1] in table II, 6D_3/2 decays to 4P_3/2, not 4D_3/2
    (6, 'D', 5/2): {'E1': 0.4045, 'lambda': 536.107, 'Gamma': 0.3577,},
    (7, 'S', 1/2): {'E1': 0.6776, 'lambda': 580.337, 'Gamma': 2.375,},
    #(7, 'D', 3/2): {'E1': 0.2614,}, # no lambda and Gamma in [1]
    #(7, 'D', 5/2): {'E1': 0.3565,}, # no lambda and Gamma in [1]
    (8, 'S', 1/2): {'E1': 0.4475, 'lambda': 534.117, 'Gamma': 1.328,},
    #(8, 'D', 3/2): {'E1': 0.2214,}, # no lambda and Gamma in [1]
    #(8, 'D', 5/2): {'E1': 0.2865,}, # no lambda and Gamma in [1]
    #(9, 'S', 1/2): {'E1': 0.3175,}, # no lambda and Gamma in [1]
    #(10, 'S', 1/2): {'E1': 0.2425,}, # no lambda and Gamma in [1]
}
