class CellStandards():

    sorted_cell_parameters = ['a', 'b/a', 'c/a', 'alpha', 'beta', 'gamma']


class VaspStandards():
    # Parameters to that will be tracked in parameterized model
    sorted_calc_parameters = ['xc', 'encut', 'nbands', 'ispin', 'kspacing',
                              'kgamma', 'ismear', 'sigma', 'ibrion', 'isif',
                              'nsw', 'nelm', 'ediff', 'ediffg', 'prec',
                              'algo', 'lwave', 'lorbit']

    u_parameters = ['ldau', 'lmaxmix', 'ldautype', 'ldau_luj']

    """Parameters are heavily inspired by MP standard settings at
    https://github.com/materialsproject/pymatgen/blob/master/pymatgen/
    io/vasp/MPRelaxSet.yaml
    """

    # is kspacing = 0.25 compatible with materials project
    # reciprocal_density = 64? (1 / 64) ** (1/3) = 0.25  ?
    calc_parameters = {'xc': 'pbe',
                       'encut': 520,  # energy cutoff for plane waves
                       'nbands': -5,  # number of bands / empty bands
                       'ispin': 2,  # number of spins
                       'kspacing': 0.25,  # kspacing
                       'kgamma': True,  # include gamma point
                       'ismear': -5,  # smearing function
                       'sigma': 0.05,  # k-point smearing
                       'ibrion': 2,  # ion dynamics
                       'isif': 3,  # degrees of freedom to relax
                       'nsw': 99,  # maximum number of ionic steps
                       'nelm': 100,  # maximum number of electronic steps
                       'ediff': 1e-5,  # sc accuracy in units of 1e-6
                       'ediffg': -0.02,  # force convergence accuracy
                       'prec': 'Accurate',  # Precision
                       'algo': 'Fast',  # optimization algorithm
                       'lwave': False,  # save wavefunctions or not
                       'ldau': True,  # USE U
                       'lmaxmix': 4,
                       'ldautype': 2,
                       'ldau_luj': {},
                       'lorbit': 11,
                       }
    molecule_calc_parameters = {'kspacing': None,
                                'ismear': 0,
                                'sigma': 1}


    paw_potentials = {'Li': '_sv',
                      'Na': '_pv',
                      'K': '_sv',
                      'Ca': '_sv',
                      'Sc': '_sv',
                      'Ti': '_sv',
                      'V': '_sv',
                      'Cr': '_pv',
                      'Mn': '_pv',
                      'Ga': '_d',
                      'Ge': '_d',
                      'Rb': '_sv',
                      'Sr': '_sv',
                      'Y': '_sv',
                      'Zr': '_sv',
                      'Nb': '_sv',
                      'Mo': '_sv',
                      'Tc': '_pv',
                      'Ru': '_pv',
                      'Rh': '_pv',
                      'In': '_d',
                      'Sn': '_d',
                      'Cs': '_sv',
                      'Ba': '_sv',
                      'Pr': '_3',
                      'Nd': '_3',
                      'Pm': '_3',
                      'Sm': '_3',
                      'Eu': '_2',
                      'Gd': '_3',
                      'Tb': '_3',
                      'Dy': '_3',
                      'Ho': '_3',
                      'Er': '_3',
                      'Tm': '_3',
                      'Yb': '_2',
                      'Lu': '_3',
                      'Hf': '_pv',
                      'Ta': '_pv',
                      'W': '_sv',
                      'Tl': '_d',
                      'Pb': '_d',
                      'Bi': '_d',
                      'Po': '_d',
                      'At': '_d',
                      'Fr': '_sv',
                      'Ra': '_sv'}


class EspressoStandards():
    # Espresso parameters to that will be tracked in parameterized model
    sorted_calc_parameters = ['xc', 'encut', 'nbands', 'ispin', 'kspacing',
                              'kgamma', 'ismear', 'sigma', 'ibrion', 'isif',
                              'nsw', 'nelm', 'ediff', 'prec', 'algo', 'lwave',
                              'ldau', 'ldautype']


class GPAWStandards():
    "TODO"
    sorted_calc_parameters = []


class CommonCalc():
    """+U values"""
    U_trickers = ['O', 'F']  # Oxides and Flourides will have +U
    U_luj = {'Au': {'L': -1, 'U': 0.0, 'J': 0.0},
             'C':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'Cu': {'L': -1, 'U': 0.0, 'J': 0.0},
             'H':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'Ir': {'L': -1, 'U': 0.0, 'J': 0.0},
             'O':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'F':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'Co': {'L': 2, 'U': 3.32, 'J': 0.0},
             'Cr': {'L': 2, 'U': 3.7, 'J': 0.0},  # Meng U: 3.5
             'Fe': {'L': 2, 'U': 5.3, 'J': 0.0},  # 'U': 4.3
             'Mn': {'L': 2, 'U': 3.9, 'J': 0.0},  # 'U': 3.75
             'Mo': {'L': 2, 'U': 4.38, 'J': 0.0},
             'Nb': {'L': 2, 'U': 4.00, 'J': 0.0},
             'Ni': {'L': 2, 'U': 6.2, 'J': 0.0},  # 'U': 6.45
             'Sn': {'L': 2, 'U': 3.5, 'J': 0.0},
             'Ta': {'L': 2, 'U': 4.00, 'J': 0.0},
             'Ti': {'L': 2, 'U': 3.00, 'J': 0.0},
             'V':  {'L': 2, 'U': 3.25, 'J': 0.0},
             'W':  {'L': 2, 'U': 6.2, 'J': 0.0},  # 'U': 2.0
             'Zr': {'L': 2, 'U': 4.00, 'J': 0.0},
             'Ce': {'L': 3, 'U': 4.50, 'J': 0.0}}

    U_metals = list(U_luj.keys())
    for U in U_trickers:
        U_metals.remove(U)

    initial_magmoms = {'Ce': 5,
                       'Co': 5,
                       'Cr': 5,
                       'Fe': 5,
                       'Mn': 5,
                       'Mo': 5,
                       'Ni': 5,
                       'V': 5,
                       'W': 5}

    magnetic_trickers = list(initial_magmoms.keys())


class CrystalStandards():

    """Reference structures for formation energies, taken from 
    Materials Project.
    Most metals falls in the standard crystal structures:
    hcp: 194
    fcc: 225
    bcc: 229
    """

    standard_lattice_mp = {
        'Li': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Li']},
        'B': {'p_name': 'A_12_h2_166',
              'spacegroup': 166,
              'species': ['B', 'B'],
              'wyckoffs': ['h', 'h'],
              'parameters': {'a': 4.899977315543144,
                             'c': 2.561437368329688,
                             'xh0': 0.4697760000000001,
                             'xh1': 0.4520915000000001,
                             'zh0': 0.309094,
                             'zh1': 0.558217}},
        'C': {'p_name': 'A_4_n_67',
              'parameters': {'a': 4.274524,
                             'b': 0.5772252068300471,
                             'c': 1.8781805880607991,
                             'xn0': 0.833421,
                             'zn0': 0.239942},
              'spacegroup': 67,
              'species': ['C'],
              'wyckoffs': ['n']},
        'Be': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Be']
               },
        'Na': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Na']
               },
        'Mg': {'p_name': 'A_3_ac_166',
               'spacegroup': 166,
               'wyckoffs': ['a', 'c'],
               'species': ['Mg', 'Mg'],
               'parameters': {'a': 3.2112466672290045,
                              'c/a': 7.139511029771984,
                              'zc1': 0.2222079999999999}},
        'Al': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Al']},
        'P': {'p_name': 'A_42_i21_2',
              'parameters': {'a': 7.85748,
                             'alpha': 102.66942804526694,
                             'b': 1.545925181580378,
                             'beta': 106.71627094942292,
                             'c': 1.660673102847688,
                             'gamma': 101.97555175777389,
                             'xi0': 0.02345500000000009,
                             'xi1': 0.032943000000000076,
                             'xi10': 0.26081999999999994,
                             'xi11': 0.27566500000000005,
                             'xi12': 0.276797,
                             'xi13': 0.332918,
                             'xi14': 0.38568,
                             'xi15': 0.414176,
                             'xi16': 0.448101,
                             'xi17': 0.47558399999999995,
                             'xi18': 0.48316,
                             'xi19': 0.48444099999999995,
                             'xi2': 0.055936000000000013,
                             'xi20': 0.48625599999999997,
                             'xi3': 0.10721600000000003,
                             'xi4': 0.11043500000000003,
                             'xi5': 0.14861900000000003,
                             'xi6': 0.19380900000000006,
                             'xi7': 0.220332,
                             'xi8': 0.229521,
                             'xi9': 0.23596800000000007,
                             'yi0': 0.41404099999999994,
                             'yi1': 0.5187029999999999,
                             'yi10': 0.8482979999999998,
                             'yi11': 0.6664039999999999,
                             'yi12': 0.7666849999999997,
                             'yi13': 0.22176299999999996,
                             'yi14': 0.6645619999999999,
                             'yi15': 0.08107100000000002,
                             'yi16': 0.5196339999999999,
                             'yi17': 0.3914249999999999,
                             'yi18': 0.019708000000000035,
                             'yi19': 0.5448409999999999,
                             'yi2': 0.9511799999999998,
                             'yi20': 0.075914,
                             'yi3': 0.8243919999999998,
                             'yi4': -0.005952000000000179,
                             'yi5': 0.42052799999999996,
                             'yi6': 0.266959,
                             'yi7': 0.7281909999999999,
                             'yi8': 0.12748999999999996,
                             'yi9': 0.320337,
                             'zi0': 0.8476719999999999,
                             'zi1': 0.7284039999999999,
                             'zi10': 0.17348699999999997,
                             'zi11': 0.8489649999999999,
                             'zi12': 0.7276329999999999,
                             'zi13': 0.3596289999999999,
                             'zi14': 0.6031289999999999,
                             'zi15': 0.42785299999999993,
                             'zi16': 0.6687049999999999,
                             'zi17': 0.15018399999999998,
                             'zi18': 0.7022179999999999,
                             'zi19': 0.08088699999999999,
                             'zi2': 0.17941799999999997,
                             'zi20': 0.17109899999999997,
                             'zi3': 0.39957699999999996,
                             'zi4': 0.36389799999999994,
                             'zi5': 0.6075249999999999,
                             'zi6': 0.6662909999999999,
                             'zi7': 0.27450699999999995,
                             'zi8': 0.17614099999999996,
                             'zi9': 0.8475539999999999},
              'spacegroup': 2,
              'species': ['P'] * 21,
              'wyckoffs': ['i'] * 21},
        'K':  {'p_name': 'A_20_cd_213',
               'spacegroup': 213,
               'wyckoffs': ['c', 'd'],
               'species': ['K', 'K'],
               'parameters': {'a': 11.435131,
                              'xc0': 0.062133,
                              'yd1': 0.202742}},
        'Rb': {'p_name': 'A_29_acg2_217',
               'spacegroup': 217,
               'wyckoffs': ['a', 'c', 'g', 'g'],
               'species': ['Rb', 'Rb', 'Rb', 'Rb'],
               'parameters': {'a': 17.338553,
                              'xc1': 0.817731,
                              'xg2': 0.639646,
                              'zg2': 0.042192,
                              'xg3': 0.09156399999999998,
                              'zg3': 0.2818120000000004}},
        'Ca': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ca']},
        'Sr': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Sr']},
        'Sc': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Sc']},
        'Y': {'p_name': 'A_2_c_194',
              'spacegroup': 194,
              'wyckoffs':  ['c'],
              'species': ['Y']},
        'V': {'p_name': 'A_2_c_194',
              'spacegroup': 194,
              'wyckoffs':  ['c'],
              'species': ['V']},
        'La': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['La']},
        'Ti': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Ti']},
        'Zr': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Zr']},
        'Hf': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Hf']},
        'V': {'p_name': 'A_1_a_229',
              'spacegroup': 229,
              'wyckoffs': ['a'],
              'species': ['V']},
        'Nb': {'p_name': 'A_1_a_166',
               'spacegroup': 166,
               'wyckoffs': ['a'],
               'species': ['Nb']},
        'Ta': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Ta']},
        'Cr': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Cr']},
        'Mo': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Mo']},
        'W': {'p_name': 'A_1_a_229',
              'spacegroup': 229,
              'wyckoffs': ['a'],
              'species': ['W']},
        'Mn': {'p_name': 'A_29_acg2_217',  # magnetic
               'spacegroup': 217,
               'wyckoffs': ['a', 'c', 'g', 'g'],
               'species': ['Mn', 'Mn', 'Mn', 'Mn'],
               'parameters': {'a': 8.618498,
                              'xc1': 0.818787,
                              'xg2': 0.643796,
                              'zg2': 0.035468,
                              'xg3': 0.9109409999999998,
                              'zg3': 0.282544}},
        'Fe': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Fe']},  # magnetic
        'Tc': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Tc']},
        'Re': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Re']},
        'Ru': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Ru']},
        'Os': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Os']},
        'Co': {'p_name': 'A_2_c_194',  # magnetic
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Co']},
        'Rh': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Rh']},
        'Ir': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ir']},
        'Ni': {'p_name': 'A_1_a_225',  # magnetic
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ni']},
        'Pd': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Pd']},
        'Pt': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Pt']},
        'Cu': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Cu']},
        'Ag': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ag']},
        'Au': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Au']},
        'Zn': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Zn']},
        'Cd': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Cd']},
        'Hg': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Hg']},
        'Ga': {'p_name': 'A_4_f_64',
               'spacegroup': 64,
               'wyckoffs': ['f'],
               'species': ['Ga']},
        'In': {'p_name': 'A_3_ac_166',
               'spacegroup': 166,
               'wyckoffs': ['a', 'c'],
               'species': ['In', 'In'],
               'parameters': {'a': 3.3328619310176943,
                              'c/a': 7.623639840442314,
                              'zc1': 0.22166400000000003}},
        'Tl': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Tl']
               },
        'Si': {'p_name': 'A_2_a_227',
               'spacegroup': 227,
               'wyckoffs': ['a'],
               'species': ['Si']},
        'Ge': {'p_name': 'A_2_a_227',
               'spacegroup': 227,
               'wyckoffs': ['a'],
               'species': ['Ge']},
        'Sn': {'p_name': 'A_2_a_227',
               'spacegroup': 227,
               'wyckoffs': ['a'],
               'species': ['Sn']},
        'Pb': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Pb']},
        'As': {'p_name': 'A_2_c_166',
               'spacegroup': 166,
               'wyckoffs': ['c'],
               'species': ['As']},
        'Sb': {'p_name': 'A_2_c_166',
               'spacegroup': 166,
               'wyckoffs': ['c'],
               'species': ['Sb']},
        'Bi': {'p_name': 'A_2_c_166',
               'spacegroup': 166,
               'wyckoffs': ['c'],
               'species': ['Bi']},
        'Se': {'p_name': 'A_64_e16_14', 'spacegroup': 14,
               'wyckoffs': ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e',
                            'e', 'e', 'e', 'e', 'e', 'e'],
               'species': ['Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se',
                           'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se'],
               'parameters': {'a': 15.739181,
                              'b/a': 0.9726696706772735,
                              'c/a': 0.6451328742335684,
                              'beta': 94.0428681461934,
                              'xe0': 0.986572,
                              'ye0': 0.593959,
                              'ze0': 0.23545999999999978,
                              'xe1': 0.984712,
                              'ye1': 0.214049,
                              'ze1': 0.12086899999999978,
                              'xe2': 0.913585,
                              'ye2': 0.675372,
                              'ze2': 0.828421,
                              'xe3': 0.906751,
                              'ye3': 0.009713,
                              'ze3': 0.847639,
                              'xe4': 0.814616,
                              'ye4': 0.142437,
                              'ze4': 0.4519159999999999,
                              'xe5': 0.806848,
                              'ye5': 0.690537,
                              'ze5': 0.978908,
                              'xe6': 0.769548,
                              'ye6': 0.047426,
                              'ze6': 0.27722899999999984,
                              'xe7': 0.76865,
                              'ye7': 0.215177,
                              'ze7': 0.888981,
                              'xe8': 0.761848,
                              'ye8': 0.508143,
                              'ze8': 0.2745009999999999,
                              'xe9': 0.695855,
                              'ye9': 0.589422,
                              'ze9': 0.43704999999999994,
                              'xe10': 0.6914370000000001,
                              'ye10': 0.736288,
                              'ze10': 0.36462799999999984,
                              'xe11': 0.646212,
                              'ye11': 0.524822,
                              'ze11': 0.842267,
                              'xe12': 0.644476,
                              'ye12': 0.185345,
                              'ze12': -0.000248000000000026,
                              'xe13': 0.52639,
                              'ye13': 0.213948,
                              'ze13': 0.8495199999999999,
                              'xe14': 0.525862,
                              'ye14': 0.645501,
                              'ze14': 0.10287099999999982,
                              'xe15': 0.524356,
                              'ye15': 0.04622,
                              'ze15': 0.24163899999999983}},
        'Te': {'p_name': 'A_3_a_152',
               'spacegroup': 152,
               'wyckoffs': ['a'],
               'species': ['Te'],
               'parameters': {'a': 4.5123742098481765,
                              'c/a': 1.3207900592536466,
                              'xa0': 0.26895}}
    }
