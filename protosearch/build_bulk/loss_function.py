import numpy as np
import scipy

from ase.data import covalent_radii as cradii
from pymatgen.io.ase import AseAtomsAdaptor
from catkit.gen.utils.connectivity import get_cutoff_neighbors
from pymatgen.analysis.ewald import EwaldSummation

from protosearch.utils.data import metal_numbers, prefered_O_state, \
     electronegs


fixed_oxi_states = {'O': -2,
                    'S': -2,
                    'N': -3,
                    'F': -1,
                    'Cl': -1}


def get_covalent_density(atoms):

    covalent_radii = np.array([cradii[n] for n in atoms.numbers])
    covalent_volume = np.sum(4/3 * np.pi * covalent_radii ** 3)
    cell_volume = atoms.get_volume()
    density = covalent_volume / cell_volume

    return density


def get_loss(atoms):
    N_metal = len([a for a in atoms if a.number in metal_numbers])
    symbols = atoms.get_chemical_symbols()

    if 'O' in symbols or 'N' in symbols or 'F' in symbols:
        loss = get_ewald_energy(atoms)

    else:
        loss = atoms.get_volume()

    return loss


def get_oxidation_states(atoms, charge_O=-2, charge_N=-3, charge_F=-1,
                         charge_H=1, use_connectivity=False):
    """Connectivity based oxidation state"""
    O_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'O'])
    N_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'N'])
    F_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'F'])
    H_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'H'])
    M_indices = np.array([i for i, a in enumerate(atoms)
                          if not a.symbol in ['O', 'H', 'N', 'F']])

    non_O_indices = np.array([i for i, a in enumerate(atoms)
                              if not a.symbol in ['O']])

    oxi_states = np.zeros([len(atoms)])
    if len(O_indices) > 0:
        oxi_states[O_indices] = charge_O
        oxi_states[M_indices] += - charge_O * len(O_indices) / len(M_indices)
    if len(N_indices) > 0:
        oxi_states[N_indices] = charge_N
        oxi_states[M_indices] += - charge_N * len(N_indices) / len(M_indices)
    if len(F_indices) > 0:
        oxi_states[F_indices] = charge_F
        oxi_states[M_indices] += - charge_F * len(F_indices) / len(M_indices)

    if not use_connectivity:
        return oxi_states

    anion_indices = np.append(
        np.append(O_indices, N_indices), F_indices).astype(int)

    con_matrix = get_cutoff_neighbors(atoms)
    M_O_Connectivity = con_matrix[M_indices, :][:, anion_indices]

    if H_indices:  # First correct O charge due to H
        oxi_states[H_indices] = charge_H
        for H_i in H_indices:
            H_O_connectivity = con_matrix[H_i][O_indices]
            norm = np.sum(H_O_connectivity)
            O_indices_H = O_indices[np.where(H_O_connectivity)[0]]
            oxi_states[O_indices_H] += charge_H / norm

    for metal_i in M_indices:  # Substract O connectivity
        M_O_connectivity = con_matrix[metal_i][anion_indices]
        norm = np.sum(con_matrix[anion_indices][:, M_indices], axis=-1)
        idx = np.where(norm > 0)[0]
        oxi_states[metal_i] = sum(
            M_O_connectivity[idx] * -oxi_states[anion_indices][idx] / norm[idx])

    if not sum(oxi_states) == 0:
        charge = sum(oxi_states)
        oxi_states[M_indices] += -charge/len(M_indices)
    return oxi_states


def get_ewald_energy(atoms):
    oxi_states = get_optimal_oxidation_states_for_composition(atoms)

    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.add_oxidation_state_by_site(oxi_states)
    e = EwaldSummation(structure).total_energy

    return e / len(atoms)


def get_oxy_loss(atoms):
    a = 1


# ametal_symbols, n_O):
def get_optimal_oxidation_states_for_composition(atoms):
    """
    A_nB_mO_k with oxidation states O_A and O_B setting O_O = 2
    Solve for integers O_A and O_B
    n_A * O_A + n_B * O_B = n_O * 2
    """
    metal_symbols = np.array([a.symbol for a in atoms
                              if not a.symbol in ['O', 'H', 'N', 'F']])

    n_M = len(metal_symbols)
    avg_oxi_state = 0  # n_O / n_M * a_charge

    n_O = 0
    n_N = 0
    n_F = 0

    oxi_states_atoms = np.zeros([len(atoms)])
    if 'O' in atoms.symbols:
        n_O = len([a for a in atoms if a.symbol == 'O'])
        a_charge = 2
        avg_oxi_state += n_O / n_M * a_charge
        A_indices = np.array([i for i, a in enumerate(atoms)
                              if a.symbol == 'O'])

        oxi_states_atoms[A_indices] = -a_charge
    if 'N' in atoms.symbols:
        n_N = len([a for a in atoms if a.symbol == 'N'])
        a_charge = 3
        avg_oxi_state += n_N / n_M * a_charge
        A_indices = np.array([i for i, a in enumerate(atoms)
                              if a.symbol == 'N'])

        oxi_states_atoms[A_indices] = -a_charge
    if 'F' in atoms.symbols:
        n_F = len([a for a in atoms if a.symbol == 'F'])
        a_charge = 1
        avg_oxi_state += n_F / n_M * a_charge
        A_indices = np.array([i for i, a in enumerate(atoms)
                              if a.symbol == 'F'])

        oxi_states_atoms[A_indices] = -a_charge

    oxi_states_dict = {}
    metal_symbols, counts = np.unique(metal_symbols, return_counts=True)

    electroneg = [electronegs.get(m, 0) for m in metal_symbols]
    indices = np.argsort(electroneg)[::-1]
    metal_symbols = metal_symbols[indices]

    pref_O_states = [prefered_O_state[m] for m in metal_symbols]

    if len(metal_symbols) == 2:
        # Make a guess for favorable oxidation states
        oxy_matches = []
        n_A, n_B = counts
        for o_A in range(1, n_O * a_charge // n_A):
            o_B = (a_charge * n_O - n_A * o_A) / n_B
            if o_B % 1 == 0:
                oxy_matches += [[o_A, int(o_B)]]

        oxy_matches = np.array(oxy_matches)
        same_state_idx = np.where(np.std(oxy_matches, axis=1) == 0)
        if len(same_state_idx) > 0:
            oxy_state_list = oxy_matches[same_state_idx[0][0]]
        else:
            pref_O_M = np.repeat(np.expand_dims(pref_O_states, 0),
                                 len(oxy_matches[:, 0]), 0)

            Oxy_fitness = np.sum(np.abs(oxy_matches - pref_O_M), axis=1)

            best_fit = np.argmin(Oxy_fitness)

            oxy_state_list = oxy_matches[best_fit]

    elif len(metal_symbols) == 1:  # only one type of metal
        oxy_state_list = [avg_oxi_state]

    oxi_states = {}
    for i, m in enumerate(metal_symbols):
        oxi_states.update({m: int(oxy_state_list[i])})

    for k, o in oxi_states.items():
        M_indices = np.array([i for i, a in enumerate(atoms)
                              if a.symbol == k])
        oxi_states_atoms[M_indices] = o

    return oxi_states_atoms


