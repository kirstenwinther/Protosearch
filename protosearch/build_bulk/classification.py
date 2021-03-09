import sys
import string
import numpy as np
from spglib import get_symmetry_dataset, standardize_cell, delaunay_reduce, niggli_reduce
import ase
from ase import Atoms
from ase.spacegroup import Spacegroup
from ase.build import cut

from .wyckoff_symmetries import WyckoffSymmetries, wrap_coordinate

from protosearch import build_bulk
path = build_bulk.__path__[0]

wyckoff_data = path + '/Wyckoff.dat'
wyckoff_pairs = path + '/Symmetric_Wyckoff_Pairs.dat'


class PrototypeClassification():  # WyckoffSymmetries):
    """Prototype classification of atomic structure in ASE Atoms format"""

    def __init__(self, atoms, tolerance=1e-3):

        self.tolerance = tolerance

        atoms = self.get_conventional_atoms(atoms)

        self.set_sorted_wyckoff_species(atoms)
        self.spacegroup = self.spglibdata['number']

    def get_conventional_atoms(self, atoms):
        """Set conventional cell and wyckoff positions"""

        lattice, scaled_positions, numbers = \
            standardize_cell(atoms,
                             to_primitive=False,
                             no_idealize=False,
                             symprec=self.tolerance)

        transform = lattice*np.linalg.inv(delaunay_reduce(lattice))

        atoms = Atoms(numbers=numbers,
                      cell=lattice,
                      pbc=True)

        atoms.set_scaled_positions(scaled_positions)

        self.set_sorted_wyckoff_species(atoms)
        atoms_list = [atoms.copy()]
        wyckoffs_list = [''.join(self.ordered_wyckoffs)]

        # Change origin of unit cell
        for idx in range(len(atoms)):
            atoms.positions -= np.repeat(
                atoms.positions[idx:idx+1], len(atoms), axis=0)
            atoms.positions *= -1
            atoms.wrap()
            atoms_list += [atoms.copy()]
            self.set_sorted_wyckoff_species(atoms)
            wyckoffs_list += [''.join(self.ordered_wyckoffs)]

        wyckoff_map_idx = np.argsort(wyckoffs_list)[0]
        atoms = atoms_list[wyckoff_map_idx]

        return atoms

    def set_sorted_wyckoff_species(self, atoms):

        self.spglibdata = get_symmetry_dataset((atoms.get_cell(),
                                                atoms.get_scaled_positions(),
                                                atoms.get_atomic_numbers()),
                                               symprec=self.tolerance)
        numbers = self.spglibdata['std_types']

        unique_numbers, counts = np.unique(numbers, return_counts=True)
        idx = np.argsort(counts)
        counts = counts[idx]
        unique_numbers = unique_numbers[idx]

        wyckoffs = self.spglibdata['wyckoffs']
        equivalent_atoms = self.spglibdata['equivalent_atoms']
        wyckoff_list = []
        species_list = list(Atoms(unique_numbers).symbols)
        self.wyckoff_multiplicities = {}
        for n in unique_numbers:
            wyckoff_n = []
            unique_eq, indices_eq = np.unique(
                equivalent_atoms, return_index=True)
            idx0 = np.where(numbers[unique_eq] == n)[0]
            idx = indices_eq[idx0]

            for eq_atom in set(idx):
                w = wyckoffs[eq_atom]
                count_w = len(np.where(equivalent_atoms == eq_atom)[0])
                wyckoff_n += [w]
                self.wyckoff_multiplicities[w] = count_w

            wyckoff_list += [''.join(sorted(wyckoff_n))]

        wyckoff_list = np.array(wyckoff_list)
        species_list = np.array(species_list)
        for c in set(counts):
            idx_count = np.where(counts == c)[0]
            if len(idx_count) == 1:
                continue

            idx_count_sort = np.argsort(wyckoff_list[idx_count])
            wyckoff_list[idx_count] = wyckoff_list[idx_count][idx_count_sort]
            species_list[idx_count] = species_list[idx_count][idx_count_sort]

        self.ordered_wyckoffs = wyckoff_list
        self.ordered_species = species_list

        self.wyckoffs = []
        self.species = []
        for s, w in zip(species_list, wyckoff_list):
            for wi in w:
                self.wyckoffs += [wi]
                self.species += [s]

    def get_primitive_atoms(self, atoms):
        """Transform from primitive to conventional cell"""

        lattice, scaled_positions, numbers = standardize_cell(atoms,
                                                              to_primitive=True,
                                                              no_idealize=False,
                                                              symprec=1e-5)

        atoms = Atoms(numbers=numbers,
                      cell=lattice,
                      pbc=True)

        atoms.set_scaled_positions(scaled_positions)

        atoms.wrap()

        return atoms

    def get_prototype_name(self):
        alphabet = list(string.ascii_uppercase)
        unique_symbols = []
        symbol_count = []
        species = self.species
        for w, s in zip(self.wyckoffs, self.species):
            if s in unique_symbols:
                index = unique_symbols.index(s)
                symbol_count[index] += self.wyckoff_multiplicities[w]
            else:
                symbol_count += [self.wyckoff_multiplicities[w]]
                unique_symbols += [s]

        min_rep = min(symbol_count)

        for n in list(range(1, min_rep + 1))[::-1]:
            if np.all(np.array(symbol_count) % n == 0):
                repetition = n
                break

        p_name = ''
        for ii, i in enumerate(np.argsort(symbol_count)):
            p_name += alphabet[ii]
            factor = symbol_count[i] // repetition
            if factor > 1:
                p_name += str(factor)

        repetition = repetition * len(set(self.spglibdata['mapping_to_primitive'])) \
            / len(self.spglibdata['std_types'])

        p_name += '_' + str(int(repetition))

        w_name = []
        for ws in self.ordered_wyckoffs:
            for letter in set(list(ws)):
                count = ws.count(letter)
                if count > 1:
                    ws = ws.replace(letter * count, letter + str(count))
            w_name += [ws]
        w_name = '_'.join(w_name)
        p_name += '_' + w_name
        p_name += '_' + str(self.spacegroup)

        return p_name

    def get_spacegroup(self):
        return self.spacegroup

    def get_wyckoff_species(self):
        return self.wyckoffs, self.species

    def get_classification(self, include_parameters=True):

        p_name = self.get_prototype_name()

        structure_name = str(self.spacegroup)
        for spec, wy_spec in zip(self.species, self.wyckoffs):
            structure_name += '_{}_{}'.format(spec, wy_spec)

        prototype = {'p_name': p_name,
                     'structure_name': structure_name,
                     'spacegroup': self.spacegroup,
                     'wyckoffs': self.wyckoffs,
                     'species': self.species}

        # if include_parameters:
        #    cell_parameters = self.get_cell_parameters(self.atoms)

        #    return prototype, cell_parameters

        # else:
        return prototype


def get_wyckoff_pair_symmetry_matrix(spacegroup):
    letters, multiplicity = get_wyckoff_letters_and_multiplicity(spacegroup)

    n_points = len(letters)
    pair_names = []
    for i in range(n_points):
        for j in range(n_points):
            pair_names.append('{}_{}'.format(letters[i], letters[j]))

    M = np.zeros([n_points**2, n_points**2])

    with open(wyckoff_pairs, 'r') as f:
        sg = 1
        for i, line in enumerate(f):
            if len(line) == 1:
                if sg < spacegroup:
                    sg += 1
                    continue
                else:
                    break
            if sg < spacegroup:
                continue
            w_1 = line[:3]
            if not w_1 in pair_names:
                continue
            k = pair_names.index(w_1)
            pairs0 = line.split('\t')[2: -1]
            for w_2 in pairs0:
                j = pair_names.index(w_2)
                M[k, j] = 1

    free_letters = []
    for l in letters:
        i = pair_names.index(l + '_' + l)
        if M[i, i] == 1:
            free_letters += [l]

    np.fill_diagonal(M, 1)

    return pair_names, M, free_letters


def get_wyckoff_letters_and_multiplicity(spacegroup):
    letters = np.array([], dtype=str)
    multiplicity = np.array([])
    with open(wyckoff_data, 'r') as f:
        i_sg = np.inf
        i_w = np.inf
        for i, line in enumerate(f):
            if '1 {} '.format(spacegroup) in line \
               and not i_sg < np.inf:
                i_sg = i
            if i > i_sg:
                if len(line) > 15:
                    continue
                if len(line) == 1:
                    break
                multi, w, sym, sym_multi = line.split(' ')
                letters = np.insert(letters, 0, w)
                multiplicity = np.insert(multiplicity, 0, multi)

    return letters, multiplicity
