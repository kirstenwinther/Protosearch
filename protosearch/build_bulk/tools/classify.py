import ase
from ase.io import read
from sys import argv
from protosearch.build_bulk.classification import PrototypeClassification
from protosearch.build_bulk.be_classification import get_classification
# Script to print structure prototype and parameters

atoms = read(argv[1])


PC = PrototypeClassification(atoms)
prototype = PC.get_classification()
print('Prototype:', prototype['p_name']) #, '\nParameters:', parameters)

prototype = get_classification(atoms)

print(prototype['p_name'])
