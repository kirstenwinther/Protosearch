import json
from ase.db import connect
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from protosearch.build_bulk.be_classification import get_classification


# Sourcing structures from ICSD/MP into ASE db with prototype classification

db = connect('MP.db')
count = db.count()
MP_ids = json.load(open('MP_ids_for_icsd.json', 'r'))

for mp_id in MP_ids[count:]:
    mp_id = mp_id['material_id']
    with MPRester() as m:    
        data = m.get_entries(mp_id,
                             property_data=['material_id',
                                            'formation_energy_per_atom',
                                            'icsd_ids'],
                             inc_structure=True)
    if len(data) == 0:
        print('    No data')
        continue
    atoms = AseAtomsAdaptor.get_atoms(data[0].structure)
    print('    ', atoms.get_chemical_formula())

    data = data[0].as_dict()['data']    
    icsd_ids = data.pop('icsd_ids')    
    key_value_pairs = data.copy()
    
    if icsd_ids:
        key_value_pairs.update({'icsd_ids': json.dumps(icsd_ids)})
        key_value_pairs.update({'icsd_id': min(icsd_ids)})
        
    try:
        prototype = get_classification(atoms)
        p_name = prototype.pop('p_name')
        spacegroup = prototype.pop('spacegroup')
        key_value_pairs.update({'proto_name': p_name,
                                'spacegroup': spacegroup})
    except:
        prototype = None        
        pass

    db.write(atoms, key_value_pairs, data=prototype)


    
