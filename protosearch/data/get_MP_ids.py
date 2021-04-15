import json

from pymatgen.ext.matproj import MPRester

# Write ICSD/MP ids to json file

with MPRester() as m:    
    data = m.query(criteria = {"icsd_ids": {"$ne": []}},
                   properties=['material_id'],
                   chunk_size=100)


json.dump(data, open('MP_ids_for_icsd.json', 'w'))


