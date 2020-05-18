import shutil
import os
import tempfile
import unittest

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.calculate.submit import TriSubmit


class TriSubmitTest(unittest.TestCase):
    def setUp(self):
        self.bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        self.atoms = self.bb_iron.get_atoms()
        self.submitter = TriSubmit(basepath_ext='tests',
                                   ncpus=1,
                                   queue='small'
                                   )
        self.submitter.set_atoms(self.atoms)
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_write_poscar(self):
        self.submitter.write_poscar('.')

    def test_write_model(self):
        self.submitter.write_model('.',
                                   master_parameters={'encut': 300,
                                                      'kspacing': 0.5}
                                   )

        with open('model.py') as f:
            model_text = f.readlines()

    def test_submit(self):
        self.submitter.submit_calculation(self.atoms)


if __name__ == '__main__':
    unittest.main()
