import shutil
import os
import tempfile
import unittest

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.calculate.submit import TriSubmit, SlurmSubmit
from protosearch.calculate.vasp import VaspModel


class SubmitTest(unittest.TestCase):

    def test_write_model(self):
        bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        atoms = bb_iron.get_atoms()
        self.submitter = TriSubmit(basepath_ext='tests')
        self.submitter.set_atoms(atoms)
        self.submitter.set_execution_path()
        self.submitter.write_submission_files()

    def test_write_model_slurm(self):
        bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        atoms = bb_iron.get_atoms()
        submitter = SlurmSubmit(partition='regular',
                                account='projectname',
                                nodes=1,
                                ntasks=1,
                                basepath='.')
        submitter.set_atoms(atoms)
        submitter.set_execution_path(strict_format=False)
        submitter.write_submission_files()


if __name__ == '__main__':
    unittest.main()
