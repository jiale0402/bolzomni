"""Native benchmark inputs and sanitized donor records; no training-code edits."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from runtime import SOURCE
sys.path.insert(0, str(SOURCE))
from api.pdb_dataset import PDBDataset
from api.templates.pep import LinearPeptide
from api.templates.antibody import Antibody
from api.templates.mol import Molecule as MoleculeTemplate
from data.bioparse.hierarchy import Molecule


class NativeLengthPeptide(LinearPeptide):
    def __call__(self, desc):
        desc = copy.deepcopy(desc)
        length = sum(len(desc.cplx[c]) for c in desc.lig_chains)
        assert length > 0
        desc.cplx = self.remove_ref_lig(desc)
        blocks, bonds = self.dummy_lig_block_bonds(desc, length)
        desc.cplx = type(desc.cplx)(name=desc.cplx.name,
            molecules=desc.cplx.molecules + [Molecule('dummy', blocks, desc.lig_chains[0])],
            bonds=desc.cplx.bonds + bonds, properties=desc.cplx.properties)
        return self.set_desc_attr(desc)


def dataset(benchmark, row):
    template = {'lnr': NativeLengthPeptide, 'crossdocked2020': MoleculeTemplate}
    config = Antibody(cdr_type=row['cdr_type']) if benchmark == 'rabd' else template[benchmark]()
    return PDBDataset([row['generation_reference']], [row['target_chain_ids']],
                      config, lig_chains=[row['ligand_chain_ids']])

