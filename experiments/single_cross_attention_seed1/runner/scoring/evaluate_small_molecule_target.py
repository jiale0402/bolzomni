#!/usr/bin/env python3
"""Run the official CBGBench measurements for one benchmark target, resume-safely."""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import traceback
import types
import typing
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import torch

# CBGBench's pinned SA scorer predates removal of rdkit.six. It only imports
# cPickle from that module, so provide that exact compatibility surface.
if "rdkit.six.moves" not in sys.modules:
    rdkit_six = types.ModuleType("rdkit.six")
    rdkit_six_moves = types.ModuleType("rdkit.six.moves")
    rdkit_six_moves.cPickle = pickle
    rdkit_six.iteritems = lambda mapping: mapping.items()
    rdkit_six.moves = rdkit_six_moves
    sys.modules["rdkit.six"] = rdkit_six
    sys.modules["rdkit.six.moves"] = rdkit_six_moves
if not hasattr(torch, "Any"):
    torch.Any = typing.Any


VERSION = 7
INTERACTION_KEYS = (
    "num_hydrophobic", "num_hydrogen", "num_wb", "num_pi_stack",
    "num_pi_cation", "num_halogen", "num_metal",
)
VDW_RADII = {
    1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.8, 16: 1.8, 17: 2.27, 35: 1.85,
}


def required_candidate_fields(benchmark):
    required = {"substructure", "chemical", "geometry", "vina", "interaction"}
    if benchmark == "mgtbind_holdout":
        required.add("molecular_glue")
    return required


def merge_recovered_fields(cached, recovered, required):
    """Preserve successful cached metrics while filling previously failed families."""
    if cached.get("version") != VERSION:
        return recovered
    merged = dict(cached)
    for key in required:
        if key not in merged and key in recovered:
            merged[key] = recovered[key]
    for key in ("atom_num", "target_id", "n", "ligand"):
        if key not in merged and key in recovered:
            merged[key] = recovered[key]
    errors = dict(cached.get("errors", {}))
    errors.update(recovered.get("errors", {}))
    for key in required:
        if key in recovered and key not in recovered.get("errors", {}):
            errors.pop(key, None)
    merged["errors"] = errors
    merged["version"] = VERSION
    return merged


def read_jsonl(path):
    rows = []
    if path.is_file():
        for raw in path.read_text(errors="replace").splitlines():
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def native(value):
    if isinstance(value, dict):
        return {str(key): native(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def atomic_write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(native(value), indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def prepare_receptor(receptor_source, work):
    import AutoDockTools

    receptor = work / "receptor.pdb"
    pqr = work / "receptor.pqr"
    pdbqt = work / "receptor.pdbqt"
    provenance = work / "receptor_preparation.json"
    if not receptor.is_file() or receptor.stat().st_size == 0:
        shutil.copy2(receptor_source, receptor)
    script = Path(AutoDockTools.__path__[0]) / "Utilities24/prepare_receptor4.py"
    preparation_method = None
    primary_error = None
    if not pdbqt.is_file() or pdbqt.stat().st_size == 0:
        pqr_tmp = work / "receptor.pqr.tmp"
        pdbqt_tmp = work / "receptor.pdbqt.tmp"
        pqr_tmp.unlink(missing_ok=True)
        pdbqt_tmp.unlink(missing_ok=True)
        try:
            if (
                not pqr.is_file() or pqr.stat().st_size == 0
                or not provenance.is_file()
            ):
                subprocess.run(
                    [
                        shutil.which("pdb2pqr30"), "--ff=AMBER", str(receptor),
                        str(pqr_tmp),
                    ],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True,
                )
                if not pqr_tmp.is_file() or pqr_tmp.stat().st_size == 0:
                    raise RuntimeError("PDB2PQR returned without a non-empty receptor")
                os.replace(pqr_tmp, pqr)
            subprocess.run(
                [sys.executable, str(script), "-r", str(pqr), "-o", str(pdbqt_tmp)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            if not pdbqt_tmp.is_file() or pdbqt_tmp.stat().st_size == 0:
                raise RuntimeError("AutoDockTools returned an empty PDBQT")
            os.replace(pdbqt_tmp, pdbqt)
            preparation_method = "pdb2pqr_amber_then_autodocktools"
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            # CrossDocked receptors are cropped pockets.  A small subset contains
            # backbone gaps or more than PDB2PQR's 10% missing-heavy-atom repair
            # limit.  AutoDockTools can deterministically assign Gasteiger charges
            # and add hydrogens directly to those observed atoms, which is preferable
            # to dropping the target.  Released and tuned evaluations share the same
            # receptor source and therefore the same fallback.
            primary_error = (
                exc.stderr[-4000:] if isinstance(exc, subprocess.CalledProcessError)
                and exc.stderr else str(exc)
            )
            pqr_tmp.unlink(missing_ok=True)
            pdbqt_tmp.unlink(missing_ok=True)
            subprocess.run(
                [
                    sys.executable, str(script), "-r", str(receptor),
                    "-o", str(pdbqt_tmp), "-A", "hydrogens",
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            if not pdbqt_tmp.is_file() or pdbqt_tmp.stat().st_size == 0:
                raise RuntimeError("direct AutoDockTools fallback returned an empty PDBQT")
            os.replace(pdbqt_tmp, pdbqt)
            preparation_method = "autodocktools_direct_incomplete_pocket_fallback"
        finally:
            pqr_tmp.unlink(missing_ok=True)
            pdbqt_tmp.unlink(missing_ok=True)
        atomic_write_json(provenance, {
            "evaluator_version": VERSION,
            "method": preparation_method,
            "receptor_source": str(receptor_source),
            "receptor_source_sha256": hashlib.sha256(receptor.read_bytes()).hexdigest(),
            "primary_error": primary_error,
        })
    if not pdbqt.is_file() or pdbqt.stat().st_size == 0:
        raise RuntimeError(f"receptor PDBQT preparation failed: {pdbqt}")
    return receptor


def interaction_counts(receptor, ligand, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    protein = Chem.MolFromPDBFile(str(receptor), sanitize=False, removeHs=False)
    ligand_mol = Chem.SDMolSupplier(str(ligand), removeHs=False)[0]
    if protein is None or ligand_mol is None:
        raise RuntimeError("RDKit could not create the PLIP complex")
    merged = cache_dir / "plip_complex.pdb"
    Chem.MolToPDBFile(Chem.CombineMols(protein, ligand_mol), str(merged))
    out_dir = cache_dir / "plip"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [shutil.which("plip"), "-f", str(merged), "-o", str(out_dir), "-x"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    reports = list(out_dir.glob("*_report.xml"))
    if len(reports) != 1:
        raise RuntimeError(f"expected one PLIP XML report in {out_dir}, found {len(reports)}")
    report = ET.parse(reports[0]).getroot()
    interactions = report.findall("bindingsite/interactions")
    if not interactions:
        return {key: 0 for key in INTERACTION_KEYS}
    interactions = interactions[0]
    tags = {
        "num_hydrophobic": "hydrophobic_interactions/*",
        "num_hydrogen": "hydrogen_bonds/*",
        "num_wb": "water_bridges/*",
        "num_pi_stack": "pi_stacks/*",
        "num_pi_cation": "pi_cation_interactions/*",
        "num_halogen": "halogen_bonds/*",
        "num_metal": "metal_complexes/*",
    }
    return {key: len(interactions.findall(tag)) for key, tag in tags.items()}


def cbg_steric_clash(mol, receptor, tolerance=0.4):
    ligand_pos = mol.GetConformer().GetPositions()
    ligand_elements = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    adjacency = np.zeros((len(ligand_pos), len(ligand_pos)), dtype=bool)
    for bond in mol.GetBonds():
        adjacency[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = True
        adjacency[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = True
    protein_pos, protein_elements = [], []
    structure = PDBParser(QUIET=True).get_structure("protein", str(receptor))
    periodic = Chem.GetPeriodicTable()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == "H":
                        continue
                    atomic_number = periodic.GetAtomicNumber(atom.element.capitalize())
                    if atomic_number in VDW_RADII:
                        protein_pos.append(atom.coord)
                        protein_elements.append(atomic_number)
        break
    protein_pos = np.asarray(protein_pos)
    protein_elements = np.asarray(protein_elements)

    def detect(dst_pos, src_pos, dst_elements, src_elements, mask):
        distances = cdist(dst_pos, src_pos)
        thresholds = (
            np.array([VDW_RADII[int(value)] for value in dst_elements])[:, None]
            + np.array([VDW_RADII[int(value)] for value in src_elements])[None, :]
            - tolerance
        )
        clashes = (distances < thresholds) & mask
        return int(len(np.unique(np.where(clashes)[0]))), bool(np.any(clashes))

    inter_num, inter_detected = detect(
        ligand_pos, protein_pos, ligand_elements, protein_elements,
        np.ones((len(ligand_pos), len(protein_pos)), dtype=bool),
    )
    intra_mask = (~adjacency) ^ np.eye(len(ligand_pos), dtype=bool)
    intra_num, intra_detected = detect(
        ligand_pos, ligand_pos, ligand_elements, ligand_elements, intra_mask,
    )
    return {
        "lig_pro_clash_detected": inter_detected,
        "lig_lig_clash_detected": intra_detected,
        "lig_pro_clash": {"clash_atom_num": inter_num},
        "lig_lig_clash": {"clash_atom_num": intra_num},
    }


def vina_scores(mol, ligand, receptor, cache_dir, exhaustiveness):
    # docking_vina imports reconstruct_mol for unrelated data-object entry
    # points. The path used here (from_generated_mol) never calls it, while the
    # upstream module transitively imports the whole torch-geometric model stack.
    # Supply only that unused symbol so the official Vina implementation remains
    # usable in the isolated evaluation environment.
    if "repo.tools.rdkit_utils" not in sys.modules:
        rdkit_utils_stub = types.ModuleType("repo.tools.rdkit_utils")

        def reconstruct_mol(*_args, **_kwargs):
            raise RuntimeError("reconstruct_mol is unavailable in benchmark-only mode")

        rdkit_utils_stub.reconstruct_mol = reconstruct_mol
        sys.modules["repo.tools.rdkit_utils"] = rdkit_utils_stub
    from repo.tools.docking_vina import VinaDockingTask

    (cache_dir / "docking_results").mkdir(parents=True, exist_ok=True)
    task = VinaDockingTask.from_generated_mol(
        mol, str(ligand), protein_path=str(receptor)
    )
    if ligand.parent.name == "reference_ligands":
        seed_label = f"{ligand.stem}:reference"
    else:
        seed_label = f"{ligand.parent.name}:candidate:{ligand.stem}"
    seed = 1 + int.from_bytes(
        hashlib.sha256(seed_label.encode("utf-8")).digest()[:4], "big"
    ) % 2_000_000_000
    output = {}
    for mode in ("score_only", "minimize", "dock"):
        result = task.run(
            mode=mode, exhaustiveness=exhaustiveness,
            save_dir=str(cache_dir / "docking_results"),
            seed=seed,
        )
        output[mode] = float(result["affinity"])
    return output


def dual_interface_contacts(receptor, ligand, target_chains, cutoff=4.0):
    protein = PDBParser(QUIET=True).get_structure("receptor", str(receptor))
    mol = Chem.SDMolSupplier(str(ligand), removeHs=True)[0]
    ligand_xyz = mol.GetConformer().GetPositions()
    counts = {}
    contacted = 0
    for chain_id in target_chains:
        coords = []
        for model in protein:
            if chain_id not in model:
                continue
            for residue in model[chain_id]:
                coords.extend(atom.coord for atom in residue if atom.element != "H")
        if coords:
            distances, _ = cKDTree(np.asarray(coords)).query(ligand_xyz, k=1)
            count = int(np.sum(distances < cutoff))
        else:
            count = 0
        counts[chain_id] = count
        contacted += int(count > 0)
    return {
        "ligand_atoms_contacting_each_chain": counts,
        "number_of_target_chains_contacted": contacted,
        "dual_interface_contact": contacted >= 2,
        "contact_cutoff_angstrom": cutoff,
    }


def evaluate_ligand(ligand, receptor, cache_dir, row, exhaustiveness, reference=False):
    from repo.tools import eval_atom_type, eval_fg_type, eval_ring_type, scoring
    from repo.tools.geometry import eval_bond_angle, eval_bond_length, eval_stability

    mol = Chem.SDMolSupplier(str(ligand), removeHs=True)[0]
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse {ligand}")
    result = {"version": VERSION, "ligand": str(ligand), "errors": {}}
    if not reference:
        try:
            result["substructure"] = {
                "atom_type": dict(Counter(eval_atom_type.get_atom_from_mol_num(mol))),
                "functional_group": dict(Counter(eval_fg_type.get_func_group_from_mol(mol))),
                "ring_type": dict(Counter(eval_ring_type.ring_type_from_mol(mol))),
            }
            # CBGBench returns a ring-size Counter alongside the four published
            # scalar chemistry metrics; ring distributions are already measured
            # by eval_ring_type above.
            result["chemical"] = {
                key: float(value) for key, value in scoring.get_chem(mol).items()
                if value is not None and key != "ring_size"
            }
        except Exception:
            result["errors"]["substructure_or_chemical"] = traceback.format_exc(limit=3)
        try:
            positions = mol.GetConformer().GetPositions()
            atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            stable = eval_stability.check_stability(positions, atom_types)
            clash = cbg_steric_clash(mol, receptor)
            result["geometry"] = {
                "bond_dist": eval_bond_length.bond_distance_from_mol(mol),
                "bond_angle": eval_bond_angle.bond_angle_from_mol(mol),
                "mol_stable": bool(stable[0]),
                "atom_stable_num": int(stable[1]),
                "atom_num": int(stable[2]),
                "intra_clash_atom_num": int(clash["lig_lig_clash"]["clash_atom_num"]),
                "inter_clash_atom_num": int(clash["lig_pro_clash"]["clash_atom_num"]),
                "inter_clash_detected": bool(clash["lig_pro_clash_detected"]),
            }
        except Exception:
            result["errors"]["geometry"] = traceback.format_exc(limit=3)
    try:
        result["vina"] = vina_scores(mol, ligand, receptor, cache_dir, exhaustiveness)
    except Exception:
        result["errors"]["vina"] = traceback.format_exc(limit=5)
    try:
        result["interaction"] = interaction_counts(receptor, ligand, cache_dir)
    except Exception:
        result["errors"]["interaction"] = traceback.format_exc(limit=5)
    if row["benchmark"] == "mgtbind_holdout" and not reference:
        try:
            result["molecular_glue"] = dual_interface_contacts(
                receptor, ligand, row["target_chain_ids"]
            )
        except Exception:
            result["errors"]["molecular_glue"] = traceback.format_exc(limit=3)
    result["atom_num"] = int(mol.GetNumAtoms())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--cbg-root", required=True, type=Path)
    parser.add_argument("--require-evaluated", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.cbg_root.resolve()))
    rows = read_jsonl(args.manifest)
    generation = read_jsonl(args.candidates / "results.jsonl")
    if args.target_index < 0:
        # Smoke tests need one candidate supported by every official CBGBench
        # core metric. Its stability routine is limited to this element set.
        supported = {1, 6, 7, 8, 9, 15, 16, 17}
        success_keys = {(item["id"], item["n"]) for item in generation}
        row = None
        for candidate_row in rows:
            for n in range(args.n_samples):
                if (candidate_row["id"], n) not in success_keys:
                    continue
                ligand = args.candidates / candidate_row["id"] / f"{n}.sdf"
                mol = Chem.SDMolSupplier(str(ligand), removeHs=True)[0]
                if mol is not None and {
                    atom.GetAtomicNum() for atom in mol.GetAtoms()
                } <= supported:
                    row = candidate_row
                    break
            if row is not None:
                break
        if row is None:
            raise RuntimeError("no generated candidate supports every CBGBench core metric")
    else:
        row = rows[args.target_index]
    target_id = row["id"]
    work = args.cache_root / target_id
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    receptor = prepare_receptor(Path(row["receptor_pdb"]), work)

    reference_cache = work / "reference.json"
    cached_reference = (
        json.loads(reference_cache.read_text()) if reference_cache.is_file() else {}
    )
    required_reference = {"vina", "interaction"}
    if (
        cached_reference.get("version") != VERSION
        or not required_reference.issubset(cached_reference)
    ):
        recovered_reference = evaluate_ligand(
            Path(row["reference_ligand_sdf"]), receptor, work / "reference_work",
            row, args.exhaustiveness, reference=True,
        )
        cached_reference = merge_recovered_fields(
            cached_reference, recovered_reference, required_reference
        )
        atomic_write_json(reference_cache, cached_reference)
    reference_vina_complete = int("vina" in cached_reference)
    reference_interaction_complete = int("interaction" in cached_reference)

    success_ns = {
        item["n"] for item in generation
        if item.get("id") == target_id and 0 <= item.get("n", -1) < args.n_samples
    }
    evaluated = 0
    complete_core = complete_vina = complete_interaction = complete_glue = 0
    required_candidate = required_candidate_fields(row["benchmark"])
    for n in sorted(success_ns):
        cache = work / f"candidate_{n:03d}.json"
        record = {}
        if cache.is_file():
            record = json.loads(cache.read_text())
        if record.get("version") != VERSION or not required_candidate.issubset(record):
            ligand = args.candidates / target_id / f"{n}.sdf"
            recovered = evaluate_ligand(
                ligand, receptor, work / f"candidate_{n:03d}_work",
                row, args.exhaustiveness,
            )
            record = merge_recovered_fields(record, recovered, required_candidate)
        record.update({"target_id": target_id, "n": n})
        atomic_write_json(cache, record)
        evaluated += 1
        complete_core += int("chemical" in record and "geometry" in record)
        complete_vina += int("vina" in record)
        complete_interaction += int("interaction" in record)
        complete_glue += int("molecular_glue" in record)
    summary = {
        "version": VERSION,
        "target_index": rows.index(row),
        "target_id": target_id,
        "requested": args.n_samples,
        "generation_successes": len(success_ns),
        "evaluated": evaluated,
        "complete_core": complete_core,
        "complete_vina": complete_vina,
        "complete_interaction": complete_interaction,
        "complete_molecular_glue": complete_glue,
        "reference_vina_complete": reference_vina_complete,
        "reference_interaction_complete": reference_interaction_complete,
    }
    atomic_write_json(work / "target_summary.json", summary)
    if args.require_evaluated and not (
        evaluated > 0 and complete_core > 0
        and complete_vina > 0 and complete_interaction > 0
        and reference_vina_complete > 0 and reference_interaction_complete > 0
        and (row["benchmark"] != "mgtbind_holdout" or complete_glue > 0)
    ):
        raise RuntimeError(
            f"incomplete smoke evaluation for {target_id}: "
            f"evaluated={evaluated}, core={complete_core}, "
            f"vina={complete_vina}, interaction={complete_interaction}, "
            f"molecular_glue={complete_glue}, reference_vina={reference_vina_complete}, "
            f"reference_interaction={reference_interaction_complete}"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
