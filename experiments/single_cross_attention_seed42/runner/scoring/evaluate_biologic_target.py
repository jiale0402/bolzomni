#!/usr/bin/env python3
"""Evaluate one LNR or RAbD target with AnewOmni paper metrics."""

import argparse
import hashlib
import io
import importlib.util
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select


VERSION = 6
CA_CLASH_DISTANCE = 3.6574
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
SEQUENCE_TOKEN = re.compile(r"f[0-9]+|.")


def read_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(errors="replace").splitlines() if line.strip()]


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def rosetta_record_complete(record):
    return "dG" in record and "ddG" in record


def normalize_candidate_record(record, target_id, n):
    if record.get("version") != VERSION:
        return {
            "version": VERSION, "target_id": target_id, "n": n, "errors": {}
        }
    record = dict(record)
    record.update({"target_id": target_id, "n": n})
    errors = dict(record.get("errors", {}))
    if "structural" in record:
        errors.pop("structural", None)
    if rosetta_record_complete(record):
        errors.pop("rosetta", None)
    record["errors"] = errors
    return record


def parse_standard_protein_pdb(path, label):
    """Parse only standard PDB protein records.

    Some generated complexes contain ligand HETATM names longer than the PDB
    four-character atom-name field.  Those records shift the chain/residue
    columns and make Bio.PDB reject the entire otherwise valid protein model.
    The biologic metrics and Rosetta input intentionally use standard protein
    residues only, so exclude HETATM records before parsing.
    """
    path = Path(path)
    records = [
        line for line in path.read_text(errors="replace").splitlines(keepends=True)
        if line.startswith(("ATOM  ", "MODEL ", "ENDMDL", "TER   "))
    ]
    if not any(line.startswith("ATOM  ") for line in records):
        raise RuntimeError(f"no standard protein ATOM records in {path}")
    return PDBParser(QUIET=True).get_structure(
        label, io.StringIO("".join(records) + "END\n")
    )


def residues_by_chain(path, chains):
    structure = parse_standard_protein_pdb(path, "structure")
    output = {chain: [] for chain in chains}
    for model in structure:
        for chain in model:
            if chain.id not in output:
                continue
            for residue in chain:
                if "CA" in residue:
                    key = (chain.id, int(residue.id[1]), str(residue.id[2]).strip())
                    output[chain.id].append((key, residue, np.asarray(residue["CA"].coord, dtype=float)))
        break
    return output


def mark_hcdr3(chain_residues):
    from data.bioparse.numbering import get_nsys

    marks = get_nsys().mark_heavy_seq([item[0][1] for item in chain_residues])
    if len(marks) != len(chain_residues):
        raise RuntimeError("antibody number-system mask length mismatch")
    return [item for item, mark in zip(chain_residues, marks) if mark == "3"]


def kabsch_transform(moving, fixed):
    moving = np.asarray(moving, dtype=float)
    fixed = np.asarray(fixed, dtype=float)
    moving_mean = moving.mean(axis=0)
    fixed_mean = fixed.mean(axis=0)
    centered_moving = moving - moving_mean
    centered_fixed = fixed - fixed_mean
    covariance = centered_moving.T @ centered_fixed
    left, _, right = np.linalg.svd(covariance)
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    rotation = left @ right
    translation = fixed_mean - moving_mean @ rotation
    return rotation, translation


def rmsd(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def tokenize_sequence(sequence):
    """Split canonical one-letter residues and AnewOmni ``f###`` blocks."""
    tokens = SEQUENCE_TOKEN.findall(sequence)
    if not tokens or "".join(tokens) != sequence:
        raise RuntimeError(f"invalid AnewOmni sequence encoding: {sequence!r}")
    return tokens


def map_sequence_blocks_to_ca(tokens, residues):
    """Map encoded sequence-block positions to the CA atoms represented in a PDB.

    LNR references can contain ``f###`` cap/noncanonical blocks with no CA atom,
    and a small number of references have an unresolved canonical CA.  Dynamic
    programming keeps residue order, maximizes represented atoms, and then
    prefers exact canonical residue matches over wildcard ``f###`` matches.
    """
    n_tokens = len(tokens)
    n_residues = len(residues)
    # State score is lexicographic: represented atoms, exact canonical matches,
    # and finally fewer unmatched PDB CA atoms.
    scores = [[None] * (n_residues + 1) for _ in range(n_tokens + 1)]
    previous = [[None] * (n_residues + 1) for _ in range(n_tokens + 1)]
    scores[0][0] = (0, 0, 0)

    def update(i, j, score, parent):
        if scores[i][j] is None or score > scores[i][j]:
            scores[i][j] = score
            previous[i][j] = parent

    for i in range(n_tokens + 1):
        for j in range(n_residues + 1):
            score = scores[i][j]
            if score is None:
                continue
            if i < n_tokens:
                update(i + 1, j, score, (i, j, "skip_token"))
            if j < n_residues:
                update(i, j + 1, (score[0], score[1], score[2] - 1),
                       (i, j, "skip_residue"))
            if i < n_tokens and j < n_residues:
                token = tokens[i]
                symbol = AA3_TO_1.get(residues[j][1].resname)
                is_exact = len(token) == 1 and token == symbol
                if token.startswith("f") or is_exact:
                    update(i + 1, j + 1,
                           (score[0] + 1, score[1] + int(is_exact), score[2]),
                           (i, j, "match"))

    block_map = {}
    i, j = n_tokens, n_residues
    while i or j:
        parent = previous[i][j]
        if parent is None:
            raise RuntimeError("failed to align encoded sequence blocks to PDB CA atoms")
        parent_i, parent_j, action = parent
        if action == "match":
            block_map[parent_i] = residues[parent_j][2]
        i, j = parent_i, parent_j
    return block_map, scores[n_tokens][n_residues]


def inner_clash_ratio(coords):
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return None
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    excluded = np.eye(len(coords), dtype=bool)
    if len(coords) > 1:
        excluded[np.arange(len(coords) - 1), np.arange(1, len(coords))] = True
        excluded[np.arange(1, len(coords)), np.arange(len(coords) - 1)] = True
    clashed = (distances < CA_CLASH_DISTANCE) & ~excluded
    return float(len(np.unique(np.where(clashed)[0])) / len(coords))


def interface_clash_ratio(ligand, target):
    ligand = np.asarray(ligand, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(ligand) == 0 or len(target) == 0:
        return None
    distances = np.linalg.norm(ligand[:, None] - target[None, :], axis=-1)
    return float(len(np.unique(np.where(distances < CA_CLASH_DISTANCE)[0])) / len(ligand))


def select_coordinates(path, row, mode):
    all_chains = list(dict.fromkeys(row["target_chain_ids"] + row["ligand_chain_ids"]))
    residues = residues_by_chain(path, all_chains)
    target = [item for chain in row["target_chain_ids"] for item in residues.get(chain, [])]
    if mode == "peptide":
        ligand = [item for chain in row["ligand_chain_ids"] for item in residues.get(chain, [])]
    else:
        heavy = row["heavy_chain_id"]
        ligand = mark_hcdr3(residues.get(heavy, []))
    return target, ligand


def structural_metrics(generated, reference, row, mode, generated_sequence):
    ref_target, ref_ligand = select_coordinates(reference, row, mode)
    gen_target, gen_ligand = select_coordinates(generated, row, mode)
    ref_target_map = {item[0]: item[2] for item in ref_target}
    gen_target_map = {item[0]: item[2] for item in gen_target}
    common = sorted(set(ref_target_map) & set(gen_target_map))
    if len(common) < 3:
        raise RuntimeError(f"only {len(common)} common target CA atoms")
    reference_tokens = tokenize_sequence(row["reference_sequence"])
    generated_tokens = tokenize_sequence(generated_sequence)
    if len(generated_tokens) != len(reference_tokens):
        raise RuntimeError(
            f"generated/reference block-count mismatch: {len(generated_tokens)} vs {len(reference_tokens)}"
        )
    ref_ligand_map, ref_alignment_score = map_sequence_blocks_to_ca(
        reference_tokens, ref_ligand
    )
    gen_ligand_map, gen_alignment_score = map_sequence_blocks_to_ca(
        generated_tokens, gen_ligand
    )
    common_ligand = sorted(set(ref_ligand_map) & set(gen_ligand_map))
    if not common_ligand:
        raise RuntimeError("no common generated/reference ligand CA atoms")
    # LNR contains terminal caps/noncanonical blocks without a CA atom and one
    # reference with an unresolved CA.  Match by chain/residue key rather than
    # rejecting the complete target because the raw CA counts differ.
    ref_ligand_xyz = np.asarray([ref_ligand_map[key] for key in common_ligand])
    gen_ligand_common_xyz = np.asarray([gen_ligand_map[key] for key in common_ligand])
    gen_ligand_xyz = np.asarray([item[2] for item in gen_ligand])
    rotation, translation = kabsch_transform(
        [gen_target_map[key] for key in common], [ref_target_map[key] for key in common]
    )
    lig_rotation, lig_translation = kabsch_transform(gen_ligand_common_xyz, ref_ligand_xyz)
    target_xyz = np.asarray([item[2] for item in gen_target])
    result = {
        "AAR": sum(a == b for a, b in zip(generated_tokens, reference_tokens)) / len(reference_tokens),
        "complex_RMSD": rmsd(
            gen_ligand_common_xyz @ rotation + translation, ref_ligand_xyz
        ),
        "intra_region_clash": inner_clash_ratio(gen_ligand_xyz),
        "interface_clash": interface_clash_ratio(gen_ligand_xyz, target_xyz),
        "region_CA_count": len(gen_ligand),
        "common_region_CA": len(common_ligand),
        "region_block_count": len(generated_tokens),
        "reference_CA_alignment_score": ref_alignment_score,
        "generated_CA_alignment_score": gen_alignment_score,
        "common_target_CA": len(common),
    }
    if mode == "peptide":
        result["peptide_RMSD"] = rmsd(
            gen_ligand_common_xyz @ lig_rotation + lig_translation, ref_ligand_xyz
        )
    return result


def load_rosetta_energy_module(pepglad_root):
    path = pepglad_root / "evaluation/dG/energy.py"
    spec = importlib.util.spec_from_file_location("anewomni_pepglad_energy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandardProteinSelect(Select):
    """Match Rosetta's ignore-unrecognized-residue input behavior explicitly."""

    def accept_residue(self, residue):
        return residue.id[0] == " " and residue.resname in AA3_TO_1


def prepare_rosetta_input(source, clean):
    """Normalize PDB columns and omit ions/waters before PyRosetta parsing."""
    if clean.is_file() and clean.stat().st_size > 0:
        return clean
    structure = parse_standard_protein_pdb(source, "rosetta_input")
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(clean), StandardProteinSelect())
    if not clean.is_file() or clean.stat().st_size == 0:
        raise RuntimeError(f"failed to prepare Rosetta input {clean}")
    return clean


def deterministic_seed(label):
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 2_000_000_000


def rosetta_metrics(source, relaxed, row, mode, energy, seed):
    flexible_chain = row["ligand_chain_ids"][0] if mode == "peptide" else row["heavy_chain_id"]
    if not relaxed.is_file() or relaxed.stat().st_size == 0:
        clean_source = prepare_rosetta_input(
            source, relaxed.with_name(relaxed.stem + "_input.pdb")
        )
        energy.pyrosetta.rosetta.numeric.random.rg().set_seed(seed)
        energy.pyrosetta_fastrelax(str(clean_source), str(relaxed), flexible_chain)
    # PepGLAD's released evaluator passes an ``A_B`` string directly to the
    # InterfaceAnalyzerMover constructor. PyRosetta 2026 exposes the same
    # calculation through a typed DockingPartners object instead.
    from pyrosetta.rosetta.core.pose import DockingPartners
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    pose = energy.pyrosetta.pose_from_pdb(str(relaxed))
    interface = "".join(row["ligand_chain_ids"]) + "_" + "".join(row["target_chain_ids"])
    partners = DockingPartners().docking_partners_from_string(interface)
    mover = InterfaceAnalyzerMover(partners)
    mover.set_pack_separated(True)
    energy.pyrosetta.rosetta.numeric.random.rg().set_seed(seed + 1)
    mover.apply(pose)
    return float(pose.scores["dG_separated"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--mode", choices=("peptide", "antibody"), required=True)
    parser.add_argument("--pepglad-root", required=True, type=Path)
    parser.add_argument("--skip-rosetta", action="store_true")
    parser.add_argument("--require-evaluated", action="store_true")
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    generation = read_jsonl(args.candidates / "results.jsonl")
    if args.target_index < 0:
        success_ids = {item["id"] for item in generation}
        row = next(item for item in rows if item["id"] in success_ids)
    else:
        row = rows[args.target_index]
    target_id = row["id"]
    work = args.cache_root / target_id
    work.mkdir(parents=True, exist_ok=True)
    reference = Path(row["reference_pdb"])
    energy = None if args.skip_rosetta else load_rosetta_energy_module(args.pepglad_root)
    reference_dg = None
    reference_cache = work / "reference_rosetta.json"
    if energy is not None:
        cached_reference = (
            json.loads(reference_cache.read_text()) if reference_cache.is_file() else {}
        )
        if cached_reference.get("version") == VERSION and "dG" in cached_reference:
            reference_dg = cached_reference["dG"]
        else:
            reference_dg = rosetta_metrics(
                reference, work / "reference_rosetta.pdb", row, args.mode, energy,
                deterministic_seed(f"{target_id}:reference"),
            )
            atomic_json(reference_cache, {"version": VERSION, "dG": reference_dg})

    success = {
        (item["id"], item["n"]): item for item in generation
        if item.get("n", -1) < args.n_samples
    }
    evaluated = structural_complete = rosetta_complete = 0
    for n in range(args.n_samples):
        item = success.get((target_id, n))
        if item is None:
            continue
        cache = work / f"candidate_{n:03d}.json"
        record = json.loads(cache.read_text()) if cache.is_file() else {}
        record = normalize_candidate_record(record, target_id, n)
        generated = args.candidates / target_id / f"{n}.pdb"
        if "structural" not in record:
            try:
                record["structural"] = structural_metrics(
                    generated, reference, row, args.mode, item.get("gen_seq", "")
                )
                record["errors"].pop("structural", None)
            except Exception:
                record["errors"]["structural"] = traceback.format_exc(limit=5)
        if energy is not None and not rosetta_record_complete(record):
            try:
                dg = rosetta_metrics(
                    generated, work / f"candidate_{n:03d}_rosetta.pdb",
                    row, args.mode, energy,
                    deterministic_seed(f"{target_id}:candidate:{n}"),
                )
                record["dG"] = dg
                record["ddG"] = dg - reference_dg
                record["errors"].pop("rosetta", None)
            except Exception:
                record["errors"]["rosetta"] = traceback.format_exc(limit=5)
        atomic_json(cache, record)
        evaluated += 1
        structural_complete += int("structural" in record)
        rosetta_complete += int(rosetta_record_complete(record))
    summary = {
        "version": VERSION,
        "target_index": rows.index(row),
        "target_id": target_id,
        "requested": args.n_samples,
        "generation_successes": sum(key[0] == target_id for key in success),
        "evaluated": evaluated,
        "structural_complete": structural_complete,
        "rosetta_complete": rosetta_complete,
        "reference_dG": reference_dg,
    }
    atomic_json(work / "target_summary.json", summary)
    if args.require_evaluated and (
        evaluated == 0 or structural_complete == 0 or (energy is not None and rosetta_complete == 0)
    ):
        raise RuntimeError(f"incomplete required biologic evaluation: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
