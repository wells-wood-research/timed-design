"""
This util file is necessary as ISAMBARD does not integrate well with SCWRL4

It contains several workarounds. See the original under isambard.modelling.scwrl
"""
import os
import re
import subprocess
import tempfile
import typing as t
from pathlib import Path

import ampal
from tqdm import tqdm


def parse_scwrl_out(scwrl_std_out: str, scwrl_pdb: str):
    """Parses SCWRL output and returns PDB and SCWRL score.
    Parameters
    ----------
    scwrl_std_out : str
        Std out from SCWRL.
    scwrl_pdb : str
        String of packed SCWRL PDB.
    Returns
    -------
    fixed_scwrl_str : str
        String of packed SCWRL PDB, with correct PDB format.
    score : float
        SCWRL Score
    """
    score = re.findall(
        r"Total minimal energy of the graph = ([-0-9.]+)", scwrl_std_out
    )[0]
    # Add temperature factors to SCWRL out
    split_scwrl = scwrl_pdb.split("\r\n")[1]
    split_scwrl = split_scwrl.split("\n")
    fixed_scwrl = []
    for line in split_scwrl:
        if len(line) < 80:
            line += " " * (80 - len(line))
        if re.search(r"H?E?T?ATO?M\s+\d+.+", line):
            front = line[:61]
            temp_factor = " 0.00"
            back = line[66:]
            fixed_scwrl.append("".join([front, temp_factor, back]))
        else:
            fixed_scwrl.append(line)
    fixed_scwrl_str = "\n".join(fixed_scwrl) + "\n"
    return fixed_scwrl_str, float(score)


def run_scwrl(
    pdb: str,
    sequence: str,
    scwrl_path: Path,
    path: bool = True,
    rigid_rotamer_model: bool = True,
    hydrogens: bool = False,
) -> (str, str):
    """Runs SCWRL on input PDB strong or path to PDB and a sequence string.

    Parameters
    ----------
    pdb : str
        PDB string or a path to a PDB file.
    sequence : str
        Amino acid sequence for SCWRL to pack in single-letter code.
    path : bool, optional
        True if pdb is a path.
    rigid_rotamer_model : bool, optional
        If True, Scwrl will use the rigid-rotamer model, which is
        faster but less accurate.
    hydrogens : bool, optional
        If False, the hydrogens produced by Scwrl will be ommitted.

    Returns
    -------
    scwrl_std_out : str
        Std out from SCWRL.
    scwrl_pdb : str
        String of packed SCWRL PDB.

    Raises
    ------
    ChildProcessError
        Raised if SCWRL failed to run.
    """
    if path:
        with open(pdb, "rb") as inf:
            pdb = inf.read()
    pdb = pdb.encode()
    sequence = sequence.encode()
    scwrl_path = str(scwrl_path)
    try:
        with tempfile.NamedTemporaryFile(
            delete=False
        ) as scwrl_tmp, tempfile.NamedTemporaryFile(
            delete=False
        ) as scwrl_seq, tempfile.NamedTemporaryFile(
            delete=False
        ) as scwrl_out:
            scwrl_tmp.write(pdb)
            scwrl_tmp.seek(0)  # Resets the buffer back to the first line
            scwrl_seq.write(sequence)
            scwrl_seq.seek(0)
            scwrl_command = f"{scwrl_path} -p {scwrl_path}.ini -i {scwrl_tmp.name} -o {scwrl_out.name} -s {scwrl_seq.name}"
            if rigid_rotamer_model:
                scwrl_command += " -v"
            if not hydrogens:
                scwrl_command += " -h"
            scwrl_std_out = subprocess.getoutput(scwrl_command)
            scwrl_out.seek(0)
            scwrl_pdb = scwrl_out.read()
    finally:
        os.remove(scwrl_tmp.name)
        os.remove(scwrl_out.name)
        os.remove(scwrl_seq.name)
    if not scwrl_pdb:
        raise ChildProcessError("SCWRL failed to run. SCWRL:\n{}".format(scwrl_std_out))
    return scwrl_std_out, scwrl_pdb.decode()


def pack_side_chains_scwrl(
    assembly: ampal.Assembly,
    sequences: t.List[str],
    scwrl_path: Path,
    rigid_rotamer_model: bool = True,
    hydrogens: bool = False,
) -> ampal.Assembly:
    """Packs side chains onto a protein structure.

    Parameters
    ----------
    assembly : AMPAL Assembly
        AMPAL object containing some protein structure.
    sequence : [str]
        A list of amino acid sequences in single-letter code for Scwrl to pack.
    rigid_rotamer_model : bool, optional
        If True, Scwrl will use the rigid-rotamer model, which is
        faster but less accurate.
    hydrogens : bool, optional
        If False, the hydrogens produced by Scwrl will be ommitted.

    Returns
    -------
    packed_structure : AMPAL Assembly
        A new AMPAL Assembly containing the packed structure, with
        the Scwrl score in the tags.
    """
    protein = [x for x in assembly if isinstance(x, ampal.Polypeptide)]
    total_seq_len = sum([len(x) for x in sequences])
    total_aa_len = sum([len(x) for x in protein])
    if total_seq_len != total_aa_len:
        raise ValueError(
            "Total sequence length ({}) does not match "
            "total Polypeptide length ({}).".format(total_seq_len, total_aa_len)
        )
    if len(protein) != len(sequences):
        raise ValueError(
            "Number of sequences ({}) does not match "
            "number of Polypeptides ({}).".format(len(sequences), len(protein))
        )
    scwrl_std_out, scwrl_pdb = run_scwrl(
        assembly.pdb,
        "".join(sequences),
        scwrl_path=scwrl_path,
        path=False,
        rigid_rotamer_model=rigid_rotamer_model,
        hydrogens=hydrogens,
    )
    packed_structure, scwrl_score = parse_scwrl_out(scwrl_std_out, scwrl_pdb)
    new_assembly = ampal.load_pdb(packed_structure, path=False)
    new_assembly.tags["scwrl_score"] = scwrl_score

    return new_assembly


def pack_sidechains(
    structure: ampal.Assembly, sequence: str, scwrl_path: Path
) -> ampal.Assembly:
    """
    Packs sequence of residues onto ampal assembly using SCWRL

    Parameters
    ----------
    structure: ampal.Assembly
        Ampal assembly to be saved
    sequence: str
        Sequence of amino acids

    Returns
    -------
    packed_structure: ampal.Assembly
        Packed structure with scwrl
    """
    return pack_side_chains_scwrl(
        assembly=structure,
        sequences=sequence,
        rigid_rotamer_model=False,
        scwrl_path=scwrl_path,
    )


def analyse_with_scwrl(
    pdb_to_seq: dict,
    pdb_to_assembly: dict,
    output_path: Path,
    suffix: str,
    scwrl_path: Path,
) -> (dict, dict):
    """
    Analyses rotamer prediction with SCWRL

    Parameters
    ----------
    pdb_to_seq: dict
        {pdb_code: sequence}
    pdb_to_assembly:
        {pdb_code: ampal_assembly}
    output_path: Path
        Path to save analysis to.
    suffix: str
        Additional information to add to file.

    Returns
    -------
    pdb_to_scores: dict
        Dict {pdb_code: scwrl_score}
    pdb_to_errors: dict
         Dict {pdb_code: Error}
    """
    pdb_to_scores = {}
    pdb_to_errors = {}
    # Loop through each PDB code and pack them with SCWRL:
    for pdb in tqdm(
        pdb_to_seq.keys(), desc=f"Packing sequence in PDB {suffix} with SCWRL"
    ):
        pdb_outpath = output_path / (pdb + "_" + suffix + ".pdb")
        if pdb_outpath.exists():
            error = f"PDB {pdb} at {pdb_outpath} already exists."
            pdb_to_errors[pdb] = error
        elif pdb[:4] in pdb_to_assembly.keys():
            try:
                # If there are more than one backbones, add their sequences up for SCWRL:
                if len(pdb_to_assembly[pdb[:4]].backbone) > 1:
                    pdb_to_seq[pdb] = [pdb_to_seq[pdb]] * len(pdb_to_assembly[pdb[:4]])
                # Else structure is already monomeric - no need to add sequences
                else:
                    pdb_to_seq[pdb] = [
                        pdb_to_seq[pdb]
                    ]  # Sequences need to be in list for SCWRL4
                # Attempt packing:
                try:
                    scwrl_structure = pack_sidechains(
                        pdb_to_assembly[pdb[:4]], pdb_to_seq[pdb], scwrl_path=scwrl_path
                    )
                    pdb_to_scores[pdb] = scwrl_structure.tags["scwrl_score"]
                    save_assembly_to_path(
                        structure=scwrl_structure,
                        output_dir=output_path,
                        name=pdb + suffix,
                    )
                except ValueError as e:
                    error = f"Attempted packing on structure {pdb}, but got {e}"
                    pdb_to_errors[pdb] = error
            except (ValueError, KeyError) as e:
                error = f"Attempted selecting backbone on structure {pdb}, but got {e}"
                pdb_to_errors[pdb] = error
            except ChildProcessError as e:
                error = f"Attempted selecting backbone on structure {pdb}, but SCWRL failed: {e}"
                pdb_to_errors[pdb] = error
        else:
            error = f"Error with structure {pdb}. Assembly not found."
            pdb_to_errors[pdb] = error
    # Saves errors to file:
    output_error_path = output_path / f"errors_scwrl{suffix}.csv"
    print(
        f"Got {len(pdb_to_errors)} errors when attempting to pack {len(pdb_to_seq)} sequences. Saved errors in file {output_error_path}"
    )
    with open(output_error_path, "w") as f:
        for pdb, err in pdb_to_errors.items():
            f.write(f"{pdb},{err}\n")
    return pdb_to_scores, pdb_to_errors


def save_assembly_to_path(
    structure: ampal.Assembly, output_dir: Path, name: str
) -> None:
    """
    Saves ampal assembly to specified path.

    Parameters
    ----------
    structure: ampal.Assembly
        Ampal assembly to be saved
    output_dir: Path
        Output Directory
    name: str
        Name of output File
    """
    # Save assembly to path:
    output_path = output_dir / (name + ".pdb")
    with open(output_path, "w") as f:
        f.write(structure.pdb)
