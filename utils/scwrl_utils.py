"""
This util file is necessary as ISAMBARD does not integrate well with SCWRL4

It contains several workarounds. See the original under isambard.modelling.scwrl
"""
import os
import re
import subprocess
import tempfile

import ampal


def parse_scwrl_out(scwrl_std_out, scwrl_pdb):
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


def run_scwrl(pdb, sequence, path=True, rigid_rotamer_model=True, hydrogens=False):
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
        with open(pdb, "r") as inf:
            pdb = inf.read()
    pdb = pdb.encode()
    sequence = sequence.encode()
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
            scwrl_command = f"/Users/leo/scwrl4/Scwrl4 -p /Users/leo/scwrl4/Scwrl4.ini -i {scwrl_tmp.name} -o {scwrl_out.name} -s {scwrl_seq.name}"
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
    assembly, sequences, rigid_rotamer_model=True, hydrogens=False
):
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
        path=False,
        rigid_rotamer_model=rigid_rotamer_model,
        hydrogens=hydrogens,
    )
    packed_structure, scwrl_score = parse_scwrl_out(scwrl_std_out, scwrl_pdb)
    new_assembly = ampal.load_pdb(packed_structure, path=False)
    new_assembly.tags["scwrl_score"] = scwrl_score

    return new_assembly
