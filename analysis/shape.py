import typing as t

import ampal
import numpy as np


def tag_packing_density(
    structure: t.Union[ampal.Polymer, ampal.Assembly], radius: float = 7
) -> None:
    """
    Function from ISAMBARD written by 'Kathryn L. Shelley'

    Calculates the packing density of each non-hydrogen atom in a polymer
    or assembly.

    An atom's packing density is a measure of the number of atoms within
    its local environment. There are several different methods of
    calculating packing density; we use atomic contact number [1], which
    is the number of non-hydrogen atoms within a specified radius (default
    7 A [1]).

    Parameters
    ----------
    structure : ampal.Polymer or ampal.Assembly
        The structure to be tagged.
    radius : float, optional
        The radius (in Angstroms) within which to count atoms. Default is 7 A.

    References
    ----------
    .. [1] Weiss MS (2007) On the interrelationship between atomic
       displacement parameters (ADPs) and coordinates in protein
       structures. *Acta Cryst.* D**63**, 1235-1242.
    """

    if not type(structure).__name__ in ["Polymer", "Assembly"]:
        raise ValueError(
            "Contact order can only be calculated for a polymer or an assembly."
        )

    atoms_list = [atom for atom in list(structure.get_atoms()) if atom.element != "H"]
    atom_coords_array = np.array([atom.array for atom in atoms_list])

    for index, atom in enumerate(atoms_list):
        distances = np.sqrt(
            np.square(atom_coords_array[:, :] - atom_coords_array[index, :]).sum(axis=1)
        )
        # Subtract 1 to correct for the atom itself being counted
        atom.tags["packing density"] = np.sum(distances < radius) - 1


def calculate_packing_density(
    assembly: ampal.Assembly, atom_filter: str
) -> t.List[float]:
    """
    Extracts packing density from ampal polypeptide

    Parameters
    ----------
    assembly: ampal.Assembly
        Ampal assembly to extract packing density from.
    atom_filter: str
        Atom filter function to use. Can be "backbone", "ca" or "all"

    Returns
    -------
    packdensity: t.List[float]
        List of packing density for each residue in assembly or polypeptide
    """
    if atom_filter == "backbone":
        filter_set = ("N", "CA", "C", "O")
    elif atom_filter == "ca":
        filter_set = "CA"
    elif atom_filter == "all":
        filter_set = None
    else:
        raise ValueError(
            f"Atom Filter function {atom_filter} not in (backbone, ca, all)"
        )

    packdensity = []
    tag_packing_density(assembly)
    # Extract iddt for each residue
    for res in assembly[0]:
        # All the atoms have the same bfactor (iddt) so select first atom:
        current_density = -1
        for atom in res:
            if filter_set:
                if atom.res_label in filter_set:  # Only backbone atoms
                    if current_density == -1:
                        current_density = atom.tags["packing density"]
                    else:
                        current_density = (
                            current_density + atom.tags["packing density"]
                        ) / 2
            else:
                if atom.res_label != "H":
                    if current_density == -1:
                        current_density = atom.tags["packing density"]
                    else:
                        current_density = (
                            current_density + atom.tags["packing density"]
                        ) / 2

        packdensity.append(current_density)
    return packdensity
