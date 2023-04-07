import xml.dom.minidom
from typing import Sequence
from mxproc.xtal import Lattice, SPACEGROUP_NAMES


def get_spacegroup(element: xml.dom.minidom.Element) -> dict:
    """
    Extract Spacegroup dictionary from xml element representing a spacegroup solution
    :param element: XML element of
    :return: dictionary of parameters
    """

    number = int(element.getElementsByTagName('SGnumber')[0].firstChild.nodeValue)
    probability = float(element.getElementsByTagName('TotalProb')[0].firstChild.nodeValue)
    screw_axis_prob = float(element.getElementsByTagName('SysAbsProb')[0].firstChild.nodeValue)
    reindex_list = element.getElementsByTagName('ReindexMatrix')[0].firstChild.nodeValue.split()
    reindex = list(map(int, reindex_list))
    reindex_matrix = (
        reindex[0], reindex[3], reindex[6], 0,
        reindex[1], reindex[4], reindex[7], 0,
        reindex[2], reindex[5], reindex[8], 0
    )
    spacegroup = {
        'name': SPACEGROUP_NAMES.get(number, 'P1'), 'number': number, 'probability': probability,
        'sys_abs_prob': screw_axis_prob, 'reindex_matrix': reindex_matrix,
        'reindex_operator': element.getElementsByTagName('ReindexOperator')[0].firstChild.nodeValue
    }
    return spacegroup


def update_solution(number: int, solution: dict) -> dict:
    """
    Up the symmetry solution to the specified spacegroup
    :param number: requested spacegroup  number
    :param solution:  symmetry solution
    :return: spacegroup dictionary
    """
    if solution['number'] == number:
        return solution
    else:
        for candidate in solution['candidates']:
            if candidate['number'] == number:
                solution.update(**candidate)
                solution['lattice'].spacegroup = number
                return solution


def parse_pointless(filename: str) -> dict:
    """
    Parse Pointless XML file
    :param filename: Name of file
    """

    doc = xml.dom.minidom.parse(filename)
    solution_element = doc.getElementsByTagName('BestSolution')[0]
    solution = get_spacegroup(solution_element)
    solution_type = doc.getElementsByTagName('BestSolutionType')[0].firstChild.nodeValue

    # extract all the tested candidates
    candidate_spacegroups = [get_spacegroup(element) for element in doc.getElementsByTagName('Spacegroup')]

    # extract the unit cell
    lattice_element = doc.getElementsByTagName('BestCell')[0]
    cell = lattice_element.getElementsByTagName('cell')[0]
    solution_lattice = Lattice(
        spacegroup=solution['number'],
        a=float(cell.getElementsByTagName('a')[0].firstChild.nodeValue),
        b=float(cell.getElementsByTagName('b')[0].firstChild.nodeValue),
        c=float(cell.getElementsByTagName('c')[0].firstChild.nodeValue),
        alpha=float(cell.getElementsByTagName('alpha')[0].firstChild.nodeValue),
        beta=float(cell.getElementsByTagName('beta')[0].firstChild.nodeValue),
        gamma=float(cell.getElementsByTagName('gamma')[0].firstChild.nodeValue),
    )

    solution.update(type=solution_type.title(), lattice=solution_lattice, candidates=candidate_spacegroups)

    return solution
