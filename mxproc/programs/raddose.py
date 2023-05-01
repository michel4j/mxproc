import csv
import re

from typing import Tuple
from mxproc.command import run_command, CommandFailed
from mxproc.xtal import Lattice, Beam

DOSE_LIMIT = 30.0   # MGy, from PNAS  (2006) 103 (13) 4912-4917
SULFUR_CONTENT = 0.037
INPUT_TEMPLATE = """
Crystal
Type Cuboid             
Dimensions {crystal_size:0.0f} {crystal_size:0.0f} {crystal_size:0.0f}  
AbsCoefCalc  RD3D       
UnitCell   {cell_constant:0.1f} {cell_constant:0.1f} {cell_constant:0.1f}                  
NumMonomers  1                
NumResidues  {num_residues}               
ProteinHeavyAtoms S {num_sulphurs:0.0f}         
SolventFraction 0.5            

Beam
Type Gaussian             
Flux {beam_flux:0.1e}                 
FWHM {fwhm_x:0.0f} {fwhm_y:0.0f}                
Energy {energy:0.0f}           
Collimation Circular {aperture:0.0f} {aperture:0.0f}

Wedge 0 {total_range:0.0f}
ExposureTime 1 
"""


def fix_key(key: str) -> str:
    """
    Convert an arbitrary string to a proper variable name
    :param key:  string to convert
    """
    key = re.sub(r"[(\[].*?[)\]]", "", key).strip()
    key = re.sub(r"[-\s]", "_", key)
    return key.lower()


def load_csv_dict(filename: str) -> dict:
    """
    Load a dictionary from a CSV file. Only the first data row is returned, keys are converted to valid
    python variable names
    :param filename:
    """

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            info = row
            return {fix_key(key): float(value) for key, value in row.items()}


def get_dose(crystal_size: float, total_range: float, lattice: Lattice, beam: Beam) -> Tuple[bool, dict]:
    """
    Calculate the Average Diffraction weigthed dose and the maximum dose for 1 second exposure
    using Raddose3D

    :param crystal_size: Crystal size in microns
    :param total_range: Total oscillation range for wedge
    :param lattice: Lattice parameters
    :param beam: Beam parameters
    :return: Tuple of book, dict where the bool indicates if the run was successful
    """
    num_residues = lattice.estimate_content()
    num_sulphurs = SULFUR_CONTENT * num_residues
    cell_constant = lattice.volume()**(1/3)
    energy = 12.398/beam.wavelength
    with open('RAD.INP', 'w') as file:
        file.write(INPUT_TEMPLATE.format(
            crystal_size=crystal_size,
            cell_constant=cell_constant,
            num_residues=num_residues,
            num_sulphurs=num_sulphurs,
            beam_flux=beam.flux,
            fwhm_x=beam.fwhm_x, fwhm_y=beam.fwhm_y,
            energy=energy,
            aperture=beam.aperture,
            total_range=total_range
        ))

    try:
        run_command("raddose -i RAD.INP -p '' -o SummaryCSV:RADDOSE.CSV", "Estimating Radiation Dose")
        details = load_csv_dict("RADDOSE.CSV")
        success = True
    except (CommandFailed, FileNotFoundError):
        details = {}
        success = False
    else:
        total_exposure = DOSE_LIMIT / details['average_dwd']
        total_exposure_worst = DOSE_LIMIT / details['max_dose']
        exposure_rate = total_range / total_exposure
        exposure_rate_worst = total_range / total_exposure_worst
        details.update({
            "total_exposure": total_exposure,
            "total_exposure_worst": total_exposure_worst,
            "exposure_rate": exposure_rate,
            "exposure_rate_worst": exposure_rate_worst
        })

    return success, details

