from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from mxio import DataSet
from parsefire import parser

XDS_ZCBF_LIB = os.getenv('XDS_ZCBF_LIB', '/cmcf_apps/xtal/dozor/xds-zcbf.so')

DOZOR_DATA = """!
detector {detector}
library {zcbf_lib}
nx {x_size}
ny {y_size}
pixel {pixel_size:0.4f}
exposure {exposure:0.4f}
spot_size 2
spot_level 3
detector_distance {distance:0.3f}
X-ray_wavelength {wavelength:0.4f}
fraction_polarization 0.990
pixel_min 3
pixel_max {count_cutoff}
orgx {x_center}
orgy {y_center}
oscillation_range {delta_angle:0.4f}
image_step 1
starting_angle {start_angle:0.4f}
first_image_number {index}
number_images 1
name_template_image {name_template}
end
"""

DOZOR_ENTRY = "<int:index> | <int:bragg_spots> <float:score> <float:resolution> <float:avg_signal>"

DOZOR_OUTPUT = {
    "root": {
        "sections": {
            "summary": {
                "fields": [DOZOR_ENTRY]
            }
        }
    }
}


def data_signal(frame_path: str | Path) -> dict:
    """
    Perform signal strength analysis on a file
    :param frame_path: full path to file
    :return: Dictionary of results
    """
    result = {
        'ice_rings': 0, 'resolution': 50, 'total_spots': 0, 'bragg_spots': 0, 'signal_avg': 0, 'signal_min': 0,
        'signal_max': 0, 'score': 0.0
    }

    start_time = time.time()
    dat_file = Path(frame_path).with_suffix('.dat')
    dset = DataSet.new_from_file(frame_path)
    detector = dset.frame.detector.replace('Dectris', '').replace(' ', '').strip().lower()

    with open(dat_file, 'w') as handle:
        handle.write(DOZOR_DATA.format(
            zcbf_lib=XDS_ZCBF_LIB,
            detector=detector,
            x_size=dset.frame.size.x,
            y_size=dset.frame.size.y,
            pixel_size=dset.frame.pixel_size.x,
            exposure=dset.frame.exposure,
            distance=dset.frame.distance,
            wavelength=dset.frame.wavelength,
            count_cutoff=dset.frame.cutoff_value,
            x_center=dset.frame.center.x,
            y_center=dset.frame.center.y,
            delta_angle=dset.frame.delta_angle,
            start_angle=dset.frame.start_angle,
            index=dset.index,
            name_template=str(dset.directory / dset.glob)
        ))

    args = ['dozor', str(dat_file)]
    output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    info = parser.parse_text(DOZOR_OUTPUT, output.decode('utf-8'))['summary']
    info['frame_number'] = dset.index
    info['duration'] = 1000*(time.time() - start_time)
    result.update(info)

    return result


def process_signal() -> int:
    """
    Run the functionality previously provided by the standalone `bin/auto.xds` script.

    This function parses `sys.argv` just like the original script and either submits
    a SLURM job via SSH or runs the command locally.
    """
    parser = argparse.ArgumentParser(description="Analyze Frames and estimate resolution.")
    parser.add_argument("images", type=str, nargs='+', help="Images")

    args = parser.parse_args()
    results = [
        data_signal(image)
        for image in args.images
    ]
    print(results)
    return 0
