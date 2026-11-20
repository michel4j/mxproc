from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from parsefire import parser
from mxproc import Experiment


CBF_LIB = os.getenv('XDS_ZCBF_LIB', shutil.which('xds-zcbf.so'))


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
number_images {num_images}
name_template_image {name_template}
end
"""

DOZOR_OUTPUT = {
    "table": [
        "<int:index> | <int:bragg_spots> <float:score> <float:resolution> <float:avg_signal>"
    ]
}


def data_resolution(expt: Experiment) -> list[float]:
    """
    Perform signal strength analysis on a file
    :param expt: full path to file
    :return: list of dictionary scores per image
    """

    detector = expt.detector.replace('Dectris', '').replace(' ', '').strip().lower()
    scores = []
    for start, end in expt.frames:
        num_images = end - start
        dat_file = Path(f'{expt.name}-{start}.dat')
        with open(dat_file, 'wt') as handle:
            handle.write(DOZOR_DATA.format(
                zcbf_lib=CBF_LIB or '',
                detector=detector,
                x_size=expt.detector_size.x,
                y_size=expt.detector_size.y,
                pixel_size=expt.pixel_size.x,
                exposure=expt.exposure,
                distance=expt.distance,
                wavelength=expt.wavelength,
                count_cutoff=expt.cutoff_value,
                x_center=expt.detector_origin.x,
                y_center=expt.detector_origin.y,
                delta_angle=expt.delta_angle,
                start_angle=expt.start_angle,
                index=start,
                num_images=num_images,
                name_template=str(expt.directory / expt.glob)
            ))

        args = ['dozor', str(dat_file)]
        try:
            proc = subprocess.run(args, capture_output=True, text=True)
            info = parser.parse_text(DOZOR_OUTPUT, proc.stdout)
        except FileNotFoundError:
            pass
        else:
            scores.extend([item['resolution'] for item in info])
    return scores
