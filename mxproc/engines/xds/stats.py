import numpy
import pandas
from matplotlib import pyplot as plt

MAX_DETECTOR_COVERAGE = 0.85    # Diffraction should cover up to 85% of the detector surface


def determine_resolution(expt):
    """
    For a given experiment, calculate the diffraction resolution of the reflections and the detector
    limit, as well as the suggested maximum resolution
    :param expt: Experiment
    """
    spots = numpy.loadtxt('SPOT.XDS')
    indexed = numpy.abs(spots[:, -3:]).sum(axis=1) > 0.0

    df = pandas.DataFrame(spots[indexed, :], columns=['x', 'y', 'angle', 'intensity', 'h', 'k', 'l'])
    df['dx'] = (df['x'] - expt.detector_origin.x) * expt.pixel_size.x
    df['dy'] = (df['y'] - expt.detector_origin.y) * expt.pixel_size.y

    max_x = (expt.detector_size.x - expt.detector_origin.x) * expt.pixel_size.x
    max_y = (expt.detector_size.y - expt.detector_origin.y) * expt.pixel_size.y

    max_radius = min(max_x, max_y)
    max_angle = 0.5 * numpy.arctan(max_radius / expt.distance)
    detector_d = expt.wavelength / (2 * numpy.sin(max_angle))

    df['radius'] = numpy.sqrt(df['dy'] ** 2 + df['dx'] ** 2)
    df['angle'] = 0.5 * numpy.arctan(df['radius'] / expt.distance)
    df['resolution'] = expt.wavelength / (2 * numpy.sin(df['angle']))
    df['inv_d_sqr'] = 1/df['resolution']**2

    observed_d = numpy.percentile(df['resolution'], 1)
    desired_d = ((observed_d**-2)/MAX_DETECTOR_COVERAGE)**-0.5

    return {
        'observed_resolution': observed_d,
        'maximum_resolution': detector_d,
        'desired_resolution': desired_d,
        'indexed_fraction': indexed.mean()
    }
