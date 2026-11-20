from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import replace
from multiprocessing import cpu_count
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Tuple
from mxio import XYPair

import numpy
import vg

from mxproc import Analysis, reporting, parse_cluster
from mxproc.command import CommandFailed
from mxproc.common import StateType, StepType, show_warnings, short_path
from mxproc.common import generate_failure, Result, select_resolution, summarize_ranges
from mxproc.engines.xds import io, stats
from mxproc.engines.xds.indexing import auto_index
from mxproc.engines.xds.parser import XDSParser, pointless
from mxproc.xtal import (
    Experiment,
    Lattice,
    assess_cell_orientation,
    max_delta_angle,
    resolution_shells,
    score_dataset,
    score_strategy,
)
from mxproc.programs import raddose
from mxproc.log import logger


class XDSAnalysis(Analysis):
    prefix = 'xds'

    def get_extras(self, args: argparse.Namespace) -> dict:
        extras = {}
        if args.frames:
            extras.update(data_range=args.frames, spot_range=args.frames)

        if args.beam_center:
            extras.update(beam_center=XYPair(*args.beam_center))

        if args.anom is not None:
            extras.update(anomalous=args.anom)

        if args.min_sigma:
            extras.update(min_sigma=args.min_sigma)

        if args.max_sigma:
            extras.update(max_sigma=args.max_sigma)

        if args.resolution:
            extras.update(resolution_limit=args.resolution)

        if args.spacegroup:
            extras.update(lattice=Lattice(spacegroup=args.spacegroup))

        if args.cluster:
            extras.update(cluster={'mode': "SLURM", **args.cluster})
        elif default_cluster := os.getenv("CLUSTER_SPECS"):
            cluster_specs = parse_cluster(default_cluster)
            extras.update(cluster={'mode': "SLURM", **cluster_specs})
        elif os.environ.get("CLUSTER_NODES"):
            node_list = os.environ.get("CLUSTER_NODES", "").split()
            node_cores = int(os.environ.get("CLUSTER_CORES", cpu_count()))
            extras.update(cluster={
                'mode': "SSH",
                'nodes': len(node_list),
                'cpus': node_cores,
            })
        else:
            extras["cluster"] = {}
        return extras

    def initialize(self, **kwargs):
        results = {}
        logger.info('Working Directories:')
        self.prepare_experiment_directories()
        for experiment in self.experiments:
            directory = self.options.working_directories[experiment.identifier]
            spot_range = [(r[0], r[1]) for r in experiment.frames]
            io_options = {
                'data_range': experiment.frames,
                'spot_range': spot_range,
                'beam_center': experiment.detector_origin
            }
            io_options.update(self.options.extras)
            io.create_input_file(('ALL',), experiment, io.XDSParameters(**io_options), work_dir=directory)
            if directory != self.options.directory:
                path_str = short_path(directory, self.options.directory.parent)
            else:
                path_str = str(directory)
            logger.info_value(f'{experiment.name}', path_str)
            results[experiment.identifier] = Result(state=StateType.SUCCESS, details={"directory": directory})
        return results

    def find_spots(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            spot_range = [(r[0], r[1]) for r in experiment.frames]
            io_options = {
                'data_range': experiment.frames,
                'spot_range': spot_range,
                'beam_center': experiment.detector_origin
            }
            io_options.update(self.options.extras)
            work_dir = self.options.working_directories[experiment.identifier]
            job = io.create_input_file(
                ('XYCORR', 'INIT', 'COLSPOT'),
                experiment,
                io.XDSParameters(**io_options),
                work_dir=work_dir,
            )
            image_range = summarize_ranges(io_options['spot_range'])

            try:
                self.runner.run(
                    'xds_par',
                    work_dir,
                    desc=f'{experiment.name}: Finding strong spots in images {image_range}',
                    cluster=True,
                    job=job,
                )
                result = Result(state=StateType.SUCCESS, details=XDSParser.parse(str(work_dir / 'COLSPOT.LP')))
                io.save_spots(work_dir=work_dir)
            except CommandFailed as err:

                result = generate_failure(f"Command failed: {err}")
            results[experiment.identifier] = result
        return results

    def index(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            # skip if the find_spots step did not succeed
            if not self.get_step_result(experiment, StepType.SPOTS):
                continue

            work_dir = self.options.working_directories[experiment.identifier]

            result = generate_failure('')
            logger.info(f'{experiment.name}:')
            spot_range = [(r[0], r[1]) for r in experiment.frames]
            io_options = {
                'data_range': experiment.frames,
                'spot_range': spot_range,
                'anomalous': self.options.anomalous,
                'beam_center': experiment.detector_origin
            }
            io_options.update(**self.options.extras)

            # ignore lattice
            io_options.pop('lattice', None)

            result = auto_index(experiment, io_options, work_dir=work_dir, runner=self.runner)

            # show final unit cell parameters
            if result.get('point_groups'):
                point_groups = ", ".join(result.get('point_groups'))
                lattice = result.get('lattice', Lattice())
                sym_lattice = result.get('symmetry_lattice', Lattice())
                logger.info_value('- Reduced Cell', lattice.cell_text())
                logger.info_value('- Compatible Point Groups', point_groups)
                logger.info_value('- Apparent Symmetry', sym_lattice.name)
                logger.info_value('- Apparent Unit Cell', sym_lattice.cell_text())
                logger.info_value('- Unit-cell Volume', f"{sym_lattice.volume():0.1f} Å³")
                logger.info_value('- Estimated Unit Cell Content', f"{sym_lattice.estimate_content()} Residues")

            results[experiment.identifier] = result
            if result.state in [StateType.WARNING, StateType.FAILURE]:
                show_warnings(f"{result.state.name}: Possible Indexing Problems", result.messages)

        return results

    def strategy(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            work_dir = self.options.working_directories[experiment.identifier]
            index_result = self.get_step_result(experiment, StepType.INDEX)
            if not index_result: continue
            lattice = index_result.get('symmetry_lattice')
            spot_range = [(r[0], r[1]) for r in experiment.frames]
            io_options = {
                'data_range': experiment.frames,
                'spot_range': spot_range,
                'anomalous': self.options.anomalous,
                'lattice': lattice,
                'beam_center': experiment.detector_origin,
                'min_fraction': 0.10
            }
            io_options.update(**self.options.extras)
            io.filter_spots(work_dir=work_dir)
            job = io.create_input_file(
                ('IDXREF', 'DEFPIX', 'XPLAN'),
                experiment,
                io.XDSParameters(**io_options),
                work_dir=work_dir,
            )
            job_params = {'nodes': 1, 'tasks': 1, 'cpus': None}
            if job.cpus and job.tasks and job.nodes:
                job_params['cpus'] = job.cpus * job.tasks // max(1, job.nodes)
            single_node_job = replace(job, **job_params)

            try:
                self.runner.run(
                    'xds_par',
                    work_dir,
                    desc=f'{experiment.name}: Calculating optimal strategy',
                    check_files=('XPLAN.LP',),
                    cluster=True,
                    job=single_node_job,
                )
                details = XDSParser.parse(str(work_dir / 'XPLAN.LP'))
            except (FileNotFoundError, CommandFailed) as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                axis_near_beam, longest_axis, long_axis_ratio, warnings = assess_cell_orientation(
                    lattice,
                    index_result.get('cell_a_axis'),
                    index_result.get('cell_b_axis'),
                    index_result.get('cell_c_axis'),
                    index_result.get('beam_axis'),
                    index_result.get('rotation_axis'),
                )
                if warnings:
                    logger.warning_value(
                        "- Long Unit-cell axis is closest to beam axis!",
                        f'{longest_axis["name"]}={longest_axis["length"]:0.1f}Å, {longest_axis["beam"]:0.1f}°'
                    )
                else:
                    logger.info_value(
                        f'- Unit-Cell axis closest to beam',
                        f'{axis_near_beam["name"]}, {axis_near_beam["beam"]:0.1f}°'
                    )
                resolution_details = stats.determine_resolution(
                    experiment, work_dir=work_dir, runner=self.runner
                )

                # calculate maximum delta to avoid overlaps - see Dauter, Acta Cryst. (1999). D55, p1707
                mosaicity = index_result.get('quality.mosaicity', 0.1)
                max_delta = max_delta_angle(
                    resolution_details['desired_resolution'], axis_near_beam['length'], mosaicity
                )

                best_strategy = details['strategies'][-1]
                score = score_strategy(
                    completeness=best_strategy['completeness'],
                    resolution=resolution_details['observed_resolution'],
                    long_axis_ratio=long_axis_ratio,
                    mosaicity=mosaicity,
                    angle_error=index_result.get('quality.angle_error'),
                    pixel_error=index_result.get('quality.pixel_error'),
                    indexed_fraction=resolution_details['indexed_fraction'],
                )

                # run raddose and estimate exposure time, assumes crystal is the same size as the beam
                # uses beam shape and flux from beamline passed in as options.
                beam = self.options.get_beam()
                beam.wavelength = experiment.wavelength
                success, dose_info = raddose.get_dose(
                    crystal_size=beam.aperture,
                    total_range=best_strategy['total_angle'],
                    lattice=lattice,
                    beam=beam,
                    work_dir=work_dir,
                    runner=self.runner,
                )

                details.update({
                    'strategy': {
                        **best_strategy,
                        'resolution': resolution_details['desired_resolution'],
                        'max_resolution': resolution_details['maximum_resolution'],
                        'max_delta': max_delta,
                        'attenuation': 0.0,
                        'exposure_rate': dose_info.get('exposure_rate', -1),
                        'exposure_rate_worst': dose_info.get('exposure_rate_worst', -1),
                        'total_exposure': dose_info.get('total_exposure', -1),
                        'total_exposure_worst': dose_info.get('total_exposure_worst', -1),
                    },
                    'quality': {
                        'mosaicity': mosaicity,
                        'resolution': resolution_details['observed_resolution'],
                        'long_axis': long_axis_ratio,
                        'score': score,
                        'warnings': warnings
                    },
                    'dose': dose_info
                })
                result = Result(state=StateType.SUCCESS, details=details)
                self.show_strategy(result)
            results[experiment.identifier] = result
        return results

    def integrate(self, **kwargs):
        results = {}
        for experiment in self.experiments:
            work_dir = self.options.working_directories[experiment.identifier]

            if not self.get_step_result(experiment, StepType.INDEX):
                continue

            logger.info(f'{experiment.name}:')
            spot_range = [(r[0], r[1]) for r in experiment.frames]
            io_options = {
                'data_range': experiment.frames,
                'spot_range': spot_range,
                'anomalous': self.options.anomalous,
            }
            io_options.update(**self.options.extras)
            io_options.pop('lattice', None)     # ignore lattice parameter if specified,

            range_text = summarize_ranges(io_options['data_range'])

            previous_integration = self.get_step_result(experiment, StepType.INTEGRATE)
            if self.options.optimize and previous_integration is not None:
                # copy parameter to new file
                shutil.copy(work_dir / 'GXPARM.XDS', work_dir / 'XPARM.XDS')

                # inject parameters to extra options
                io_options.update(
                    refl_range=previous_integration.get('parameters.refl_range'),
                    refl_range_esd=previous_integration.get('parameters.refl_range_esd'),
                    divergence=previous_integration.get('parameters.divergence'),
                    divergence_esd=previous_integration.get('parameters.divergence_esd'),
                )

            job = io.create_input_file(
                ('DEFPIX', 'INTEGRATE', 'CORRECT'),
                experiment,
                io.XDSParameters(**io_options),
                work_dir=work_dir,
            )

            try:
                self.runner.run(
                    'xds_par',
                    work_dir,
                    desc=f'- Integrating frames {range_text}',
                    check_files=['INTEGRATE.LP', 'CORRECT.LP'],
                    cluster=True,
                    job=job,
                )
                integration = XDSParser.parse(str(work_dir / 'INTEGRATE.LP'))
                correction = XDSParser.parse(str(work_dir / 'CORRECT.LP'))
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                details = {
                    'parameters': integration['parameters'],
                    'lattices': correction['lattices'],
                    'quality': {
                        'batches': integration['batches'],
                        'frames': integration['frames'],
                        'statistics': correction['statistics'],
                        'summary': correction['summary'],
                        'errors': correction['errors'],
                    },
                }
                result = Result(state=StateType.SUCCESS, details=details)
            results[experiment.identifier] = result

        return results

    def symmetry(self, **kwargs):
        results = {}
        reference = self.experiments[0]
        symmetry_experiments = []

        for experiment in self.experiments:
            work_dir = self.options.working_directories[experiment.identifier]

            if self.get_step_result(experiment, StepType.INTEGRATE):
                logger.info(f'{experiment.name}:')
                symmetry_experiments.append(experiment)
                try:
                    self.runner.run(
                        'pointless xdsin INTEGRATE.HKL xmlout pointless.xml',
                        work_dir,
                        desc=f'- Determining symmetry',
                        check_files=['pointless.xml'],
                    )
                    details = pointless.parse_pointless(str(work_dir / 'pointless.xml'))
                    experiment.lattice = details['lattice']
                except CommandFailed as err:
                    result = generate_failure(f"Command failed: {err}")
                else:
                    result = Result(state=StateType.SUCCESS, details=details)
                    lattice_message = f'{experiment.lattice.name} - #{experiment.lattice.spacegroup}'
                    logger.info_value(f'- Recommended {details["type"]}', lattice_message)
                    if experiment.lattice.spacegroup > reference.lattice.spacegroup:
                        reference = experiment
                results[experiment.identifier] = result

        for experiment in symmetry_experiments:
            if experiment.identifier not in results:
                continue

            result = self.apply_symmetry(experiment, reference)
            result.details['symmetry'] = results[experiment.identifier].details
            results[experiment.identifier] = result

            if result.state == StateType.SUCCESS:
                logger.info_value(f'- Refined Cell:', result.get('lattice').cell_text())
                self.show_quality(result.details['quality'])

        return results

    def apply_symmetry(self, experiment: Experiment, reference: Experiment) -> Result:
        """
        Apply specified symmetry to the experiment and reindex the data
        :param experiment: Experiment instance
        :param reference: Reference experiment
        """
        logger.info(f'{experiment.name}:')
        integrate_result = self.get_step_result(experiment, StepType.INTEGRATE)
        target_lattice = self.options.extras.get('lattice', reference.lattice)
        if target_lattice != reference.lattice:
            self.settings['symmetry_method'] = 'manually chosen'
        else:
            self.settings['symmetry_method'] = 'automatically assigned'

        reindex_lattice, reindex_matrix = io.find_lattice(target_lattice, integrate_result.details['lattices'])
        if reindex_lattice:
            directory = self.options.working_directories[experiment.identifier]
            if reference != experiment:
                reference_directory = self.options.working_directories[reference.identifier]
                reference_data = Path(os.path.relpath((reference_directory / "XDS_ASCII.HKL"), directory))
            else:
                reference_data = None

            try:
                spot_range = [(r[0], r[1]) for r in experiment.frames]
                io_options = {
                    "data_range": experiment.frames,
                    "spot_range": spot_range,
                    "reference": reference_data,
                }
                io_options.update(**self.options.extras)
                io_options.update(lattice=reindex_lattice, reindex=reindex_matrix)
                job = io.create_input_file(
                    ('CORRECT',), experiment, io.XDSParameters(**io_options), work_dir=directory
                )

                job_params = {'nodes': 1, 'tasks': 1, 'cpus': None}
                if job.cpus and job.tasks and job.nodes:
                    job_params['cpus'] = job.cpus * job.tasks // max(1, job.nodes)
                single_node_job = replace(job, **job_params)

                self.runner.run(
                    'xds_par',
                    directory,
                    desc=f'- Applying symmetry {reindex_lattice.name} - #{reindex_lattice.spacegroup}',
                    cluster=True,
                    job=single_node_job,
                )
                self.runner.run(
                    'echo "XDS_ASCII.HKL" | xdsstat 20 3 > XDSSTAT.LP',
                    directory,
                    desc=f'- Gathering extra statistics',
                )
                correction = XDSParser.parse(str(directory / 'CORRECT.LP'))
                parameters = XDSParser.parse(str(directory / 'GXPARM.XDS'))
                stats_tables = XDSParser.parse(str(directory / 'XDSSTAT.LP'))
            except CommandFailed as err:
                result = generate_failure(f"Command failed: {err}")
            else:
                resolution, resolution_method = select_resolution(
                    correction['statistics'], manual=self.options.extras.get('resolution_limit')
                )
                self.settings['resolution_method'] = resolution_method
                details = {
                    'parameters': parameters,
                    'lattice': Lattice(parameters['sg_number'], *parameters['unit_cell']),
                    'lattices': correction['lattices'],
                    'quality': {
                        'frames': stats_tables['frame_statistics'],
                        'differences': stats_tables['diff_statistics'],
                        'statistics': correction['statistics'],
                        'summary': {
                            **correction['quality'], **correction['summary'],
                            "i_sigma_a": correction['correction']['parameters']['i_sigma_a'],
                            "resolution": resolution,
                            "resolution_method": resolution_method
                        },
                        'inner_shell': correction['statistics'][0],
                        'outer_shell': correction['statistics'][-1],
                        'errors': correction['errors'],
                    },
                    'wilson': correction['wilson'],
                    'correction': correction['correction'],
                }
                details['quality']['summary']['score'] = self.score(details['quality'])
                result = Result(state=StateType.SUCCESS, details=details)
        else:
            logger.warning(f'{experiment.name} not compatible with selected spacegroup #{target_lattice.spacegroup}')
            result = generate_failure(f"Incompatible Lattice type: {target_lattice.character}")
        return result

    def scale(self, **kwargs):
        scalable_experiments = {
            experiment.name: experiment
            for experiment in self.experiments
            if self.get_step_result(experiment, StepType.SYMMETRY)
        }
        if not scalable_experiments:
            logger.error('Skipping! No datasets available to scale!')
            return {}

        anom_option = self.options.anomalous
        resolution_limit = self.options.extras.get('resolution_limit', 0.0)

        scale_configs = []
        quality_contrib = defaultdict(list)
        self.options.merge &= len(scalable_experiments) > 1
        if self.options.merge:
            method = "Combined Output"
            # prepare merge parameters
            inputs = []
            resolutions = []
            for name, experiment in scalable_experiments.items():
                directory = self.options.working_directories[experiment.identifier]
                symmetry_result = self.get_step_result(experiment, StepType.SYMMETRY)
                quality_contrib["XSCALE.HKL"].append(symmetry_result.get('quality.summary'))
                input_file = str(Path(os.path.relpath(directory / "XDS_ASCII.HKL", self.options.directory)))
                resolution = symmetry_result.get('quality.summary.resolution')
                inputs.append({'input_file': input_file, 'resolution': resolution})
                resolutions.append(resolution)

            # one configuration for merging
            scale_configs.append({
                'anomalous': anom_option,
                'shells': resolution_shells(max(resolution_limit, min(resolutions))),
                'output_file': "XSCALE.HKL",
                'inputs': inputs,
            })
            backup_files('XSCALE.HKL', 'XSCALE.LP')
        else:
            method = "Separate Output"
            for name, experiment in scalable_experiments.items():
                directory = self.options.working_directories[experiment.identifier]
                symmetry_result = self.get_step_result(experiment, StepType.SYMMETRY)
                input_file = str(Path(os.path.relpath(directory / "XDS_ASCII.HKL", self.options.directory)))
                output_file = str(Path(os.path.relpath(directory / "XSCALE.HKL", self.options.directory)))
                resolution = symmetry_result.get('quality.summary.resolution')
                quality_contrib[output_file].append(symmetry_result.get('quality.summary'))
                scale_configs.append({
                    'anomalous': anom_option,
                    'shells': resolution_shells(max(resolution, resolution_limit)),
                    'output_file': output_file,
                    'inputs': [{'input_file': input_file, 'resolution': resolution}],
                })

        try:
            job = io.write_xscale_input(scale_configs, work_dir=self.options.directory, **self.options.extras)
            single_node_job = replace(job, nodes=1, tasks=1)
            self.runner.run(
                'xscale_par',
                self.options.directory,
                desc=f'- Scaling {len(scalable_experiments)} dataset(s) for {method}',
                cluster=True,
                job=single_node_job,
            )
            scaling = XDSParser.parse_xscale(work_dir=self.options.directory)
        except CommandFailed as err:
            result = generate_failure(f'Command Failed: {err}')
            results = {
                experiment.identifier: result
                for experiment in scalable_experiments.values()
            }
        else:
            # check twinning and prepare details
            details = {}
            for key, section in scaling.items():

                resolution, resolution_method = select_resolution(
                    section['statistics'], manual=self.options.extras.get('resolution_limit')
                )
                self.settings['resolution_method'] = resolution_method
                quality = self.assess_quality(section['output_file'])

                # xtriage completeness is a fraction
                if quality.get('summary', {}):
                    quality['summary']['completeness'] *= 100
                    quality['summary']['anom_completeness'] *= 100

                # carry forward some quality parameters from symmetry step
                # as averages of contributing datasets
                extra_stats = {
                    k: numpy.mean([d.get(k, 0.0) for d in quality_contrib[key]])
                    for k in ('pixel_error', 'angle_error', 'mosaicity', 'i_sigma_a')
                }
                details[key] = {
                    'output': section['output_file'],
                    'lattice': Lattice(section['spacegroup'], *section['unit_cell']),
                    'quality': {
                        'statistics': section['statistics'],
                        'summary': {
                            **section['summary'],
                            **quality.get('summary', {}),
                            **extra_stats,
                            'resolution': resolution,
                            'resolution_method': resolution_method,
                        },
                        'inner_shell': section['statistics'][0],
                        'outer_shell': section['statistics'][-1],
                        'twinning': quality.get('twinning', {}),
                    },
                    'wilson': section['wilson'],
                    'datasets': section['datasets'],
                    'correlations': section['correlations'],
                }
                details[key]['quality']['summary']['score'] = self.score(details[key]['quality'])
                logger.info_value('Final Data Quality', key)
                self.show_quality(details[key]['quality'])

            # for merging, all experiments get the single section
            exportable = defaultdict(list)
            results = {}
            for name, experiment in scalable_experiments.items():
                work_dir = self.options.working_directories[experiment.identifier]
                if self.options.merge:
                    scale_file = 'XSCALE.HKL'
                else:
                    scale_file = (work_dir / 'XSCALE.HKL').relative_to(self.options.directory)
                expt_details = details[str(scale_file)]
                results[experiment.identifier] = Result(state=StateType.SUCCESS, details=expt_details)
                exportable[expt_details['output']].append(name)

            self.settings['outputs'] = dict(exportable)

        return results

    def assess_quality(self, data_file: str | Path) -> dict:
        """
        Check the data quality for the specified file and return a dictionary of results
        :param data_file: reflection file
        """

        try:
            data_path = Path(data_file)
            if not data_path.is_absolute():
                data_path = self.options.directory / data_path
            log_file = str(data_path.parent / "XTRIAGE.LP")
            command = f'phenix.xtriage {data_file} loggraphs=True nproc=8 log={log_file}'
            self.runner.run(
                command,
                self.options.directory,
                desc=f'- Assessing quality of "{data_file}"',
                check_files=[log_file],
            )
            quality = XDSParser.parse(log_file)
        except CommandFailed:
            quality = {}

        return quality

    def export(self, **kwargs):
        outputs = {}
        work_dir = self.options.directory
        for scaled_input, names in self.settings.get('outputs', {}).items():
            # get common name and strip special characters from ending.
            common_name = re.sub(r"[-_+]$", '', os.path.commonprefix(names))
            prefix = common_name or "final"

            # SHELX
            io.write_xdsconv_input({
                'format': 'SHELX',
                'anomalous': self.options.anomalous,
                'input_file': scaled_input,
                'prefix': prefix,
            }, work_dir=work_dir)
            self.runner.run('xdsconv', work_dir, desc=f'- Exporting SHELX reflection file "{prefix}-shelx.hkl"')
            outputs['shelx'] = f"{prefix}-shelx.hkl"

            # MTZ
            command = f'phenix.reflection_file_converter  {scaled_input} --write_unmerged --mtz={prefix}.mtz'
            self.runner.run(command, work_dir, desc=f'- Exporting Unmerged MTZ reflection file "{prefix}.mtz"')
            outputs['mtz'] = f"{prefix}.mtz"

            # Unmerged MTZ
            command = f'phenix.reflection_file_converter  {scaled_input} --generate_r_free_flags --mtz={prefix}-freer.mtz'
            self.runner.run(command, work_dir, desc=f'- Exporting Unmerged MTZ reflection file with FreeR "{prefix}-freer.mtz"')
            outputs['freer'] = f"{prefix}-freer.mtz"

        self.settings['reflection_files'] = outputs

    @staticmethod
    def show_quality(info: dict):
        """
        Display the data quality details
        :param info: Quality dictionary
        """
        quality = Result(state=StateType.SUCCESS, details=info)

        i_sigma_a = quality.get('summary.i_sigma_a')
        completeness = quality.get('summary.completeness')
        resolution = quality.get('summary.resolution')
        lo_r_meas = quality.get('inner_shell.r_meas')
        lo_i_sigma = quality.get('inner_shell.i_sigma')
        resolution_method = quality.get('summary.resolution_method')
        score = quality.get('summary.score')

        logger.info_value(f'- Inner-Shell R-meas:', f'{lo_r_meas:0.1f} %')
        logger.info_value(f'- Inner-Shell I/Sigma:', f'{lo_i_sigma:0.1f}')
        logger.info_value(f'- Completeness:', f'{completeness:0.1f} %')
        logger.info_value(f'- Resolution limit (by {resolution_method}):', f'{resolution:0.2f}')
        if i_sigma_a is not None:
            logger.info_value(f'- I/Sigma(I) Asymptote [ISa]:', f'{i_sigma_a:0.1f}')

        logger.info_value(f'- Data Quality Score', f"{score:0.2}")

    @staticmethod
    def show_strategy(info: Result):
        """
        Display the screening strategy results
        :param info: strategy
        """
        start_angle = info.get('strategy.start_angle', 0)
        angle_range = info.get('strategy.total_angle', 180.0)
        completeness = info.get('strategy.completeness')
        max_delta = info.get('strategy.max_delta')
        multiplicity = info.get('strategy.multiplicity')
        mosaicity = info.get('quality.mosaicity')
        observed_resolution = info.get('quality.resolution')
        desired_resolution = info.get('strategy.resolution')
        exposure_rate = info.get('strategy.exposure_rate')
        exposure_rate_worst = info.get('strategy.exposure_rate_worst')
        total_exposure = info.get('strategy.total_exposure')
        total_exposure_worst = info.get('strategy.total_exposure_worst')
        score = info.get('quality.score')

        logger.info_value(f'- Starting angle', f'{start_angle:0.1f}°')
        logger.info_value(f'- Total Oscillation Range', f'{angle_range:0.1f}°')
        logger.info_value(f'- Predicted Completeness:', f'{completeness:0.1f} %')
        logger.info_value(f'- Predicted Multiplicity:', f'{multiplicity:0.1f}')
        logger.info_value(f'- Crystal Mosaicity:', f'{mosaicity:0.2f}')
        logger.info_value(f'- Maximum Delta to avoid overlaps', f'{max_delta:0.2f}°')
        logger.info_value(f'- Resolution of Diffraction', f'{observed_resolution:0.2f} Å')
        logger.info_value(f'- Recommended High-Resolution Limit', f'{desired_resolution:0.2f} Å')
        logger.info_value(
            f'- Maximum Exposure Rate (recommended, worst-case)',
            f'{exposure_rate:0.1f}°/s, {exposure_rate_worst:0.1f}°/s'
        )
        logger.info_value(
            f'- Maximum Total Exposure (recommended, worst-case)',
            f'{total_exposure:0.0f} s, {total_exposure_worst:0.0f} s'
        )
        logger.info_value(f'- Screening Score', f"{score:0.2f}")

    @staticmethod
    def score(info: dict) -> float:
        """
        Calculate and return a data quality score for the quality dictionary.
        """
        return score_dataset(info)

    def report(self, **kwargs):
        payload = reporting.build_report_payload(self)
        for html_path in reporting.generate_reports(payload):
            report_file = short_path(html_path, self.options.directory.parent)
            logger.info(f'- HTML report: {report_file}')

