"""
This script executes a distributed ForceBalance optimization using the OpenFF Evaluator.
It sets up the environment, prepares the ForceBalance input, and runs the optimization
on a SLURM cluster using Dask for distributed computing.

It does the following steps:
- sets up ForceBalance options and writes them to `targets/phys-prop/options.json`
- checks if a previous run can be continued, and prepares the restart
- renames any previous log files to avoid overwriting
- sets up a Dask SLURM backend with the specified resources
- starts an Evaluator server to handle requests
- runs ForceBalance with the specified input file and logs the output to force_balance.log

This run generates the following output:
- force_balance.log: The log file for the ForceBalance run
- results/: the directory with resulting force field
- optimize.tmp: directory with training properties calculated from intermediate force fields
- optimize.sav: the saved state of the ForceBalance optimization, if it was continued
- optimize.bak: a backup of the ForceBalance optimization state
- working-directory/: the directory with the working files for the Evaluator server
- worker-logs/: the directory with the logs for the Dask workers
- conda-env.yaml: the conda environment used for the run
- slurm-*.out: the SLURM output files for the workers
"""

import logging
import sys
import subprocess
import os
import pathlib
import re
import shutil
import typing
import click
from click_option_group import optgroup


from openff.units import unit
from openff.evaluator.server import EvaluatorServer
from openff.evaluator.backends import QueueWorkerResources, ComputeResources
from openff.evaluator.backends.dask import DaskSLURMBackend
from openff.evaluator.storage import LocalFileStorage

from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.properties import Density, EnthalpyOfMixing
from forcebalance.evaluator_io import Evaluator_SMIRNOFF

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _remove_previous_files():
    """Remove any previous files that might interfere with the run."""
    restart_files = [
        # "optimize.tmp",
        "optimize.bak",
        "optimize.sav",
        "result",
        "worker-logs",
        "working-data",
    ]
    for restart_file in restart_files:

        path = pathlib.Path(restart_file)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            os.unlink(path)
        else:
            continue

        logger.info(f"Removing {restart_file}.")


def write_options(
    n_molecules: int = 1000,
    port: int = 8021,
    output_file: str = "targets/phys-prop/options.json"
):
    """
    Write the options for the ForceBalance optimization to a JSON file.
    This includes the equilibration properties, estimation options, and weights.
    The options are used by the Evaluator to run the optimization.
    """
    options = Evaluator_SMIRNOFF.OptionsFile()
    options.connection_options.server_port = port

    # set up request options
    potential_energy = EquilibrationProperty()
    potential_energy.observable_type = ObservableType.PotentialEnergy
    potential_energy.n_uncorrelated_samples = 15

    density = EquilibrationProperty()
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 15

    options.estimation_options.calculation_layers = ["PreequilibratedSimulationLayer"]
    density_schema = Density.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        equilibration_max_iterations=5, # 5 * 200 ps = 1 ns
        equilibration_error_on_failure=False,
        n_uncorrelated_samples=100,
        max_iterations=5, # 10 ns
        error_on_failure=False,

    )

    dhmix_schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        equilibration_max_iterations=5, # 5 * 200 ps = 1 ns
        equilibration_error_on_failure=False,
        n_uncorrelated_samples=100,
        max_iterations=5, # 10 ns
        error_on_failure=False

    )

    options.estimation_options.add_schema(
        "PreequilibratedSimulationLayer", "Density", density_schema
    )
    options.estimation_options.add_schema(
        "PreequilibratedSimulationLayer", "EnthalpyOfMixing", dhmix_schema
    )

    # weights and denominators
    options.data_set_path = "training-set.json"
    options.weights["Density"] = 1.0
    options.weights["EnthalpyOfMixing"] = 1.0
    options.denominators["Density"] = 0.05 * unit.grams / unit.milliliters
    options.denominators["EnthalpyOfMixing"] = 1.6 * unit.kilojoules / unit.mole

    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        f.write(options.to_json())
    logger.info(f"Wrote options to {output_file}.")


def _prepare_restart(
    input_file: str = "optimize.in",
    target_name: str = "phys-prop",
):
    """
    Check for correct completed data and remove incomplete data from previous runs.

    This reads the input file to determine the maximum number of iterations,
    and checks for completed iterations. If any incomplete iterations are found,
    it removes them from the target directory.
    """

    # parse input file
    with open(input_file, "r") as file:
        content = file.read()

    # get max iterations
    try:
        maxstep = int(re.search(r"maxstep\s*(\d+)", content).groups()[0])
    except AttributeError:
        maxstep = 100  # default
    
    # check how many iterations have been completed
    incomplete_iteration_subdirs = [
        f"iter_{iteration:04d}"
        for iteration in range(maxstep)
    ]
    while incomplete_iteration_subdirs:
        subdir = incomplete_iteration_subdirs[0]
        target_directory = pathlib.Path("optimize.tmp") / target_name / subdir
        # has objective.p been written?
        if (target_directory / "objective.p").exists():
            # all's good
            incomplete_iteration_subdirs = incomplete_iteration_subdirs[1:]
        else:
            # check if mvals.txt and force-field.offxml exist
            if (
                (target_directory / "mvals.txt").exists()
                and (target_directory / "force-field.offxml").exists()
            ):
                # we can leave these, but need to remove any later directories
                incomplete_iteration_subdirs = incomplete_iteration_subdirs[1:]
            else:
                # start deleting from here
                break
    for subdir in incomplete_iteration_subdirs:
        target_directory = pathlib.Path("optimize.tmp") / target_name / subdir
        if target_directory.exists():
            shutil.rmtree(target_directory)
            logger.info(f"Removing {target_directory}.")


def rename_log_file(log_file):
    """Rename the log file to have the correct extension"""
    counter = 0
    original_log_file = pathlib.Path(log_file)
    log_file = pathlib.Path(log_file)
    if not log_file.exists():
        return
    while log_file.exists():
        counter += 1
        log_file = pathlib.Path(
            f"{original_log_file.stem}_{counter}{original_log_file.suffix}"
        )
    original_log_file.rename(log_file)
    logger.info(f"Renamed existing {original_log_file} to {log_file}.")


@click.command()
@click.option(
    "--input",
    "input_file",
    type=str,
    default="optimize.in",
    help="The input file for ForceBalance",
)
@click.option(
    "--log",
    "log_file",
    type=str,
    default="force_balance.log",
    help="The output log file for ForceBalance",
)
@optgroup.group("Server configuration")
@optgroup.option(
    "--port",
    type=int,
    default=8000,
    help="The port for the server",
)
@optgroup.option(
    "--working-directory",
    type=str,
    default="working-directory",
    help="The working directory for the server",
)
@optgroup.option(
    "--enable-data-caching/--no-enable-data-caching",
    default=True,
    help="Enable data caching",
)
@optgroup.option(
    "--continue-run",
    type=bool,
    default=True,
    help="Continue a previous run",
)
@optgroup.group("Distributed configuration")
@optgroup.option(
    "--n-min-workers",
    type=int,
    default=1,
    help="The minimum number of workers to keep running",
)
@optgroup.option(
    "--n-max-workers",
    type=int,
    default=1,
    help="The maximum number of workers to start running",
)
@optgroup.option(
    "--queue",
    "queue_name",
    type=str,
    default="free-gpu",
    help="The queue name to start the workers on",
)
@optgroup.option(
    "--n-threads",
    type=int,
    default=1,
    help="The number of threads per worker",
)
@optgroup.option(
    "--n-gpus",
    type=int,
    default=1,
    help="The number of GPUs per worker",
)
@optgroup.option(
    "--memory-per-worker",
    type=int,
    default=4,
    help="The memory per worker in GB",
)
@optgroup.option(
    "--walltime",
    type=str,
    default="8:00",
    help=(
        "The walltime for the workers. "
        "This should be whatever format the computer expects. "
        "Note that some computers will interpret '8:00' as 8 hours, "
        "while others will interpret it as 8 minutes."
    ),
)
@optgroup.option(
    "--gpu-toolkit",
    type=click.Choice(["CUDA", "OpenCL"]),
    default="CUDA",
    help="The GPU toolkit to use",
)
@optgroup.option(
    "--conda-env",
    type=str,
    default="naglmbis-refit",
    help="The conda environment to use",
)
@click.option(
    "--extra-script-option",
    "extra_script_options",
    type=str,
    multiple=True,
    help="Extra options to pass to the script",
)
def main(
    input_file: str = "optimize.in",
    log_file: str = "force_balance.log",
    # run args
    port: int = 8000,
    working_directory: str = "working-directory",
    enable_data_caching: bool = True,
    continue_run: bool = True,

    # distributed args
    n_min_workers: int = 1,
    n_max_workers: int = 1,
    queue_name: str = "free-gpu",
    n_threads: int = 1,
    n_gpus: int = 1,
    memory_per_worker: int = 4, # GB
    walltime: str = "8:00",
    gpu_toolkit: typing.Literal["CUDA", "OpenCL"] = "CUDA",
    conda_env: str = "sage-2.3.0-vdw",

    extra_script_options: list[str] = []
):
    # prepare ForceBalance arguments
    force_balance_arguments = ["ForceBalance.py", input_file]

    # check if we should continue
    _continue = continue_run
    if _continue and pathlib.Path("optimize.sav").exists():
        _prepare_restart(input_file)
        force_balance_arguments = ["ForceBalance.py", "--continue", input_file]
    else:
        # _remove_previous_files()
        pass

    write_options(port=port)

    # set up queue worker resources
    worker_resources = QueueWorkerResources(
        number_of_threads=n_threads,
        number_of_gpus=n_gpus,
        preferred_gpu_toolkit=ComputeResources.GPUToolkit[gpu_toolkit],
        per_thread_memory_limit=memory_per_worker * unit.gigabyte,
        wallclock_time_limit=walltime,
    )

    # set up dask backend
    backend = DaskSLURMBackend(
        minimum_number_of_workers=n_min_workers,
        maximum_number_of_workers=n_max_workers,
        resources_per_worker=worker_resources,
        queue_name=queue_name,
        setup_script_commands=[
            "source ~/.bashrc",
            "micromamba activate " + conda_env,
            "micromamba env export > conda-env.yaml",
        ],
        extra_script_options=extra_script_options,
        adaptive_interval="1000ms",
    )

    # rename any previous log files
    rename_log_file(log_file)

    # actually run ForceBalance
    with backend:
        server = EvaluatorServer(
            calculation_backend=backend,
            working_directory=working_directory,
            port=port,
            enable_data_caching=enable_data_caching,
            delete_working_files=False, # for debugging -- set true if too large
            # we need to cache for speed reasons
            # otherwise just querying the storage backend can take 10+ hours for 1k properties
            storage_backend=LocalFileStorage(cache_objects_in_memory=True)
        )
        with server:
            with open(log_file, "w") as file:
                subprocess.check_call(force_balance_arguments, stderr=file, stdout=file)


if __name__ == "__main__":
    main()
