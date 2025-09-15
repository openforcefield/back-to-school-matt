"""
Benchmark a single property from a dataset using a specified force field.
"""

import os
import pickle
import click
import pathlib
import sys

from loguru import logger

from openff.units import unit
from openff.evaluator.backends.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.client import RequestOptions
from openff.evaluator.backends import ComputeResources
from openff.evaluator.storage.localfile import LocalFileStorage

from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.server.server import EvaluatorServer

from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.toolkit import ForceField

logger.remove()
logger.add(sys.stdout)

@click.command()
@click.option(
    "--input-path",
    "-i",
    default="../../01_download-data/physprop/final/output/validation-set.json",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Path to the input dataset.",
)
@click.option(
    "--property-index",
    "-p",
    default=0,
    type=int,
    help="Index of the property to evaluate.",
)
@click.option(
    "--storage-directory",
    "-s",
    default="stored_data",
    type=click.Path(exists=True, dir_okay=True),
    help="Storage directory",
)
@click.option(
    "--forcefield",
    "-ff",
    type=str,
    default="../../forcefields/openff-2.2.1.offxml",
    help="Path to the forcefield file.",
)
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="validation",
    help="Path to the output directory.",
)
@click.option(
    "--replicate",
    "-r",
    type=int,
    default=1,
    help="Replicate ID to run.",
)
@click.option(
    "--base-port",
    "-bp",
    type=int,
    default=8100,
    help="Base port to use for the server.",
)
@click.option(
    "--options-file",
    "-of",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default="request-options.json",
    help="Path to the request options file.",
)
def main(
    input_path: str,
    property_index: int,
    forcefield: str,
    output_directory: str,
    storage_directory: str,
    replicate: int = 1,
    base_port: int = 8100,
    options_file: str = "request-options.json",
    
):
    dataset = PhysicalPropertyDataSet.from_json(input_path)
    logger.info(f"Loaded dataset with {len(dataset.properties)} properties from {input_path}")
    logger.info(f"Property index {property_index} ID is {dataset.properties[property_index].id}")
    small_dataset = PhysicalPropertyDataSet()
    small_dataset.add_properties(dataset.properties[property_index])
    logger.info(f"Property ID: {small_dataset.properties[0].id}")
    
    ff = ForceField(forcefield)
    logger.info(f"Loaded force field from {forcefield}")
    # check this has constraints
    handler = ff.get_parameter_handler("Constraints")
    assert len(handler.parameters) == 3


    ff_name = pathlib.Path(forcefield).stem
    force_field_source = SmirnoffForceFieldSource.from_object(ff)

    options = RequestOptions.from_json(options_file)
    logger.info(f"Loaded request options from {options_file}")

    output_directory = pathlib.Path(output_directory)
    storage_directory = pathlib.Path(storage_directory)
    storage_directory = LocalFileStorage(
        root_directory=str(storage_directory.resolve()),
        cache_objects_in_memory=True,
    )
    logger.info(f"Loaded {len(storage_directory._cached_retrieved_objects)} into memory")
    
    output_directory = output_directory / f"rep-{replicate}" / ff_name
    output_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(output_directory)

    pickle_file = f"prop-{property_index:04d}.pkl"
    results_file = f"prop-{property_index:04d}.json"

    if pathlib.Path(results_file).exists():
        logger.info(f"{results_file} already exists")
        return

    port = base_port + property_index
    logger.info(f"Starting server on port {port}")

    with DaskLocalCluster(
        number_of_workers=1,
        resources_per_worker=ComputeResources(
            number_of_gpus=1,
            preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
        )
    ) as calculation_backend:
        server = EvaluatorServer(
            calculation_backend=calculation_backend,
            working_directory="working-directory",
            delete_working_files=True,
            enable_data_caching=False,
            storage_backend=storage_directory,
            port=port,
        )
        with server:
            client = EvaluatorClient(
                connection_options=ConnectionOptions(server_port=port)
            )
            request, error = client.request_estimate(
                small_dataset,
                force_field_source,
                options
            )

            assert error is None
            results, exception = request.results(synchronous=True, polling_interval=30)

    assert exception is None

    logger.info(f"Simulation complete")
    logger.info(f"# estimated: {len(results.estimated_properties)}")
    logger.info(f"# unsuccessful: {len(results.unsuccessful_properties)}")
    logger.info(f"# exceptions: {len(results.exceptions)}")
    #pickle_file.parent.mkdir(exist_ok=True, parents=True)

    # below is only necessary for debugging -- so commented out for now.
    # with open(pickle_file, "wb") as f:
    #     pickle.dump(results, f)
    # logger.info(f"Results dumped to {pickle_file}")

    results.json(results_file, format=True)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
