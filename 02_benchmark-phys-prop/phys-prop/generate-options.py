import pickle
import click

from openff.units import unit
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.client import RequestOptions

from openff.evaluator.backends import ComputeResources, QueueWorkerResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.backends.dask import DaskSLURMBackend

from openff.evaluator.client import EvaluatorClient, RequestOptions, ConnectionOptions
from openff.evaluator.server.server import EvaluatorServer
from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.utils.observables import ObservableType

from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.toolkit import ForceField


@click.command()
@click.option(
    "--n-molecules",
    "-n",
    default=1000,
    help="Number of molecules in the simulation.",
)
@click.option(
    "--output-file",
    "-o",
    default="request-options.json",
    help="Output file for the request options.",
)
def main(
    n_molecules: int = 1000,
    output_file: str = "request-options.json"
):

    potential_energy = EquilibrationProperty()
    potential_energy.observable_type = ObservableType.PotentialEnergy
    potential_energy.n_uncorrelated_samples = 50

    density = EquilibrationProperty()
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 50

    dhmix_schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        n_uncorrelated_samples=200, # at least 200 samples to calculate observable
        equilibration_max_iterations=5, # max 1 ns equilibration
        max_iterations=2, # max 4 ns production
        error_on_failure=False,

    )
    density_schema = Density.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
        equilibration_error_tolerances=[potential_energy, density],
        n_uncorrelated_samples=200, # at least 200 samples to calculate observable
        equilibration_max_iterations=5, # max 1 ns equilibration
        max_iterations=2, # max 4 ns production
        error_on_failure=False,
    )

    options = RequestOptions()
    options.calculation_layers = ["PreequilibratedSimulationLayer"]
    options.add_schema("PreequilibratedSimulationLayer", "Density", density_schema)
    options.add_schema("PreequilibratedSimulationLayer", "EnthalpyOfMixing", dhmix_schema)
    options.json(file_path=output_file, format=True)
    print(f"Request options saved to {output_file}")


if __name__ == "__main__":
    main()
