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


@click.command()
@click.option(
    "--n-molecules",
    "-n",
    type=int,
    default=1000,
)
def main(
    n_molecules: int = 1000, 
):

    potential_energy = EquilibrationProperty()
    potential_energy.relative_tolerance = 0.05
    potential_energy.observable_type = ObservableType.PotentialEnergy
    potential_energy.n_uncorrelated_samples = 300

    density = EquilibrationProperty()
    density.relative_tolerance = 0.05
    density.observable_type = ObservableType.Density
    density.n_uncorrelated_samples = 300

    options = RequestOptions()
    options.calculation_layers = ["EquilibrationLayer"]
    density_schema = Density.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=[potential_energy, density],
        # every iteration is 200 ps
        max_iterations=1000, # go up to 200 ns
        error_on_failure=False,
    )

    dhmix_schema = EnthalpyOfMixing.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=[potential_energy, density],
        max_iterations=1000,
        error_on_failure=False,
    )

    # note: output frequency is every 10 ps.

    options.add_schema("EquilibrationLayer", "Density", density_schema)
    options.add_schema("EquilibrationLayer", "EnthalpyOfMixing", dhmix_schema)
    options.json("options.json", format=True) 

if __name__ == "__main__":
    main()

