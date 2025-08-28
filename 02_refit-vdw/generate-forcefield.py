import collections
import logging
import pathlib

import click
import tqdm

import numpy as np

from openff.toolkit import ForceField, Molecule, Quantity
from openff.evaluator.datasets import PhysicalPropertyDataSet

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@click.command()
@click.option(
    "--input-forcefield",
    "-i",
    default="openff-2.2.1.offxml",
    help="The input force field file to modify",
)
@click.option(
    "--output-forcefield",
    "-o",
    default="forcefield/force-field.offxml",
    help="The output force field file to save",
)
@click.option(
    "--training-set",
    "-t",
    default="../01_download-data/physprop/final/output/training-set.json",
    help="The training set file to use for parameterization",
)
@click.option(
    "--n-properties",
    "-n",
    default=5,
    help="The minimum number of properties a vdw parameter must be associated with to be trained"
    "in the output force field",
)
def main(
    input_forcefield: str = "openff-2.2.1.offxml",
    output_forcefield: str = "forcefield/force-field.offxml",
    training_set: str = "../../01_download-data/physprop/final/output/training-set.json",
    n_properties: int = 5,
):
    forcefield = ForceField(input_forcefield)

    training_dataset = PhysicalPropertyDataSet.from_json(training_set)
    
    vdw_handler = forcefield.get_parameter_handler("vdW")

    # add a new vdW parameter, nitrogen not in a primary amine
    vdw_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#7!H2X3:1]",
            "epsilon": Quantity("0.1676915150424 kilocalorie / mole"),
            "rmin_half": Quantity("1.799798315098 angstrom"),
            "id": "n20_split1"
        },
        after="[#7:1]", # put specific parameter comes after more general parameter
    )

    label_counter = collections.Counter()

    for prop in tqdm.tqdm(training_dataset.properties):
        all_vdw_label_ids = set()
        for component in prop.substance.components:
            mol = Molecule.from_smiles(component.smiles, allow_undefined_stereo=True)
            labels = forcefield.label_molecules(mol.to_topology())[0]["vdW"]
            for parameter in labels.values():
                all_vdw_label_ids.add(parameter.id)

        for label_id in all_vdw_label_ids:
            label_counter[label_id] += 1

    for parameter in vdw_handler.parameters:
        if "tip3p" in parameter.id:
            continue  # skip water parameters
        property_count = label_counter.get(parameter.id, 0)
        logger.info(
            f"Parameter {parameter.id} {parameter.smirks} has {property_count} properties associated with it."
        )
        if property_count >= n_properties:
            # parameter.add_cosmetic_attribute("parameterize", "epsilon, rmin_half")
            # we want to keep epsilon non-zero
            # there are more elegant and continuous ways to do this, but we also want
            # something that renders as nonzero in a reasonable number of decimal places
            # so we arbitrarily set the minimum value to 1e-5
            # parameter.add_cosmetic_attribute("constrained_epsilon", parameter.epsilon)

            # set constrained_epsilon to inverse softplus
            eps_ = parameter.epsilon.m_as("kilocalories_per_mole")
            constrained_epsilon = np.log(np.exp(eps_ - 1e-5) - 1)
            parameter.add_cosmetic_attribute(
                "constrained_epsilon",
                f"{constrained_epsilon:.6f} * mole ** -1 * kilocalorie ** 1",
            )
            
            # I believe we just ignore units here -- see PR below
            # https://github.com/leeping/forcebalance/pull/281
            prm = f"PRM['vdW/Atom/constrained_epsilon/{parameter.smirks}']"
            parameter.add_cosmetic_attribute(
                "parameter_eval",
                # we need to avoid commas as they break the regex
                # use a softplus function bounded at 1e-5
                # numpy is imported much more frequently than math so use that
                f"epsilon=(np.log(1 + np.exp({prm})) + 1e-5)"
            )
            parameter.add_cosmetic_attribute("parameterize", "rmin_half, constrained_epsilon")

            logger.info(f"Training {parameter.id} {parameter.smirks}")


    output_forcefield = pathlib.Path(output_forcefield)
    output_forcefield.parent.mkdir(parents=True, exist_ok=True)
    forcefield.to_file(output_forcefield)
    logger.info(f"Saved force field to {output_forcefield}")


if __name__ == "__main__":
    main()
