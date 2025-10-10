"""
This script removes cosmetic attributes from a force field.
"""
import click
import pathlib
import logging

from openff.toolkit import ForceField
from openff.toolkit.utils.exceptions import SMIRNOFFSpecError, DuplicateParameterError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@click.command()
@click.option(
    "--input",
    "-i",
    "input_forcefield",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the input force field file.",
)
@click.option(
    "--output",
    "-o",
    "output_forcefield",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    required=True,
    help="The path to the output force field file.",
)
def main(
    input_forcefield: str = "../04_fit-forcefield/nor4/fb-fit/result/optimize/force-field.offxml",
    output_forcefield: str = "openff-2.2.1-rc1.offxml",
):

    # Delete cosmetic attributes
    force_field = ForceField(
        input_forcefield,
        allow_cosmetic_attributes=True
    )
    # remove all cosmetic attributes
    for _, handler in force_field._parameter_handlers.items():
        for parameter in handler.parameters:
            for attribute_to_remove in [
                'parameterize',
                'constrained_epsilon',
                'parameter_eval',
            ]:
                try:
                    parameter.delete_cosmetic_attribute(attribute_to_remove)
                except AttributeError:
                    pass
    
    # Save unconstrained version
    output_forcefield = pathlib.Path(output_forcefield)
    stem = output_forcefield.stem
    unconstrained_path = output_forcefield.parent / f"{stem}_unconstrained.offxml"
    force_field.to_file(unconstrained_path)
    logger.info(f"Saved unconstrained force field to {unconstrained_path}")

    # Add constraints
    constraint_handler = force_field.get_parameter_handler('Constraints')
    constraint_param = constraint_handler.ConstraintType(smirks="[#1:1]-[*:2]", id="c1")
    try:
        constraint_handler.add_parameter(parameter=constraint_param,before=0)
    except DuplicateParameterError as error:
        if "[#1:1]-[*:2]" in str(error):
            logger.info("Generic H-bond constraint already exists, not adding")
        else:
            raise DuplicateParameterError from error

    force_field.to_file(output_forcefield)
    logger.info(f"Saved constrained force field to {output_forcefield}")


if __name__ == "__main__":
    main()
