# OpenFOAM Template Placeholder

Place a validated 2D `rhoCentralFoam` scramjet inlet case here.

The intended workflow is:

1. Clone this template per row in `cfd/sweep_manifest.csv`.
2. Substitute freestream Mach, density/pressure/temperature, and ramp angle.
3. Run `rhoCentralFoam`.
4. Sample pressure and temperature fields onto a fixed Cartesian grid.
5. Export the arrays into the HDF5 schema documented in the root README.
