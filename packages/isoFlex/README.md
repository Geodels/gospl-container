# isoFlex 


A wrapper around **[gFlex](https://github.com/awickert/gFlex)** to estimate global-scale flexural isostasy based on tiles distribution and projection in parallel.

> This is not the most elegant method, but will probably do the trick for now...

The globe is divided into 142 overriding tiles to account for UTM projection distortions and **[gFlex](https://github.com/awickert/gFlex)** is apply on each tile before reprojecting it back to either lon/lat or cartesian coordinates...

In **[gFlex](https://github.com/awickert/gFlex)** we use the finite difference (FD) method using the van Wees and Cloetingh (1994) ('vWC1994') option for the plate solution type.

The code can either be used in isolation by providing a data containing required information or called from another library. See below for some of the main input definitions.

### Dependencies and installation

+ numpy
+ scipy
+ xarray
+ rioxarray
+ pyproj
+ mpi4py
+ gflex

Installation:
```bash
cd isoFlex
pip install . 
```

alternatively

```bash
pip install --no-build-isolation -e .  
```

### Simple usage

An example is provided in the `example` folder:

```bash
mpirun -np 10 python3 runflex.py
```

where `runflex.py` is

```python
from isoflex.model import Model as iflex

model = iflex(filename='data/init_conditions.nc',
              fileout='data/cpt_flex.nc',
              verbose=True)

model.runFlex(False)
```

Model runtime on 10 CPUs:

```bash
> cd example
> mpirun -np 10 python3 runflex.py
--- Build cartesian tiles (1.00 seconds)
--- Perform flexural isostasy (9.41 s)
--- Perform spherical projection (1.26 s)
Total runtime (11.67 s)
```

### Options for initialization

Input dataset is specified by either `filename`, `dsin`, or `data`.

_Case 1:_ in case where `filename` is chosen, a netcdf file is required with the following variables should be provided:
   - 'erodep': erosion deposition thickness in metres,
   - 'te': elastic thickness in metres

_Case 2:_ similarly if a xarray dataset is given (`dsin`), it should contain the same variable names (i.e., 'erodep', 'te').
 
The resolution for the input dataset (either `filename` or `dsin`) needs to be in longitude/latitude and be 0.25 deg resolution.

```yaml
Dimensions:    (latitude: 721, longitude: 1441).

Coordinates:
   * latitude   (latitude) float64 6kB -90.0 -89.75 -89.5 ... 89.5 89.75 90.0
   * longitude  (longitude) float64 12kB -180.0 -179.8 -179.5 ... 179.8 180.0
Data variables:
     erodep     (latitude, longitude) float64 8MB ...
     te         (latitude, longitude) float64 8MB ...
```

_Case 3:_ optionally one might want to give scattered values distributed across the globe in cartesian coordinates.

In such a case the user needs to use the `data` variable as a numpy array of dimensions (n,5) where n is the number of records on the globe and 5 corresponds to the following 'X,Y,Z,erodep,te'.
Here, a Kd-tree interpolation will then be performed to map the variables in a regular lon/lat mesh.

### Additional parameters

+ The flexural response could be saved in a netcdf file defined in 'fileout'.
+ Young's Modulus `young` (default 65.e9),
+ Poisson's Ratio `nu` (default 0.25),
+ Mantle density `rho_m` (default 3300),
+ Sediment density `rho_s` (default 2300)
