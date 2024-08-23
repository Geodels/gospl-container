from mpi4py import MPI

from time import process_time
from .flex import projectUTM as _projectUTM
from .flex import computeFlex as _computeFlex
from .flex import projectSpherical as _projectSpherical

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Model(_projectUTM, _computeFlex, _projectSpherical):

    def __init__(self, filename=None, dsin=None, data=None, fileout=None,
                 young=65.e9, nu=0.25, rho_m=3300.0, rho_s=2300.0, resg=2,
                 verbose=False):

        self.verbose = verbose
        _projectSpherical.__init__(self, fileout, size)

        # In case scattered data are provided, we first build an xarray
        if data is not None:
            dsin = _projectSpherical.buildLonLatMesh(self, data, res=0.25,
                                                     nghb=4)

        # Initialize model run
        _projectUTM.__init__(self, filename, dsin)
        _computeFlex.__init__(self, young, nu, rho_m, rho_s, resg)

        return

    def updateFlex(self, data):

        self.ds = _projectSpherical.updateData(self, data)

    def runFlex(self, gospl=True):

        tstart = process_time()

        # Split the global grids into tiles and project in UTM coordinates
        _projectUTM.projectData(self)

        tstep = process_time()
        # Compute flexural isostasy locally and reproject to lon/lat
        _computeFlex.getFlex(self)
        comm.Barrier()
        if rank == 0 and self.verbose:
            print(
                "--- Perform flexural isostasy (%0.02f s)"
                % (process_time() - tstep),
                flush=True
            )

        # Combine local UTM grids
        if rank == 0:
            tstep = process_time()
            _projectSpherical.projectLonLat(self)
            if self.verbose:
                print(
                    "--- Perform spherical projection (%0.02f s)"
                    % (process_time() - tstep),
                    flush=True
                )
                print(
                    "Total runtime (%0.02f s)"
                    % (process_time() - tstart),
                    flush=True
                )
            if gospl:
                _projectSpherical.interp2umesh(self)

        comm.Barrier()

        return
