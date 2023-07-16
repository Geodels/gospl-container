import os
import gc
import sys
import petsc4py
import numpy as np

from mpi4py import MPI
from time import process_time

if "READTHEDOCS" not in os.environ:
    from gospl._fortran import strataonesed
    from gospl._fortran import stratathreesed
    from gospl._fortran import stratafullsed

petsc4py.init(sys.argv)
MPIrank = petsc4py.PETSc.COMM_WORLD.Get_rank()
MPIsize = petsc4py.PETSc.COMM_WORLD.Get_size()
MPIcomm = petsc4py.PETSc.COMM_WORLD


class STRAMesh(object):
    """
    This class encapsulates all the functions related to underlying stratigraphic information. As mentionned previously, `gospl` has the ability to track different types of clastic sediment size and one type of carbonate (still under development). Sediment compaction in stratigraphic layers geometry and properties change are also considered.

    """

    def __init__(self):
        """
        The initialisation of `STRAMesh` class related to stratigraphic informations.
        """

        self.stratH = None
        self.stratF = None
        self.stratW = None
        self.stratZ = None
        self.stratC = None

        self.phiS = None
        self.phiF = None
        self.phiC = None

        return

    def readStratLayers(self):
        """
        When stratigraphic layers are turned on, this function reads any initial stratigraphic layers provided by within the YAML input file (key: `npstrata`).

        The following variables will be read from the file:

        - thickness of each stratigrapic layer `strataH` accounting for both erosion & deposition events.
        - proportion of fine sediment `strataF` contains in each stratigraphic layer.
        - proportion of weathered sediment `strataW` contains in each stratigraphic layer.
        - elevation at time of deposition, considered to be to the current elevation for the top stratigraphic layer `strataZ`.
        - porosity of coarse sediment `phiS` in each stratigraphic layer computed at center of each layer.
        - porosity of fine sediment `phiF` in each stratigraphic layer computed at center of each layer.
        - porosity of weathered sediment `phiW` in each stratigraphic layer computed at center of each layer.
        - proportion of carbonate sediment `strataC` contains in each stratigraphic layer if the carbonate module is turned on.
        - porosity of carbonate sediment `phiC` in each stratigraphic layer computed at center of each layer when the carbonate module is turned on.

        """

        if self.strataFile is not None:
            fileData = np.load(self.strataFile)
            stratVal = fileData["strataH"]
            self.initLay = stratVal.shape[1]
            self.stratNb += self.initLay

            # Create stratigraphic arrays
            self.stratH = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.stratH[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            stratVal = fileData["strataF"]
            self.stratF = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.stratF[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            stratVal = fileData["strataW"]
            self.stratW = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.stratW[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            stratVal = fileData["strataZ"]
            self.stratZ = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.stratZ[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            stratVal = fileData["phiS"]
            self.phiS = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.phiS[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            stratVal = fileData["phiF"]
            self.phiF = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.phiF[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            stratVal = fileData["phiW"]
            self.phiW = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.phiW[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            if self.carbOn:
                stratVal = fileData["strataC"]
                self.stratC = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
                self.stratC[:, 0 : self.initLay] = stratVal[
                    self.locIDs, 0 : self.initLay
                ]

                stratVal = fileData["phiC"]
                self.phiC = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
                self.phiC[:, 0 : self.initLay] = stratVal[self.locIDs, 0 : self.initLay]

            if self.memclear:
                del fileData, stratVal
                gc.collect()
        else:
            self.stratH = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.phiS = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)
            self.stratZ = np.zeros((self.lpoints, self.stratNb), dtype=np.float64)

        return

    def deposeStrat(self, stype):
        """
        Add deposition on top of an existing stratigraphic layer. The following variables will be recorded:

        - thickness of each stratigrapic layer `stratH` accounting for both erosion & deposition events.
        - proportion of fine sediment `stratF` contains in each stratigraphic layer.
        - proportion of weathered sediment `stratW` contains in each stratigraphic layer.
        - porosity of coarse sediment `phiS` in each stratigraphic layer computed at center of each layer.
        - porosity of fine sediment `phiF` in each stratigraphic layer computed at center of each layer.
        - porosity of weathered sediment `phiW` in each stratigraphic layer computed at center of each layer.
        - proportion of carbonate sediment `stratC` contains in each stratigraphic layer if the carbonate module is turned on.
        - porosity of carbonate sediment `phiC` in each stratigraphic layer computed at center of each layer when the carbonate module is turned on.

        :arg stype: sediment type (integer)
        """

        self.dm.globalToLocal(self.tmp, self.tmpL)
        depo = self.tmpL.getArray().copy()
        depo[depo < 1.0e-4] = 0.0
        if self.stratF is not None:
            fineH = self.stratH[:, self.stratStep] * self.stratF[:, self.stratStep]
        if self.stratW is not None:
            clayH = self.stratH[:, self.stratStep] * self.stratW[:, self.stratStep]
        if self.carbOn:
            carbH = self.stratH[:, self.stratStep] * self.stratC[:, self.stratStep]
        self.stratH[:, self.stratStep] += depo
        ids = np.where(depo > 0)[0]

        if stype == 0:
            self.phiS[ids, self.stratStep] = self.phi0s
        elif stype == 1:
            fineH[ids] += depo[ids]
            self.phiF[ids, self.stratStep] = self.phi0f
        elif stype == 2:
            carbH[ids] += depo[ids]
            self.phiC[ids, self.stratStep] = self.phi0c
        elif stype == 3:
            clayH[ids] += depo[ids]
            self.phiW[ids, self.stratStep] = self.phi0w

        if self.stratF is not None:
            self.stratF[ids, self.stratStep] = (
                fineH[ids] / self.stratH[ids, self.stratStep]
            )
            if self.memclear:
                del fineH
        if self.stratW is not None:
            self.stratW[ids, self.stratStep] = (
                clayH[ids] / self.stratH[ids, self.stratStep]
            )
            if self.memclear:
                del clayH
        if self.carbOn:
            self.stratC[ids, self.stratStep] = (
                carbH[ids] / self.stratH[ids, self.stratStep]
            )
            if self.memclear:
                del carbH

        # Cleaning arrays
        if self.memclear:
            del depo, ids
            gc.collect()

        return

    def _initialiseStrat(self):
        """
        This function initialise zeros thickness arrays in regions experiencing no erosion at a given iteration.
        """

        self.thCoarse = np.zeros(self.lpoints)
        if self.stratF is not None:
            self.thFine = np.zeros(self.lpoints)
        if self.stratW is not None:
            self.thClay = np.zeros(self.lpoints)
        if self.carbOn:
            self.thCarb = np.zeros(self.lpoints)

        return

    def erodeStrat(self):
        """
        This function removes eroded sediment thicknesses from the stratigraphic pile. The function takes into account the porosity values of considered lithologies in each eroded stratigraphic layers.

        It follows the following assumptions:

        - Eroded thicknesses from stream power law and hillslope diffusion are considered to encompass both the solid and void phase.
        - Only the solid phase will be moved dowstream by surface processes.
        - The corresponding deposit thicknesses for those freshly eroded sediments correspond to uncompacted thicknesses based on the porosity at surface given from the input file.
        """

        self.dm.globalToLocal(self.tmp, self.tmpL)
        ero = self.tmpL.getArray().copy()
        ero[ero > 0] = 0.0

        # Nodes experiencing erosion
        nids = np.where(ero < 0)[0]
        if len(nids) == 0:
            self._initialiseStrat()
            return

        # Cumulative thickness for each node
        self.stratH[nids, 0] += 1.0e6
        cumThick = np.cumsum(self.stratH[nids, self.stratStep :: -1], axis=1)[:, ::-1]
        boolMask = cumThick < -ero[nids].reshape((len(nids), 1))
        mask = boolMask.astype(int)

        if self.stratF is not None:
            # Get fine sediment eroded from river incision
            thickF = (
                self.stratH[nids, 0 : self.stratStep + 1]
                * self.stratF[nids, 0 : self.stratStep + 1]
            )
            # From fine thickness extract the solid phase that is eroded
            thFine = thickF * (1.0 - self.phiF[nids, 0 : self.stratStep + 1])
            thFine = np.sum((thFine * mask), axis=1)

        if self.stratW is not None:
            # Get weathered sediment eroded from river incision
            thickW = (
                self.stratH[nids, 0 : self.stratStep + 1]
                * self.stratW[nids, 0 : self.stratStep + 1]
            )
            # From weathered thickness extract the solid phase that is eroded
            thClay = thickW * (1.0 - self.phiW[nids, 0 : self.stratStep + 1])
            thClay = np.sum((thClay * mask), axis=1)

        # Get carbonate sediment eroded from river incision
        if self.carbOn:
            thickC = (
                self.stratH[nids, 0 : self.stratStep + 1]
                * self.stratC[nids, 0 : self.stratStep + 1]
            )
            # From carbonate thickness extract the solid phase that is eroded
            thCarb = thickC * (1.0 - self.phiC[nids, 0 : self.stratStep + 1])
            thCarb = np.sum((thCarb * mask), axis=1)
            # From sand thickness extract the solid phase that is eroded
            thickS = (
                self.stratH[nids, 0 : self.stratStep + 1] - thickF - thickC - thickW
            )
            thCoarse = thickS * (1.0 - self.phiS[nids, 0 : self.stratStep + 1])
            thCoarse = np.sum((thCoarse * mask), axis=1)
        else:
            # From sand thickness extract the solid phase that is eroded
            if self.stratF is not None:
                thickS = self.stratH[nids, 0 : self.stratStep + 1] - thickF - thickW
            else:
                thickS = self.stratH[nids, 0 : self.stratStep + 1]
            thCoarse = thickS * (1.0 - self.phiS[nids, 0 : self.stratStep + 1])
            thCoarse = np.sum((thCoarse * mask), axis=1)

        # Clear all stratigraphy points which are eroded
        cumThick[boolMask] = 0.0
        tmp = self.stratH[nids, : self.stratStep + 1]
        tmp[boolMask] = 0
        self.stratH[nids, : self.stratStep + 1] = tmp

        # Erode remaining stratal layers
        # Get non-zero top layer number
        eroLayNb = np.bincount(np.nonzero(cumThick)[0]) - 1
        eroVal = cumThick[np.arange(len(nids)), eroLayNb] + ero[nids]

        if self.stratF is not None:
            # Get thickness of each sediment type eroded in the remaining layer
            self.thFine = np.zeros(self.lpoints)
            # From fine thickness extract the solid phase that is eroded from this last layer
            tmp = (self.stratH[nids, eroLayNb] - eroVal) * self.stratF[nids, eroLayNb]
            tmp[tmp < 1.0e-8] = 0.0
            thFine += tmp * (1.0 - self.phiF[nids, eroLayNb])
            # Define the uncompacted fine thickness that will be deposited dowstream
            self.thFine[nids] = thFine / (1.0 - self.phi0f)
            self.thFine[self.thFine < 0.0] = 0.0

        if self.stratW is not None:
            # Get thickness of each sediment type eroded in the remaining layer
            self.thClay = np.zeros(self.lpoints)
            # From weathered thickness extract the solid phase that is eroded from this last layer
            tmp = (self.stratH[nids, eroLayNb] - eroVal) * self.stratW[nids, eroLayNb]
            tmp[tmp < 1.0e-8] = 0.0
            thClay += tmp * (1.0 - self.phiW[nids, eroLayNb])
            # Define the uncompacted weathered thickness that will be deposited dowstream
            self.thClay[nids] = thClay / (1.0 - self.phi0w)
            self.thClay[self.thClay < 0.0] = 0.0

        self.thCoarse = np.zeros(self.lpoints)
        if self.carbOn:
            # From carb thickness extract the solid phase that is eroded from this last layer
            self.thCarb = np.zeros(self.lpoints)
            tmp = (self.stratH[nids, eroLayNb] - eroVal) * self.stratC[nids, eroLayNb]
            tmp[tmp < 1.0e-8] = 0.0
            thCarb += tmp * (1.0 - self.phiC[nids, eroLayNb])
            # Define the uncompacted carbonate thickness that will be deposited dowstream
            self.thCarb[nids] = thCarb / (1.0 - self.phi0c)
            self.thCarb[self.thCarb < 0.0] = 0.0
            # From sand thickness extract the solid phase that is eroded from this last layer
            tmp = self.stratH[nids, eroLayNb] - eroVal
            tmp *= (
                1.0
                - self.stratC[nids, eroLayNb]
                - self.stratW[nids, eroLayNb]
                - self.stratF[nids, eroLayNb]
            )
            # Define the uncompacted sand thickness that will be deposited dowstream
            thCoarse += tmp * (1.0 - self.phiS[nids, eroLayNb])
            self.thCoarse[nids] = thCoarse / (1.0 - self.phi0s)
            self.thCoarse[self.thCoarse < 0.0] = 0.0
        else:
            # From sand thickness extract the solid phase that is eroded from this last layer
            tmp = self.stratH[nids, eroLayNb] - eroVal
            if self.stratF is not None:
                tmp *= 1.0 - self.stratF[nids, eroLayNb] - self.stratW[nids, eroLayNb]
            tmp[tmp < 1.0e-8] = 0.0
            # Define the uncompacted sand thickness that will be deposited dowstream
            thCoarse += tmp * (1.0 - self.phiS[nids, eroLayNb])
            self.thCoarse[nids] = thCoarse / (1.0 - self.phi0s)
            self.thCoarse[self.thCoarse < 0.0] = 0.0

        # Update thickness of top stratigraphic layer
        self.stratH[nids, eroLayNb] = eroVal
        self.stratH[nids, 0] -= 1.0e6
        self.stratH[self.stratH < 0] = 0.0
        self.phiS[self.stratH < 0] = 0.0
        if self.stratF is not None:
            self.stratF[self.stratH < 0] = 0.0
            self.phiF[self.stratH < 0] = 0.0
        if self.stratW is not None:
            self.stratW[self.stratH < 0] = 0.0
            self.phiW[self.stratH < 0] = 0.0
        if self.carbOn:
            self.stratC[self.stratH < 0] = 0.0
            self.phiC[self.stratH < 0] = 0.0

        self.thCoarse /= self.dt
        if self.stratF is not None:
            self.thFine /= self.dt
        if self.stratW is not None:
            self.thClay /= self.dt
        if self.carbOn:
            self.thCarb /= self.dt

        return

    def elevStrat(self):
        """
        This function updates the current stratigraphic layer elevation.
        """

        self.stratZ[:, self.stratStep] = self.hLocal.getArray()

        return

    def _depthPorosity(self, depth):
        """
        This function uses the depth-porosity relationships to compute the porosities for each lithology and then the solid phase to get each layer thickness changes due to compaction.

        .. note::

            We assume that porosity cannot increase after unloading.

        :arg depth: depth below basement for each sedimentary layer

        :return: newH updated sedimentary layer thicknesses after compaction
        """

        # Depth-porosity functions
        phiS = self.phi0s * np.exp(depth / self.z0s)
        phiS = np.minimum(phiS, self.phiS[:, : self.stratStep + 1])
        if self.stratF is not None:
            phiF = self.phi0f * np.exp(depth / self.z0f)
            phiF = np.minimum(phiF, self.phiF[:, : self.stratStep + 1])
            phiW = self.phi0w * np.exp(depth / self.z0w)
            phiW = np.minimum(phiW, self.phiW[:, : self.stratStep + 1])
        if self.carbOn:
            phiC = self.phi0c * np.exp(depth / self.z0c)
            phiC = np.minimum(phiC, self.phiC[:, : self.stratStep + 1])

        # Compute the solid phase in each layers
        if self.stratF is not None:
            tmpF = (
                self.stratH[:, : self.stratStep + 1]
                * self.stratF[:, : self.stratStep + 1]
            )
            tmpF *= 1.0 - self.phiF[:, : self.stratStep + 1]
            tmpW = (
                self.stratH[:, : self.stratStep + 1]
                * self.stratW[:, : self.stratStep + 1]
            )
            tmpW *= 1.0 - self.phiW[:, : self.stratStep + 1]

        if self.carbOn:
            tmpC = (
                self.stratH[:, : self.stratStep + 1]
                * self.stratC[:, : self.stratStep + 1]
            )
            tmpC *= 1.0 - self.phiC[:, : self.stratStep + 1]
            tmpS = (
                self.stratC[:, : self.stratStep + 1]
                + self.stratF[:, : self.stratStep + 1]
                + self.stratW[:, : self.stratStep + 1]
            )
            tmpS = self.stratH[:, : self.stratStep + 1] * (1.0 - tmpS)
            tmpS *= 1.0 - self.phiS[:, : self.stratStep + 1]
            solidPhase = tmpC + tmpS + tmpF + tmpW
        else:
            if self.stratF is not None:
                tmpS = self.stratH[:, : self.stratStep + 1] * (
                    1.0
                    - self.stratF[:, : self.stratStep + 1]
                    - self.stratW[:, : self.stratStep + 1]
                )
                tmpS *= 1.0 - self.phiS[:, : self.stratStep + 1]
                solidPhase = tmpS + tmpF + tmpW
            else:
                tmpS = self.stratH[:, : self.stratStep + 1]
                tmpS *= 1.0 - self.phiS[:, : self.stratStep + 1]
                solidPhase = tmpS

        # Get new layer thickness after porosity change
        if self.stratF is not None:
            tmpF = self.stratF[:, : self.stratStep + 1] * (
                1.0 - phiF[:, : self.stratStep + 1]
            )
            tmpW = self.stratW[:, : self.stratStep + 1] * (
                1.0 - phiW[:, : self.stratStep + 1]
            )
        if self.carbOn:
            tmpC = self.stratC[:, : self.stratStep + 1] * (
                1.0 - phiC[:, : self.stratStep + 1]
            )
            tmpS = (
                1.0
                - self.stratF[:, : self.stratStep + 1]
                - self.stratC[:, : self.stratStep + 1]
                - self.stratW[:, : self.stratStep + 1]
            )
            tmpS *= 1.0 - phiS[:, : self.stratStep + 1]
            tot = tmpS + tmpC + tmpF + tmpW
        else:
            if self.stratF is not None:
                tmpS = (
                    1.0
                    - self.stratF[:, : self.stratStep + 1]
                    - self.stratW[:, : self.stratStep + 1]
                ) * (1.0 - phiS[:, : self.stratStep + 1])
                tot = tmpS + tmpF + tmpW
            else:
                tot = 1.0 - phiS[:, : self.stratStep + 1]

        ids = np.where(tot > 0.0)
        newH = np.zeros(tot.shape)
        newH[ids] = solidPhase[ids] / tot[ids]
        newH[newH <= 0] = 0.0
        phiS[newH <= 0] = 0.0
        if self.stratF is not None:
            phiF[newH <= 0] = 0.0
            phiW[newH <= 0] = 0.0
        if self.carbOn:
            phiC[newH <= 0] = 0.0

        # Update porosities in each sedimentary layer
        self.phiS[:, : self.stratStep + 1] = phiS
        if self.stratF is not None:
            self.phiF[:, : self.stratStep + 1] = phiF
            self.phiW[:, : self.stratStep + 1] = phiW
        if self.carbOn:
            self.phiC[:, : self.stratStep + 1] = phiC

        if self.memclear:
            del phiS, solidPhase
            del ids, tmpS, tot
            if self.stratF is not None:
                del tmpF, phiF, tmpW, phiW
            if self.carbOn:
                del phiC, tmpC
            gc.collect()

        return newH

    def getCompaction(self):
        """
        This function computes the changes in sedimentary layers porosity and thicknesses due to compaction.

        .. note::

            We assume simple depth-porosiy relationships for each sediment type available in each layers.
        """

        t0 = process_time()
        topZ = np.vstack(self.hLocal.getArray())
        totH = np.sum(self.stratH[:, : self.stratStep + 1], axis=1)

        # Height of the sediment column above the center of each layer is given by
        cumZ = -np.cumsum(self.stratH[:, self.stratStep :: -1], axis=1) + topZ
        elev = np.append(topZ, cumZ[:, :-1], axis=1)
        zlay = np.fliplr(elev - np.fliplr(self.stratH[:, : self.stratStep + 1] / 2.0))

        # Compute lithologies porosities for each depositional layers
        # Get depth below basement
        depth = zlay - topZ

        # Now using depth-porosity relationships we compute the porosities
        newH = self._depthPorosity(depth)

        # Get the total thickness changes induced by compaction and
        # update the elevation accordingly
        dz = totH - np.sum(newH, axis=1)
        dz[dz <= 0] = 0.0
        self.hLocal.setArray(topZ.flatten() - dz.flatten())
        self.dm.localToGlobal(self.hLocal, self.hGlobal)

        # Update each layer thicknesses
        self.stratH[:, : self.stratStep + 1] = newH
        if self.memclear:
            del dz, newH, totH, topZ
            del depth, zlay, cumZ, elev
            gc.collect()

        if MPIrank == 0 and self.verbose:
            print(
                "Compute Lithology Porosity Values (%0.02f seconds)"
                % (process_time() - t0),
                flush=True,
            )

        return

    def stratalRecord(self, indices, weights, onIDs):
        """
        Once the interpolation has been performed, the following function updates the stratigraphic records based on the advected mesh.

        The function relies on 3 fortran subroutines (for loop performance purposes):

        1. strataonesed
        2. stratathreesed
        3. stratafullsed

        :arg indices: indices of the closest nodes used for interpolation
        :arg weights: weights based on the distances to closest nodes
        :arg onIDs: index of nodes remaining at the same position.

        """

        # Get local stratal dataset after displacements
        loc_stratH = self.stratH[:, : self.stratStep]
        loc_stratZ = self.stratZ[:, : self.stratStep]
        loc_phiS = self.phiS[:, : self.stratStep]
        if self.stratF is not None:
            loc_stratF = self.stratF[:, : self.stratStep]
            loc_phiF = self.phiF[:, : self.stratStep]
        if self.stratW is not None:
            loc_stratW = self.stratW[:, : self.stratStep]
            loc_phiW = self.phiW[:, : self.stratStep]
        if self.carbOn:
            loc_stratC = self.stratC[:, : self.stratStep]
            loc_phiC = self.phiC[:, : self.stratStep]

        if self.carbOn:
            (
                nstratH,
                nstratZ,
                nstratF,
                nstratW,
                nstratC,
                nphiS,
                nphiF,
                nphiW,
                nphiC,
            ) = stratafullsed(
                self.lpoints,
                self.stratStep,
                indices,
                weights,
                loc_stratH,
                loc_stratZ,
                loc_stratF,
                loc_stratW,
                loc_stratC,
                loc_phiS,
                loc_phiF,
                loc_phiW,
                loc_phiC,
            )
        elif self.stratF is not None:
            nstratH, nstratZ, nstratF, nstratW, nphiS, nphiF, nphiW = stratathreesed(
                self.lpoints,
                self.stratStep,
                indices,
                weights,
                loc_stratH,
                loc_stratZ,
                loc_stratF,
                loc_stratW,
                loc_phiS,
                loc_phiF,
                loc_phiW,
            )
        else:
            nstratH, nstratZ, nphiS = strataonesed(
                self.lpoints,
                self.stratStep,
                indices,
                weights,
                loc_stratH,
                loc_stratZ,
                loc_phiS,
            )

        if len(onIDs) > 0:
            nstratZ[onIDs, :] = loc_stratZ[indices[onIDs, 0], :]
            nstratH[onIDs, :] = loc_stratH[indices[onIDs, 0], :]
            nphiS[onIDs, :] = loc_phiS[indices[onIDs, 0], :]
            if self.stratF is not None:
                nstratF[onIDs, :] = loc_stratF[indices[onIDs, 0], :]
                nphiF[onIDs, :] = loc_phiW[indices[onIDs, 0], :]
            if self.stratW is not None:
                nstratW[onIDs, :] = loc_stratW[indices[onIDs, 0], :]
                nphiW[onIDs, :] = loc_phiF[indices[onIDs, 0], :]
            if self.carbOn:
                nstratC[onIDs, :] = loc_stratC[indices[onIDs, 0], :]
                nphiC[onIDs, :] = loc_phiC[indices[onIDs, 0], :]

        # Updates stratigraphic records after mesh advection on the edges of each partition
        # to ensure that all stratigraphic information on the adjacent nodes of the neighbouring
        # partition are equals on all processors sharing a common number of nodes.
        for k in range(self.stratStep):
            self.tmp.setArray(nstratZ[:, k])
            self.dm.globalToLocal(self.tmp, self.tmpL)
            self.stratZ[:, k] = self.tmpL.getArray().copy()

            self.tmp.setArray(nstratH[:, k])
            self.dm.globalToLocal(self.tmp, self.tmpL)
            self.stratH[:, k] = self.tmpL.getArray().copy()
            self.tmp.setArray(nphiS[:, k])
            self.dm.globalToLocal(self.tmp, self.tmpL)
            self.phiS[:, k] = self.tmpL.getArray().copy()

            if self.stratF is not None:
                self.tmp.setArray(nstratF[:, k])
                self.dm.globalToLocal(self.tmp, self.tmpL)
                self.stratF[:, k] = self.tmpL.getArray().copy()
                self.tmp.setArray(nphiF[:, k])
                self.dm.globalToLocal(self.tmp, self.tmpL)
                self.phiF[:, k] = self.tmpL.getArray().copy()

            if self.stratW is not None:
                self.tmp.setArray(nstratW[:, k])
                self.dm.globalToLocal(self.tmp, self.tmpL)
                self.stratW[:, k] = self.tmpL.getArray().copy()
                self.tmp.setArray(nphiW[:, k])
                self.dm.globalToLocal(self.tmp, self.tmpL)
                self.phiW[:, k] = self.tmpL.getArray().copy()

            if self.carbOn:
                self.tmp.setArray(nstratC[:, k])
                self.dm.globalToLocal(self.tmp, self.tmpL)
                self.stratC[:, k] = self.tmpL.getArray().copy()
                self.tmp.setArray(nphiC[:, k])
                self.dm.globalToLocal(self.tmp, self.tmpL)
                self.phiC[:, k] = self.tmpL.getArray().copy()

        return
