dnl Process this m4 file to produce 'C' language file.
dnl
dnl If you see this line, you can ignore the next one.
/* Do not edit this file. It is produced from the corresponding .m4 source */
dnl
/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the corresponding APIs defined in
 * src/dispatchers/var_getput.m4
 *
 * ncmpi_get_varn_<type>() : dispatcher->get_varn()
 * ncmpi_put_varn_<type>() : dispatcher->put_varn()
 */


#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <mpi.h>

#include <pnc_debug.h>
#include <common.h>
#include "ncmpio_NC.h"

include(`utils.m4')

dnl
define(`VARN',dnl
`dnl
/*----< ncmpio_$1_varn() >--------------------------------------------------*/
int
ncmpio_$1_varn(void              *ncdp,
               int                varid,
               int                num,
               MPI_Offset* const *starts,
               MPI_Offset* const *counts,
               ifelse(`$1',`put',`const') void *buf,
               MPI_Offset         bufcount,
               MPI_Datatype       buftype,
               int                reqMode)
{
    int reqid=NC_REQ_NULL, err=NC_NOERR, status;

    if (!fIsSet(reqMode, NC_REQ_ZERO)) {
        err = ncmpio_i`$1'_varn(ncdp, varid, num, starts, counts, buf,
                                bufcount, buftype, &reqid, reqMode);
        if (err != NC_NOERR && fIsSet(reqMode, NC_REQ_INDEP))
            return err;
    }

    status = ncmpio_wait(ncdp, 1, &reqid, NULL, reqMode);

    return (err != NC_NOERR) ? err : status;
}
')dnl

VARN(put)
VARN(get)
