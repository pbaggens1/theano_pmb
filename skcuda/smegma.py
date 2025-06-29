#!/usr/bin/env python

"""
Python interface to SMEGMA toolkit.
"""

from __future__ import absolute_import, division, print_function

import sys
import ctypes
import atexit
import numpy as np

from . import cuda
from . import misc

# Load SMEGMA library:
if 'linux' in sys.platform:
    #_libsmegma_libname_list = ['/home/paul.baggenstoss/software/smegma/lib/libsmegma.so']
    _libsmegma_libname_list = ['libsmegma.so']
elif sys.platform == 'darwin':
    _libsmegma_libname_list = ['smegma.so', 'libsmegma.dylib']
elif sys.platform == 'win32':
    _libsmegma_libname_list = ['smegma.dll']
else:
    raise RuntimeError('unsupported platform')

_load_err = ''
for _lib in _libsmegma_libname_list:
    try:
        _libsmegma = ctypes.cdll.LoadLibrary(_lib)
    except OSError:
        _load_err += ('' if _load_err == '' else ', ') + _lib
    else:
        _load_err = ''
        break
if _load_err:
    raise OSError('%s not found' % _load_err)

c_int_type = ctypes.c_longlong

# Exceptions corresponding to various SMEGMA errors:
# _libsmegma.smegma_strerror.restype = ctypes.c_char_p
#_libsmegma.smegma_strerror.argtypes = [c_int_type]
def smegma_strerror(error):
    """
    Return string corresponding to specified SMEGMA error code.
    """
    if error != 0:
         rval = 'smegma error ' + str(error)
    else:
         rval = 'zero error'
    return rval

class SmegmaError(Exception):
    def __init__(self, status, info=None):
        self._status = status
        self._info = info
        errstr = "%s (Code: %d)" % (smegma_strerror(status), status)
        super(SmegmaError,self).__init__(errstr)


def smegmaCheckStatus(status):
    """
    Raise an exception corresponding to the specified SMEGMA status code.
    """

    if status != 0:
        raise SmegmaError(status)

# Utility functions:
#_libsmegma.smegma_version.argtypes = [ctypes.c_void_p,
#    ctypes.c_void_p, ctypes.c_void_p]

def smegma_version():
    """
    Get SMEGMA version.
    """
    majv = c_int_type()
    minv = c_int_type()
    micv = c_int_type()
    #_libsmegma.smegma_version(ctypes.byref(majv),
    #    ctypes.byref(minv), ctypes.byref(micv))
    majv.value=1
    minv.value=0
    micv.value=0
    return (majv.value, minv.value, micv.value)
def smegma_init():
    """
    Initialize SMEGMA.
    """

def smegma_finalize():
    """
    Finalize SMEGMA.
    """

    ret_val = 0

def smegma_getdevice_arch():
    """
    Get device architecture.
    """
    arch = 'gpu'
    return arch

def smegma_getdevice():
    """
    Get current device used by SMEGMA.
    """
    return 'gpu'

def smegma_setdevice(dev):
    """
    Get current device used by SMEGMA.
    """
    do_nothing = 0

def smegma_device_sync():
    """
    Synchronize device used by SMEGMA.
    """
    do_nothing = 0


#---- PMB ----
_libsmegma.smegma_dinv_nl.restype = int
_libsmegma.smegma_dinv_nl.argtypes = [c_int_type, c_int_type,
                                  c_int_type, c_int_type,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p]

def smegma_dinv_nl(n, m, nl, nbatch, a, b, x, y):
    info = c_int_type()
    status = _libsmegma.smegma_dinv_nl(n, m, nl, nbatch, int(a), int(b), int(x), int(y), ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_sinv_nl.restype = int
_libsmegma.smegma_sinv_nl.argtypes = [c_int_type, c_int_type,
                                  c_int_type, c_int_type,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p]

def smegma_sinv_nl(n, m, nl, nbatch, a, b, x, y):
    info = c_int_type()
    status = _libsmegma.smegma_sinv_nl(n, m, nl, nbatch, int(a), int(b), int(x), int(y), ctypes.byref(info))
    magmaCheckStatus(status)




_libsmegma.smegma_spotrf_gpu_batch.restype = int
_libsmegma.smegma_spotrf_gpu_batch.argtypes = [c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p]
def smegma_spotrf_gpu_batch(uplo, n, A, lda, nbatch):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libsmegma.smegma_spotrf_gpu_batch(uplo, n, int(A), lda, nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dpotrf_gpu_batch.restype = int
_libsmegma.smegma_dpotrf_gpu_batch.argtypes = [c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p]
def smegma_dpotrf_gpu_batch(uplo, n, A, lda, nbatch):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libsmegma.smegma_dpotrf_gpu_batch(uplo, n, int(A), lda, nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_schol_solv.restype = int
_libsmegma.smegma_schol_solv.argtypes = [c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]

def smegma_schol_solv(uplo, n, m, A, b, c, nbatch):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libsmegma.smegma_schol_solv(uplo, n, m, int(A), int(b),  int(c), nbatch, ctypes.byref(info))
    magmaCheckStatus(status)


_libsmegma.smegma_dchol_solv.restype = int
_libsmegma.smegma_dchol_solv.argtypes = [c_int_type,
                                  c_int_type,
                                  c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]

def smegma_dchol_solv(uplo, n, m, A, b, c, nbatch):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libsmegma.smegma_dchol_solv(uplo, n, m, int(A), int(b),  int(c), nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_sbatched_dot.restype = int
_libsmegma.smegma_sbatched_dot.argtypes = [ c_int_type, c_int_type, c_int_type,
                              c_int_type, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  c_int_type, ctypes.c_void_p]

def smegma_sbatched_dot(nA, mA, nB, mB, A, B, C, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_sbatched_dot(nA, mA, nB, mB, int(A), int(B), int(C), nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dbatched_dot.restype = int
_libsmegma.smegma_dbatched_dot.argtypes = [ c_int_type, c_int_type, c_int_type,
                              c_int_type, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                  c_int_type, ctypes.c_void_p]

def smegma_dbatched_dot(nA, mA, nB, mB, A, B, C, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_dbatched_dot(nA, mA, nB, mB, int(A), int(B), int(C), nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dtri_solv.restype = int
_libsmegma.smegma_dtri_solv.argtypes = [c_int_type, c_int_type, c_int_type, c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]

def smegma_dtri_solv(uplo, trans, n, m, A, b, nbatch):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libsmegma.smegma_dtri_solv(uplo, trans, n, m, int(A), int(b),  nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_stri_solv.restype = int
_libsmegma.smegma_stri_solv.argtypes = [c_int_type, c_int_type, c_int_type, c_int_type,
                                  ctypes.c_void_p,
                                  ctypes.c_void_p,
                                  c_int_type,
                                  ctypes.c_void_p]

def smegma_stri_solv(uplo, trans, n, m, A, b, nbatch):
    """
    Cholesky factorization of positive symmetric matrix.
    """

    uplo = _uplo_conversion[uplo]
    info = c_int_type()
    status = _libsmegma.smegma_stri_solv(uplo, trans, n, m, int(A), int(b),  nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dlevinson.restype = int
_libsmegma.smegma_dlevinson.argtypes = [c_int_type, ctypes.c_void_p, ctypes.c_void_p,
                                       c_int_type, c_int_type, ctypes.c_void_p]

def smegma_dlevinson(p, A, j, direc, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_dlevinson(p, int(A), int(j), direc, nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_slevinson.restype = int
_libsmegma.smegma_slevinson.argtypes = [c_int_type, ctypes.c_void_p, ctypes.c_void_p,
                                       c_int_type, c_int_type, ctypes.c_void_p]

def smegma_slevinson(p, A, j, direc, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_slevinson(p, int(A), int(j), direc, nbatch, ctypes.byref(info))
    magmaCheckStatus(status)


_libsmegma.smegma_sar2rc.restype = int
_libsmegma.smegma_sar2rc.argtypes = [c_int_type, ctypes.c_void_p, ctypes.c_void_p, c_int_type, c_int_type, ctypes.c_void_p]

def smegma_sar2rc(p, a, j, direc, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_sar2rc(p, int(a), int(j), direc, nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dar2rc.restype = int
_libsmegma.smegma_dar2rc.argtypes = [c_int_type, ctypes.c_void_p, ctypes.c_void_p, c_int_type, c_int_type, ctypes.c_void_p]

def smegma_dar2rc(p, a, j, direc, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_dar2rc(p, int(a), int(j), direc, nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_stoep_solv.restype = int
_libsmegma.smegma_stoep_solv.argtypes = [c_int_type, c_int_type, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int_type, ctypes.c_void_p]

def smegma_stoep_solv(n, M, r, q, h, ldet, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_stoep_solv(n, M, int(r), int(q), int(h),  int(ldet), \
              nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dtoep_solv.restype = int
_libsmegma.smegma_dtoep_solv.argtypes = [c_int_type, c_int_type, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int_type, ctypes.c_void_p]

def smegma_dtoep_solv(n, M, r, q, h, ldet, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_dtoep_solv(n, M, int(r), int(q), int(h),  int(ldet), \
              nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_diir.restype = int
_libsmegma.smegma_diir.argtypes = [c_int_type, c_int_type, ctypes.c_void_p, ctypes.c_void_p, c_int_type, ctypes.c_void_p]

def smegma_diir(p, N, a, x, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_diir(p, N, int(a), int(x), nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_siir.restype = int
_libsmegma.smegma_siir.argtypes = [c_int_type, c_int_type, ctypes.c_void_p, ctypes.c_void_p, c_int_type, ctypes.c_void_p]

def smegma_siir(p, N, a, x, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_siir(p, N, int(a), int(x), nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_dtggen.restype = int
_libsmegma.smegma_dtggen.argtypes = [c_int_type, ctypes.c_void_p, ctypes.c_void_p, c_int_type, ctypes.c_void_p, ctypes.c_void_p]

def smegma_dtggen(dim, mu, x, nbatch, seed):
    info = c_int_type()
    status = _libsmegma.smegma_dtggen(dim, int(mu), int(x), nbatch, int(seed), ctypes.byref(info))
    magmaCheckStatus(status)

_libsmegma.smegma_stggen.restype = int
_libsmegma.smegma_stggen.argtypes = [c_int_type, ctypes.c_void_p, ctypes.c_void_p, c_int_type, ctypes.c_void_p, ctypes.c_void_p]

def smegma_stggen(dim, mu, x, nbatch, seed):
    info = c_int_type()
    status = _libsmegma.smegma_stggen(dim, int(mu), int(x), nbatch, int(seed), ctypes.byref(info))
    magmaCheckStatus(status)


_libsmegma.smegma_ssp_exp.restype = int
_libsmegma.smegma_ssp_exp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int_type, c_int_type, c_int_type, ctypes.c_void_p]

def smegma_ssp_exp(dW, dz, dlam, dldetr, dk, N, M, nbatch):
    info = c_int_type()
    status = _libsmegma.smegma_ssp_exp(int(dW), int(dz), int(dlam), int(dldetr), int(dk), N,M,nbatch, ctypes.byref(info))
    magmaCheckStatus(status)

