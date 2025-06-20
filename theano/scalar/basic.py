"""
.. warning::

This directory is for the internal of Theano.

You are strongly advised not to use it, except if you know
what you are doing!

If you want to use a scalar variable in a Theano graph,
you probably want to use theano.tensor.[c,z,f,d,b,w,i,l,]scalar!
"""

import math
import warnings
from collections.abc import Callable
from copy import copy
from functools import partial
from itertools import chain
from textwrap import dedent

import numpy as np

import theano
from theano import config, gof, printing
from theano.gof import Apply, Constant, FunctionGraph, Op, Type, Variable, utils
from theano.gradient import DisconnectedType, grad_undefined
from theano.misc.safe_asarray import _asarray
from theano.printing import pprint


builtin_bool = bool
builtin_complex = complex
builtin_int = int
builtin_float = float


class ComplexError(NotImplementedError):
    """
    Raised if complex numbers are used in an unsupported operation.

    """


class IntegerDivisionError(Exception):
    """
    Raised if someone tries to divide integers with '/' instead of '//'.

    """


def upcast(dtype, *dtypes):
    # This tries to keep data in floatX or lower precision, unless we
    # explicitely request a higher precision datatype.
    keep_float32 = [
        (config.cast_policy == "numpy+floatX" and config.floatX == "float32")
    ]
    keep_float16 = [
        (config.cast_policy == "numpy+floatX" and config.floatX == "float16")
    ]

    def make_array(dt):
        if dt == "float64":
            # There is an explicit float64 dtype: we cannot keep float32.
            keep_float32[0] = False
            keep_float16[0] = False
        if dt == "float32":
            keep_float16[0] = False
        return np.zeros((), dtype=dt)

    z = make_array(dtype)
    for dt in dtypes:
        z = z + make_array(dt=dt)
    rval = str(z.dtype)
    if rval == "float64":
        if keep_float16[0]:
            return "float16"
        if keep_float32[0]:
            return "float32"
    elif rval == "float32":
        if keep_float16[0]:
            return "float16"
    return rval


def as_common_dtype(*vars):
    """
    For for theano.scalar.Scalar and TensorVariable.
    """
    dtype = upcast(*[v.dtype for v in vars])
    return (v.astype(dtype) for v in vars)


def get_scalar_type(dtype):
    """
    Return a Scalar(dtype) object.

    This caches objects to save allocation and run time.

    """
    if dtype not in get_scalar_type.cache:
        get_scalar_type.cache[dtype] = Scalar(dtype=dtype)
    return get_scalar_type.cache[dtype]


get_scalar_type.cache = {}


def as_scalar(x, name=None):
    from ..tensor import TensorType, scalar_from_tensor

    if isinstance(x, gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a multi-output"
                " Op has to be fetched.",
                x,
            )
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if isinstance(x.type, Scalar):
            return x
        elif isinstance(x.type, TensorType) and x.ndim == 0:
            return scalar_from_tensor(x)
        else:
            raise TypeError("Variable type field must be a Scalar.", x, x.type)
    try:
        return constant(x)
    except TypeError:
        raise TypeError(f"Cannot convert {x} to Scalar", type(x))


class NumpyAutocaster:
    """
    This class is used to cast python ints and floats to numpy arrays.

    The behavior when called on scalar `x` depends on `config.cast_policy`:
        - 'numpy' will simply use the same type as found by `numpy.asarray(x)`.
        - 'numpy+floatX' will do the same, except it will use float32 instead
          of float64 if `x` is a Python float and `config.floatX` is set to
          'float32' (note that if `x` is a numpy scalar whose data type is
          float64, it is not modified since we assume the user is purposely
          using float64).
        - 'custom' lets one define a tuple of data types such that:
            - if `x` is already a numpy scalar and its data type is in this
              tuple, then it is returned unchanged;
            - otherwise, the first data type in this tuple that can represent
              `x` without loss of precision will be used, unless `x` is a float
              and 'float32' is in the tuple (in which case `x` is cast as a
              float32);
            - if no data type can represent `x` without loss of precision, then
              the last data type in the tuple will be used.


    Parameters
    ----------
    dtypes: tuple of strings
        The ordered list of preferred data types (only used when
        `config.cast_policy` is set to 'custom', see the `NumpyAutocaster`
        help for details).

    """

    def __init__(self, dtypes):
        self.dtypes = tuple(dtypes)

    def __call__(self, x):
        # Make sure we only deal with scalars.
        assert isinstance(x, (int, builtin_float)) or (
            isinstance(x, np.ndarray) and x.ndim == 0
        )

        if config.cast_policy == "numpy":
            return np.asarray(x)
        elif config.cast_policy == "numpy+floatX":
            rval = np.asarray(x)
            if (
                not hasattr(x, "dtype")
                and rval.dtype in ("float64", "float32")
                and rval.dtype != config.floatX
            ):
                rval = _asarray(rval, dtype=config.floatX)
            return rval

        # The following is the original code, corresponding to the 'custom'
        # option for `config.cast_policy`.
        assert config.cast_policy == "custom"

        try:
            # Pass through numpy scalars, since they are already typed on
            # purpose typically.
            if str(x.dtype) in self.dtypes:
                # No need to cast `x` into a new dtype. Note that we still
                # need to convert it into an array, because it may not be
                # one already (e.g. if x == numpy.float64(1.1)).
                return np.asarray(x)
        except AttributeError:
            # Means `x` has no 'dtype' attribute.
            pass

        # unsafe downcast of float64 variables when config.floatX == 'float32'
        # recall: float is numpy.float
        if (
            isinstance(x, float)
            and config.floatX in self.dtypes
            and config.floatX != "float64"
        ):
            return _asarray(x, dtype=config.floatX)

        # Don't autocast to float16 unless config.floatX is float16
        try_dtypes = [
            d for d in self.dtypes if config.floatX == "float16" or d != "float16"
        ]

        for dtype in try_dtypes:
            x_ = _asarray(x, dtype=dtype)
            if np.all(x == x_):
                break
        # returns either an exact x_==x, or the last cast x_
        return x_


autocast_int = NumpyAutocaster(("int8", "int16", "int32", "int64"))
# autocast_float dtypes might be manipulated in tensor.*
autocast_float = NumpyAutocaster(("float16", "float32", "float64"))


class autocast_float_as:
    """
    Temporarily adjust autocasting behavior.

    This class makes it possible to temporarily and locally adjust autocasting
    behavior when `config.cast_policy` is set to 'custom'.
    If `config.cast_policy` is not 'custom', an exception is raised.
    This class might be convenient in some code, but it definitely
    helps to test the autocasting mechanism.

    Examples
    --------
    >>> with autocast_float_as('float32'):
    ...    assert (fvector() + 1.1).dtype == 'float32'  # temporary downcasting
    >>> assert (fvector() + 1.1).dtype == 'float64' # back to default behaviour

    """

    def __init__(self, *dtypes):
        self.dtypes = dtypes
        assert config.cast_policy == "custom"

    def __enter__(self):
        assert config.cast_policy == "custom"
        self.old_dtypes = autocast_float.dtypes
        autocast_float.dtypes = self.dtypes

    def __exit__(self, *args):
        assert config.cast_policy == "custom"
        autocast_float.dtypes = self.old_dtypes


def convert(x, dtype=None):
    """Convert the input to a properly typed NumPy value according to the current casting policy.

    Parameters
    ----------
    x : Number, numpy.ndarray, or Sequence[Number]
        The value(s) to be converted
    dtype : str or numpy.dtype (optional)
        The dtype to use for the conversion of `x`.

    """
    if dtype is not None:
        # in this case, the semantics are that the caller is forcing the dtype
        x_ = _asarray(x, dtype=dtype)
    else:
        # In this case, this function should infer the dtype according to the
        # autocasting rules. See autocasting above.
        x_ = None
        if isinstance(x, int):
            try:
                x_ = autocast_int(x)
            except OverflowError:
                # This is to imitate numpy behavior which tries to fit
                # bigger numbers into a uint64.
                x_ = _asarray(x, dtype="uint64")
        elif isinstance(x, builtin_float):
            x_ = autocast_float(x)
        elif isinstance(x, np.ndarray):
            x_ = x
        else:
            # Here x is probably a list or a tuple. If it contains a
            # long, we will behave like the current NumPy version: it
            # will work if the long fits in int64 or uint64.
            x_ = np.asarray(x)
            if x_.size == 0 and not hasattr(x, "dtype"):
                x_ = np.asarray(x, dtype=config.floatX)
    assert issubclass(type(x_), (np.ndarray, np.memmap))
    return x_


def constant(x, name=None, dtype=None):
    x = convert(x, dtype=dtype)
    assert x.ndim == 0
    return ScalarConstant(get_scalar_type(str(x.dtype)), x, name=name)


class Scalar(Type):

    """
    Internal class, should not be used by clients.

    Primarily used by tensor.elemwise and tensor.reduce.
    Analogous to TensorType, but for zero-dimensional objects.
    Maps directly to C primitives.

    TODO: refactor to be named ScalarType for consistency with TensorType.

    """

    __props__ = ("dtype",)
    ndim = 0

    def __init__(self, dtype):
        if dtype == "floatX":
            dtype = config.floatX
        self.dtype = dtype
        self.dtype_specs()  # error checking

    @staticmethod
    def may_share_memory(a, b):
        # This class represent basic c type, represented in python
        # with numpy.scalar. They are read only. So from python, they
        # can never share memory.
        return False

    def filter(self, data, strict=False, allow_downcast=None):
        py_type = self.dtype_specs()[0]
        if strict and not isinstance(data, py_type):
            raise TypeError(
                f"{self} expected a {py_type}, got {data} of type {type(data)}",
                data,
            )
        try:
            converted_data = py_type(data)
            if (
                allow_downcast
                or (
                    allow_downcast is None
                    and isinstance(data, (float, np.floating))
                    and self.dtype == theano.config.floatX
                )
                or data == converted_data
            ):
                return py_type(data)
            else:
                raise TypeError(
                    "Value cannot accurately be converted to dtype"
                    f" ({self.dtype}) and allow_downcast is not True"
                )
        except Exception:
            raise TypeError(
                f"Could not convert {type(data)} (value={data}) to {self.dtype}",
            )

    def values_eq_approx(self, a, b, tolerance=1e-4):
        # The addition have risk of overflow especially with [u]int8
        if self.dtype == "bool":
            return a == b
        diff = a - b
        if diff == 0:
            return True
        return abs(diff) <= (abs(a) * tolerance) + (abs(b) * tolerance)

    def c_element_type(self):
        return self.dtype_specs()[1]

    def c_headers(self, c_compiler):
        l = ["<math.h>"]
        # These includes are needed by Scalar and TensorType,
        # we declare them here and they will be re-used by TensorType
        l.append("<numpy/arrayobject.h>")
        l.append("<numpy/arrayscalars.h>")
        if config.lib.amdlibm and c_compiler.supports_amdlibm:
            l += ["<amdlibm.h>"]
        return l

    def c_libraries(self, c_compiler):
        l = []
        if config.lib.amdlibm and c_compiler.supports_amdlibm:
            l += ["amdlibm"]
        return l

    def c_compile_args(self, c_compiler):
        if config.lib.amdlibm and c_compiler.supports_amdlibm:
            return ["-DREPLACE_WITH_AMDLIBM"]
        else:
            return []

    def dtype_specs(self):
        try:
            # To help debug dtype/typenum problem, here is code to get
            # the list of numpy typenum.  This list change between 32
            # and 64 bit platform and probably also also between
            # Windows and Linux.
            # NOTE: equivalent type on a platform can have different typenum.
            #     This is the source of all dtype/typenum problem found up to
            #     now, as Theano always expect the exact typenum that
            #     correspond to our supported dtype.
            """
            for dtype in ['bool', 'int8', 'uint8', 'short', 'ushort', 'intc',
                          'uintc',
                          'longlong', 'ulonglong', 'single', 'double',
                          'longdouble', 'csingle', 'cdouble', 'clongdouble',
                          'float32', 'float64', 'int8', 'int16', 'int32',
                          'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                          'complex64', 'complex128', 'float', 'double',
                          'int', 'uint']:
                print(dtype, np.zeros(1, dtype=dtype).dtype.num)
            """
            return {  # dtype: (py_type, c_type, cls_name)
                "float16": (np.float16, "npy_float16", "Float16"),
                "float32": (np.float32, "npy_float32", "Float32"),
                "float64": (np.float64, "npy_float64", "Float64"),
                "complex128": (np.complex128, "theano_complex128", "Complex128"),
                "complex64": (np.complex64, "theano_complex64", "Complex64"),
                "bool": (np.bool_, "npy_bool", "Bool"),
                "uint8": (np.uint8, "npy_uint8", "UInt8"),
                "int8": (np.int8, "npy_int8", "Int8"),
                "uint16": (np.uint16, "npy_uint16", "UInt16"),
                "int16": (np.int16, "npy_int16", "Int16"),
                "uint32": (np.uint32, "npy_uint32", "UInt32"),
                "int32": (np.int32, "npy_int32", "Int32"),
                "uint64": (np.uint64, "npy_uint64", "UInt64"),
                "int64": (np.int64, "npy_int64", "Int64"),
            }[self.dtype]
        except KeyError:
            raise TypeError(
                f"Unsupported dtype for {self.__class__.__name__}: {self.dtype}"
            )

    def upcast(self, *others):
        return upcast(*[x.dtype for x in [self] + list(others)])

    def make_variable(self, name=None):
        return ScalarVariable(self, name=name)

    def __str__(self):
        return str(self.dtype)

    def __repr__(self):
        return f"Scalar({self.dtype})"

    def c_literal(self, data):
        if "complex" in self.dtype:
            raise NotImplementedError("No literal for complex values.")
        if self.dtype == "bool":
            return "1" if data else "0"
        return str(data)

    def c_declare(self, name, sub, check_input=True):
        dtype = self.dtype_specs()[1]

        if check_input:
            pre = f"""
                typedef {dtype} dtype_{name};
            """
        else:
            pre = ""

        return f"""{pre}
        {dtype} {name};
        """

    def c_init(self, name, sub):
        return f"""
        {name} = 0;
        """

    def c_extract(self, name, sub, check_input=True):
        if self.dtype == "float16":
            # This doesn't work at the numpy level
            raise NotImplementedError("float16")
        specs = self.dtype_specs()
        if check_input:
            pre = """
            if (!PyObject_TypeCheck(py_%(name)s, &%(pyarr_type)s))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (%(dtype)s)");
                %(fail)s
            }
            """ % dict(
                sub, name=name, dtype=specs[1], pyarr_type="Py%sArrType_Type" % specs[2]
            )
        else:
            pre = ""
        return (
            pre
            + """
        PyArray_ScalarAsCtype(py_%(name)s, &%(name)s);
        """
            % dict(sub, name=name)
        )

    def c_sync(self, name, sub):
        specs = self.dtype_specs()
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = PyArrayScalar_New(%(cls)s);
        if (!py_%(name)s)
        {
            Py_XINCREF(Py_None);
            py_%(name)s = Py_None;
            PyErr_Format(PyExc_MemoryError,
                "Instantiation of new Python scalar failed (%(dtype)s)");
            %(fail)s
        }
        PyArrayScalar_ASSIGN(py_%(name)s, %(cls)s, %(name)s);
        """ % dict(
            sub, name=name, dtype=specs[1], cls=specs[2]
        )

    def c_cleanup(self, name, sub):
        return ""

    def c_support_code(self):

        if self.dtype.startswith("complex"):
            cplx_types = ["theano_complex64", "theano_complex128"]
            real_types = [
                "npy_int8",
                "npy_int16",
                "npy_int32",
                "npy_int64",
                "npy_float32",
                "npy_float64",
            ]
            # If the 'int' C type is not exactly the same as an existing
            # 'npy_intX', some C code may not compile, e.g. when assigning
            # the value 0 (cast to 'int' in C) to a theano_complex64.
            if np.dtype("intc").num not in [np.dtype(d[4:]).num for d in real_types]:
                # In that case we add the 'int' type to the real types.
                real_types.append("int")

            template = """
            struct theano_complex%(nbits)s : public npy_complex%(nbits)s
            {
                typedef theano_complex%(nbits)s complex_type;
                typedef npy_float%(half_nbits)s scalar_type;

                complex_type operator +(const complex_type &y) const {
                    complex_type ret;
                    ret.real = this->real + y.real;
                    ret.imag = this->imag + y.imag;
                    return ret;
                }

                complex_type operator -() const {
                    complex_type ret;
                    ret.real = -this->real;
                    ret.imag = -this->imag;
                    return ret;
                }
                bool operator ==(const complex_type &y) const {
                    return (this->real == y.real) && (this->imag == y.imag);
                }
                bool operator ==(const scalar_type &y) const {
                    return (this->real == y) && (this->imag == 0);
                }
                complex_type operator -(const complex_type &y) const {
                    complex_type ret;
                    ret.real = this->real - y.real;
                    ret.imag = this->imag - y.imag;
                    return ret;
                }
                complex_type operator *(const complex_type &y) const {
                    complex_type ret;
                    ret.real = this->real * y.real - this->imag * y.imag;
                    ret.imag = this->real * y.imag + this->imag * y.real;
                    return ret;
                }
                complex_type operator /(const complex_type &y) const {
                    complex_type ret;
                    scalar_type y_norm_square = y.real * y.real + y.imag * y.imag;
                    ret.real = (this->real * y.real + this->imag * y.imag) / y_norm_square;
                    ret.imag = (this->imag * y.real - this->real * y.imag) / y_norm_square;
                    return ret;
                }
                template <typename T>
                complex_type& operator =(const T& y);

                theano_complex%(nbits)s() {}

                template <typename T>
                theano_complex%(nbits)s(const T& y) { *this = y; }

                template <typename TR, typename TI>
                theano_complex%(nbits)s(const TR& r, const TI& i) { this->real=r; this->imag=i; }
            };
            """

            def operator_eq_real(mytype, othertype):
                return f"""
                template <> {mytype} & {mytype}::operator=<{othertype}>(const {othertype} & y)
                {{ this->real=y; this->imag=0; return *this; }}
                """

            def operator_eq_cplx(mytype, othertype):
                return f"""
                template <> {mytype} & {mytype}::operator=<{othertype}>(const {othertype} & y)
                {{ this->real=y.real; this->imag=y.imag; return *this; }}
                """

            operator_eq = "".join(
                operator_eq_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            ) + "".join(
                operator_eq_cplx(ctype1, ctype2)
                for ctype1 in cplx_types
                for ctype2 in cplx_types
            )

            # We are not using C++ generic templating here, because this would
            # generate two different functions for adding a complex64 and a
            # complex128, one returning a complex64, the other a complex128,
            # and the compiler complains it is ambiguous.
            # Instead, we generate code for known and safe types only.

            def operator_plus_real(mytype, othertype):
                return f"""
                const {mytype} operator+(const {mytype} &x, const {othertype} &y)
                {{ return {mytype}(x.real+y, x.imag); }}

                const {mytype} operator+(const {othertype} &y, const {mytype} &x)
                {{ return {mytype}(x.real+y, x.imag); }}
                """

            operator_plus = "".join(
                operator_plus_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            )

            def operator_minus_real(mytype, othertype):
                return f"""
                const {mytype} operator-(const {mytype} &x, const {othertype} &y)
                {{ return {mytype}(x.real-y, x.imag); }}

                const {mytype} operator-(const {othertype} &y, const {mytype} &x)
                {{ return {mytype}(y-x.real, -x.imag); }}
                """

            operator_minus = "".join(
                operator_minus_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            )

            def operator_mul_real(mytype, othertype):
                return f"""
                const {mytype} operator*(const {mytype} &x, const {othertype} &y)
                {{ return {mytype}(x.real*y, x.imag*y); }}

                const {mytype} operator*(const {othertype} &y, const {mytype} &x)
                {{ return {mytype}(x.real*y, x.imag*y); }}
                """

            operator_mul = "".join(
                operator_mul_real(ctype, rtype)
                for ctype in cplx_types
                for rtype in real_types
            )

            return (
                template % dict(nbits=64, half_nbits=32)
                + template % dict(nbits=128, half_nbits=64)
                + operator_eq
                + operator_plus
                + operator_minus
                + operator_mul
            )

        else:
            return ""

    def c_init_code(self):
        return ["import_array();"]

    def c_code_cache_version(self):
        return (13, np.__version__)

    def get_shape_info(self, obj):
        return obj.itemsize

    def get_size(self, shape_info):
        return shape_info


# Register C code for ViewOp on Scalars.
theano.compile.register_view_op_c_code(
    Scalar,
    """
    %(oname)s = %(iname)s;
    """,
    1,
)


bool = get_scalar_type("bool")
int8 = get_scalar_type("int8")
int16 = get_scalar_type("int16")
int32 = get_scalar_type("int32")
int64 = get_scalar_type("int64")
uint8 = get_scalar_type("uint8")
uint16 = get_scalar_type("uint16")
uint32 = get_scalar_type("uint32")
uint64 = get_scalar_type("uint64")
float16 = get_scalar_type("float16")
float32 = get_scalar_type("float32")
float64 = get_scalar_type("float64")
complex64 = get_scalar_type("complex64")
complex128 = get_scalar_type("complex128")

int_types = int8, int16, int32, int64
uint_types = uint8, uint16, uint32, uint64
float_types = float16, float32, float64
complex_types = complex64, complex128

integer_types = int_types + uint_types
discrete_types = (bool,) + integer_types
continuous_types = float_types + complex_types
all_types = discrete_types + continuous_types


class _scalar_py_operators:
    # So that we can simplify checking code when we have a mixture of Scalar
    # variables and Tensor variables
    ndim = 0

    dtype = property(lambda self: self.type.dtype)
    """The dtype of this scalar."""

    # UNARY
    def __abs__(self):
        return abs_(self)

    def __neg__(self):
        return neg(self)

    # CASTS
    # def __int__(self): return AsInt(self).out
    # def __float__(self): return AsDouble(self).out
    # def __complex__(self): return AsComplex(self).out

    # BITWISE
    def __invert__(self):
        return invert(self)

    def __and__(self, other):
        return and_(self, other)

    def __or__(self, other):
        return or_(self, other)

    def __xor__(self, other):
        return xor(self, other)

    def __rand__(self, other):
        return and_(other, self)

    def __ror__(self, other):
        return or_(other, self)

    def __rxor__(self, other):
        return xor(other, self)

    # COMPARISONS
    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    # ARITHMETIC - NORMAL
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div_proxy(self, other)

    def __div__(self, other):
        return div_proxy(self, other)

    def __floordiv__(self, other):
        return int_div(self, other)

    def __mod__(self, other):
        return mod_check(self, other)

    def __pow__(self, other):
        return pow(self, other)

    # ARITHMETIC - RIGHT-OPERAND
    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        return mul(other, self)

    def __rdiv__(self, other):
        return div_proxy(other, self)

    def __rmod__(self, other):
        return mod(other, self)

    def __rpow__(self, other):
        return pow(other, self)

    def zeros_like(self, dtype=None):
        # The second is needed for Elemwise ops to work right
        if dtype is None:
            dtype = str(self.type.dtype)
        return second(self, ScalarConstant(get_scalar_type(dtype), 0))

    def ones_like(self, dtype=None):
        # The second is needed for Elemwise ops to work right
        if dtype is None:
            dtype = str(self.type.dtype)
        return second(self, ScalarConstant(get_scalar_type(dtype), 1))

    def astype(self, dtype):
        return cast(self, dtype)


class ScalarVariable(_scalar_py_operators, Variable):
    pass


class ScalarConstant(_scalar_py_operators, Constant):
    pass


# Register ScalarConstant as the type of Constant corresponding to Scalar
Scalar.Constant = ScalarConstant


# Easy constructors


def _multi(*fns):
    def f2(f, names):
        if len(names) == 1:
            return f(names)
        else:
            return [f(name) for name in names]

    if len(fns) == 1:
        return partial(f2, fns[0])
    else:
        return [partial(f2, f) for f in fns]


ints = _multi(int64)
floats = _multi(float64)
complexs = _multi(complex128)
complexs64 = _multi(complex64)
complexs128 = _multi(complex128)


def upcast_out(*types):
    dtype = Scalar.upcast(*types)
    return (get_scalar_type(dtype),)


def upcast_out_nobool(*types):
    type = upcast_out(*types)
    if type[0] == bool:
        raise TypeError("bool output not supported")
    return type


def upcast_out_min8(*types):
    type = upcast_out(*types)
    if type[0] == bool:
        return (int8,)
    return type


def upgrade_to_float(*types):
    """
    Upgrade any int types to float32 or float64 to avoid losing precision.

    """
    conv = {
        bool: float32,
        int8: float32,
        int16: float32,
        int32: float64,
        int64: float64,
        uint8: float32,
        uint16: float32,
        uint32: float64,
        uint64: float64,
    }
    return (get_scalar_type(Scalar.upcast(*[conv.get(type, type) for type in types])),)


def upgrade_to_float64(*types):
    """
    Upgrade any int and float32 to float64 to do as SciPy.

    """
    return (get_scalar_type("float64"),)


def same_out(type):
    return (type,)


def same_out_nobool(type):
    if type == bool:
        raise TypeError("bool input not supported")
    return (type,)


def same_out_min8(type):
    if type == bool:
        return (int8,)
    return (type,)


def upcast_out_no_complex(*types):
    if any([type in complex_types for type in types]):
        raise TypeError("complex type are not supported")
    return (get_scalar_type(dtype=Scalar.upcast(*types)),)


def same_out_float_only(type):
    if type not in float_types:
        raise TypeError("only float type are supported")
    return (type,)


class transfer_type(gof.utils.object2):
    __props__ = ("transfer",)

    def __init__(self, *transfer):
        assert all(type(x) in [int, str] or x is None for x in transfer)
        self.transfer = transfer

    def __str__(self):
        return f"transfer_type{self.transfer}"

    def __call__(self, *types):
        upcast = upcast_out(*types)
        retval = []
        for i in self.transfer:
            if i is None:
                retval += [upcast]
            elif isinstance(i, str):
                retval += [i]
            else:
                retval += [types[i]]
        return retval
        # return [upcast if i is None else types[i] for i in self.transfer]


class specific_out(gof.utils.object2):
    __props__ = ("spec",)

    def __init__(self, *spec):
        self.spec = spec

    def __call__(self, *types):
        return self.spec


def int_out(*types):
    return (int64,)


def float_out(*types):
    return (float64,)


def upgrade_to_float_no_complex(*types):
    """
    Don't accept complex, otherwise call upgrade_to_float().

    """
    for type in types:
        if type in complex_types:
            raise TypeError("complex argument not supported")
    return upgrade_to_float(*types)


def same_out_nocomplex(type):
    if type in complex_types:
        raise TypeError("complex argument not supported")
    return (type,)


def int_out_nocomplex(*types):
    for type in types:
        if type in complex_types:
            raise TypeError("complex argument not supported")
    return (int64,)


def float_out_nocomplex(*types):
    for type in types:
        if type in complex_types:
            raise TypeError("complex argument not supported")
    return (float64,)


class unary_out_lookup(gof.utils.object2):
    """
    Get a output_types_preference object by passing a dictionary:

    unary_out_lookup({int8:int32, float32:complex128})

    The result is an op that maps in8 to int32 and float32 to
    complex128 and other input types lead to a TypeError.

    """

    def __init__(self, type_table):
        self.tbl = type_table

    def __call__(self, *types):
        if len(types) == 1:
            types = types[0]
        try:
            rval = self.tbl[types]
        except Exception:
            raise TypeError(types)
        if isinstance(types, (list, tuple)):
            return rval
        else:
            return [rval]

    def __eq__(self, other):
        return type(self) == type(other) and self.tbl == other.tbl

    def __hash__(self):
        return hash(type(self))  # ignore hash of table


def real_out(type):
    if type == complex64:
        return (float32,)
    if type == complex128:
        return (float64,)
    return (type,)


class ScalarOp(Op):

    nin = -1
    nout = 1

    def __init__(self, output_types_preference=None, name=None):
        self.name = name
        if output_types_preference is not None:
            if not isinstance(output_types_preference, Callable):
                raise TypeError(
                    f"Expected a callable for the 'output_types_preference' argument to {self.__class__}. "
                    f"(got: {output_types_preference})"
                )
            self.output_types_preference = output_types_preference

    def make_node(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError(
                    f"Wrong number of inputs for {self}.make_node "
                    f"(got {len(inputs)}({inputs}), expected {self.nin})"
                )
        inputs = [as_scalar(input) for input in inputs]
        outputs = [t() for t in self.output_types([input.type for input in inputs])]
        if len(outputs) != self.nout:
            inputs_str = (", ".join(str(input) for input in inputs),)
            raise TypeError(
                f"Not the right number of outputs produced for {self}({inputs_str}). "
                f"Expected {self.nout}, got {len(outputs)}."
            )
        return Apply(self, inputs, outputs)

    def output_types(self, types):
        if hasattr(self, "output_types_preference"):
            variables = self.output_types_preference(*types)
            if not isinstance(variables, (list, tuple)) or any(
                not isinstance(x, Type) for x in variables
            ):
                raise TypeError(
                    "output_types_preference should return a list or a tuple of types",
                    self.output_types_preference,
                    variables,
                )
            if len(variables) != self.nout:
                variables_str = ", ".join(str(type) for type in variables)
                raise TypeError(
                    "Not the right number of outputs types produced for "
                    f"{self}({variables_str}) by {self.output_types_preference}. "
                    f"Expected {self.nout}, got {len(variables)}."
                )
            return variables
        else:
            raise NotImplementedError(f"Cannot calculate the output types for {self}")

    def perform(self, node, inputs, output_storage):
        if self.nout == 1:
            output_storage[0][0] = self.impl(*inputs)
        else:
            variables = utils.from_return_values(self.impl(*inputs))
            assert len(variables) == len(output_storage)
            for storage, variable in zip(output_storage, variables):
                storage[0] = variable

    def impl(self, *inputs):
        raise utils.MethodNotDefined("impl", type(self), self.__class__.__name__)

    def grad(self, inputs, output_gradients):
        raise utils.MethodNotDefined("grad", type(self), self.__class__.__name__)

    def L_op(self, inputs, outputs, output_gradients):
        return self.grad(inputs, output_gradients)

    def __eq__(self, other):
        test = type(self) == type(other) and getattr(
            self, "output_types_preference", None
        ) == getattr(other, "output_types_preference", None)
        return test

    def __hash__(self):
        return hash((type(self), getattr(self, "output_types_preference", 0)))

    def __str__(self):
        if hasattr(self, "name") and self.name:
            return self.name
        else:
            param = [
                (k, v)
                for k, v in self.__dict__.items()
                if k
                not in ["name", "_op_use_c_code", "bool", "output_types_preference"]
            ]
            if param:
                return "{}{{{}}}".format(
                    self.__class__.__name__,
                    ", ".join("{}={}".format(k, v) for k, v in param),
                )
            else:
                return self.__class__.__name__

    def c_code_cache_version(self):
        return (4,)

    def c_code_contiguous(self, node, name, inp, out, sub):
        """
        This function is called by Elemwise when all inputs and outputs are
        c_contiguous. This allows to use the SIMD version of this op.

        The inputs are the same as c_code except that:

            - inp and out must be the names of the variables associated to the
              ndarrays in the C code
            - node must be the elemwise node (this is needed to know
              the inputs/outputs types)

        """
        raise theano.gof.utils.MethodNotDefined()

    def supports_c_code(self, inputs, outputs):
        """Returns True if the current op has functioning C code for
        the given Elemwise inputs, outputs.

        """
        try:
            tmp_s_input = []
            # To keep the same aliasing between inputs
            mapping = dict()
            for ii in inputs:
                if ii in mapping:
                    tmp_s_input.append(mapping[ii])
                else:
                    tmp = get_scalar_type(ii.dtype).make_variable()
                    tmp_s_input.append(tmp)
                    mapping[ii] = tmp_s_input[-1]

            with theano.change_flags(compute_test_value="ignore"):
                s_op = self(*tmp_s_input, return_list=True)

            # if the scalar_op don't have a c implementation,
            # we skip its fusion to allow the fusion of the
            # other ops.
            self.c_code(
                s_op[0].owner,
                "test_presence_of_c_code",
                ["x" for x in inputs],
                ["z" for z in outputs],
                {"fail": "%(fail)s"},
            )
        except (theano.gof.utils.MethodNotDefined, NotImplementedError):
            return False
        return True


class UnaryScalarOp(ScalarOp):
    nin = 1
    amd_float32 = None
    amd_float64 = None

    def c_code_contiguous(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if (
            not theano.config.lib.amdlibm
            or
            # We compare the dtype AND the broadcast flag
            # as this function do not broadcast
            node.inputs[0].type != node.outputs[0].type
        ):
            raise theano.gof.utils.MethodNotDefined()

        dtype = node.inputs[0].type.dtype_specs()[1]
        fct_call = self.c_code_contiguous_raw(dtype, "n", "x", "z")
        return f"""
{{
        npy_intp n = PyArray_SIZE({z});
        {dtype} * x = ({dtype}*) PyArray_DATA({x});
        {dtype} * z = ({dtype}*) PyArray_DATA({z});
        {fct_call};
}}
        """

    def c_code_contiguous_raw(self, dtype, n, i, o):
        if not config.lib.amdlibm:
            raise theano.gof.utils.MethodNotDefined()
        if dtype.startswith("npy_"):
            dtype = dtype[4:]
        if dtype == "float32" and self.amd_float32 is not None:
            dtype = "float"
            fct = self.amd_float32
        elif dtype == "float64" and self.amd_float64 is not None:
            dtype = "double"
            fct = self.amd_float64
        else:
            raise theano.gof.utils.MethodNotDefined()
        return f"{fct}({n}, {i}, {o})"


class BinaryScalarOp(ScalarOp):
    # One may define in subclasses the following fields:
    #   - `identity`: for an associative operation, identity corresponds to
    #     the neutral element. For instance, it will be 0 for addition, 1 for
    #     multiplication, True for "and", False for "or".
    #   - `commutative`: whether op(a, b) == op(b, a)
    #   - `associative`: whether op(op(a, b), c) == op(a, op(b, c))
    nin = 2


###############
# Comparisons
###############


class LogicalComparison(BinaryScalarOp):
    def __init__(self, *args, **kwargs):
        BinaryScalarOp.__init__(self, *args, **kwargs)
        # This is for compat with old pickles.
        self.bool = True

    def __eq__(self, other):
        return BinaryScalarOp.__eq__(self, other) and getattr(
            self, "bool", False
        ) == getattr(other, "bool", False)

    def __hash__(self):
        # bool should always be True
        return BinaryScalarOp.__hash__(self)

    def output_types(self, *input_dtypes):
        return [bool] if getattr(self, "bool", False) else [int8]

    def L_op(self, inputs, outputs, output_gradients):
        x, y = inputs
        assert outputs[0].type == bool
        return [
            x.zeros_like().astype(theano.config.floatX),
            y.zeros_like().astype(theano.config.floatX),
        ]

    def c_code_cache_version(self):
        super_version = super().c_code_cache_version()
        return super_version + (0,)


class FixedLogicalComparison(UnaryScalarOp):
    """
    Comparison to a fixed value.

    """

    def __init__(self, *args, **kwargs):
        UnaryScalarOp.__init__(self, *args, **kwargs)
        # This is for compat with old pickles
        self.bool = True

    def __eq__(self, other):
        return UnaryScalarOp.__eq__(self, other) and getattr(
            self, "bool", False
        ) == getattr(other, "bool", False)

    def __hash__(self):
        # bool should always be True
        return UnaryScalarOp.__hash__(self)

    def output_types(self, *input_dtypes):
        return [bool] if getattr(self, "bool", False) else [int8]

    def L_op(self, inputs, outputs, output_gradients):
        (x,) = inputs
        assert outputs[0].type == bool
        return [x.zeros_like().astype(theano.config.floatX)]

    def c_code_cache_version(self):
        super_version = super().c_code_cache_version()
        return super_version + (0,)


class LT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("less", 2, 1)

    def impl(self, x, y):
        # built-in < don't support complex
        return np.less(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} < {y});"


lt = LT()


class GT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("greater", 2, 1)

    def impl(self, x, y):
        # built-in > don't support complex
        return np.greater(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} > {y});"


gt = GT()


class LE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("less_equal", 2, 1)

    def impl(self, x, y):
        # built-in <= don't support complex
        return np.less_equal(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} <= {y});"


le = LE()


class GE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ("greater_equal", 2, 1)

    def impl(self, x, y):
        # built-in >= don't support complex
        return np.greater_equal(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} >= {y});"


ge = GE()


class EQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    nfunc_spec = ("equal", 2, 1)

    def impl(self, x, y):
        return x == y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} == {y});"


eq = EQ()


class NEQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    nfunc_spec = ("not_equal", 2, 1)

    def impl(self, x, y):
        return x != y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = ({x} != {y});"


neq = NEQ()


class IsNan(FixedLogicalComparison):
    nfunc_spec = ("isnan", 1, 1)

    def impl(self, x):
        return np.isnan(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        # Discrete type can never be nan
        if node.inputs[0].type in discrete_types:
            return f"{z} = false;"

        # Windows tries to be different and sometimes return -1, but we want
        # to be consistent with numpy (which returns True), hence the "abs".
        return f"{z} = abs(isnan({x}));"

    def c_code_cache_version(self):
        scalarop_version = super().c_code_cache_version()
        return tuple(scalarop_version) + (3,)


isnan = IsNan()


class IsInf(FixedLogicalComparison):
    nfunc_spec = ("isinf", 1, 1)

    def impl(self, x):
        return np.isinf(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        # Discrete type can never be inf
        if node.inputs[0].type in discrete_types:
            return f"{z} = false;"

        # Note that the C isinf returns -1 for -Inf and +1 for +Inf, while
        # numpy simply returns True: we mimic numpy's behavior here, thus
        # the absolute value.
        return f"{z} = abs(isinf({x}));"

    def c_code_cache_version(self):
        scalarop_version = super().c_code_cache_version()
        return tuple(scalarop_version) + (3,)


isinf = IsInf()


class InRange(LogicalComparison):
    nin = 3

    def __init__(self, openlow, openhi):
        self.openlow = openlow
        self.openhi = openhi

    def impl(self, x, low, hi):
        if self.openlow and x <= low:
            return False
        elif not self.openlow and x < low:
            return False
        if self.openhi and x >= hi:
            return False
        elif not self.openhi and x > hi:
            return False
        return True

    def c_code(self, node, name, inputs, outputs, sub):
        (x, low, hi) = inputs
        (z,) = outputs

        cmp1 = ">" if self.openlow else ">="
        cmp2 = "<" if self.openhi else "<="

        return f"{z} = {x} {cmp1} {low} && {x} {cmp2} {hi};"

    def get_grad(self, elem):
        if elem.type in complex_types:
            msg = (
                "No gradient implemented for complex numbers in "
                "class scalar.basic.InRange"
            )
            raise NotImplementedError(msg)
        elif elem.type in discrete_types:
            return elem.zeros_like().astype(theano.config.floatX)
        else:
            return elem.zeros_like()

    def L_op(self, inputs, outputs, gout):
        (x, low, hi) = inputs
        (gz,) = gout
        grads = []
        for elem in [x, low, hi]:
            grads.append(self.get_grad(elem))
        return grads


inopenrange = InRange(True, True)
inclosedrange = InRange(False, False)


class Switch(ScalarOp):
    nin = 3
    nfunc_spec = ("where", 3, 1)

    def impl(self, cond, ift, iff):
        return ift if cond else iff

    def c_code(self, node, name, inputs, outputs, sub):
        (cond, ift, iff) = inputs
        (z,) = outputs
        return f"{z} = {cond} ? {ift} : {iff};"

    def L_op(self, inputs, outputs, gout):
        (cond, ift, iff) = inputs
        (gz,) = gout
        first_part = switch(cond, gz, 0.0)
        second_part = switch(cond, 0.0, gz)

        if outputs[0].type.dtype in discrete_types:
            first_part = 0.0
            second_part = 0.0

        # cond does affect the elements of the output so it is connected.
        # For the sake of making the gradient convenient we assume that
        # condition + epsilon always triggers the same branch as condition
        condition_grad = cond.zeros_like().astype(theano.config.floatX)

        return (condition_grad, first_part, second_part)

    def output_types(self, types):
        (cond_t, ift_t, iff_t) = types
        return upcast_out(ift_t, iff_t)


switch = Switch()

####################
# BIT-WISE OPERATORS
####################


class UnaryBitOp(UnaryScalarOp):
    def output_types(self, *input_types):
        for i in input_types[0]:
            if i not in discrete_types:
                raise TypeError(
                    "input to a BitOp must have type (u)int8, "
                    f"(u)int16, (u)int32 or (u)int64 or bool not {i}"
                )
        return upcast_out(*input_types[0])

    def grad(self, inputs, output_gradients):
        return [inputs[0].zeros_like().astype(theano.config.floatX)]


class BinaryBitOp(BinaryScalarOp):
    def output_types(self, *input_types):
        t0, t1 = input_types[0]
        if t0 == bool and t1 == bool:
            return [bool]
        for i in input_types[0]:
            if i not in integer_types:
                raise TypeError(
                    "input to a BitOp must have type (u)int8, "
                    "(u)int16, (u)int32 or (u)int64 or "
                    f"be all bools not {i}"
                )
        return upcast_out(*input_types[0])

    def grad(self, inputs, output_gradients):
        a, b = inputs
        return [
            a.zeros_like().astype(theano.config.floatX),
            b.zeros_like().astype(theano.config.floatX),
        ]


class OR(BinaryBitOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ("bitwise_or", 2, 1)

    def impl(self, x, y):
        return x | y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} | {y});"


or_ = OR()


class XOR(BinaryBitOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ("bitwise_xor", 2, 1)

    def impl(self, x, y):
        return x ^ y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} ^ {y});"


xor = XOR()


class AND(BinaryBitOp):
    identity = -1
    commutative = True
    associative = True
    nfunc_spec = ("bitwise_and", 2, 1)

    def impl(self, x, y):
        return x & y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = ({x} & {y});"

    def c_code_cache_version(self):
        super_version = super().c_code_cache_version()
        return super_version + (3,)


and_ = AND()


class Invert(UnaryBitOp):
    nfunc_spec = ("invert", 1, 1)

    def impl(self, x):
        return ~x

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type == bool:
            return f"{z} = (!{x});"
        return f"{z} = (~{x});"


invert = Invert()


##############
# Arithmetic
##############
class Maximum(BinaryScalarOp):
    commutative = True
    associative = True
    nfunc_spec = ("maximum", 2, 1)
    nfunc_variadic = "maximum"

    def impl(self, *inputs):
        # The built-in max function don't support complex type
        return np.maximum(*inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if any([i.type in complex_types for i in node.inputs]):
            raise NotImplementedError()
        # Test for both y>x and x>=y to detect NaN
        return f"{z} = (({y})>({x})? ({y}): " f'(({x})>=({y})? ({x}): nan("")));'

    def L_op(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            # max is currently defined for complex_types,
            # but the gradient for complex is not.
            raise NotImplementedError()

        if outputs[0].type in discrete_types:
            return [
                x.zeros_like().astype(theano.config.floatX),
                y.zeros_like().astype(theano.config.floatX),
            ]
        # This form handle the case when both value are the same.
        # In that case, gx will be gz, gy will be 0.
        e = eq(outputs[0], x)
        gx = e * gz
        gy = (constant(1, dtype=gz.dtype) - e) * gz
        return (gx, gy)


maximum = Maximum(upcast_out, name="maximum")


class Minimum(BinaryScalarOp):
    commutative = True
    associative = True
    nfunc_spec = ("minimum", 2, 1)
    nfunc_variadic = "minimum"

    def impl(self, *inputs):
        # The built-in min function don't support complex type
        return np.minimum(*inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if any([i.type in complex_types for i in node.inputs]):
            raise NotImplementedError()
        return f"{z} = (({y})<({x})? ({y}): " f'(({x})<=({y})? ({x}): nan("")));'

    def L_op(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            # min is currently defined for complex_types,
            # but the gradient for complex is not.
            raise NotImplementedError()

        if outputs[0].type in discrete_types:
            return [
                x.zeros_like().astype(theano.config.floatX),
                y.zeros_like().astype(theano.config.floatX),
            ]
        # This form handle the case when both value are the same.
        # In that case, gx will be gz, gy will be 0.
        e = eq(outputs[0], x)
        gx = e * gz
        gy = (constant(1, dtype=gz.dtype) - e) * gz
        return (gx, gy)


minimum = Minimum(upcast_out, name="minimum")


class Add(ScalarOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ("add", 2, 1)
    nfunc_variadic = "sum"

    def impl(self, *inputs):
        return sum(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        op = " + "
        if node.outputs[0].type == bool:
            op = " || "
        if not inputs:
            return z + " = 0;"
        else:
            return z + " = " + op.join(inputs) + ";"

    def L_op(self, inputs, outputs, gout):
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            assert gz is not None
            retval = []
            for ii, inp in enumerate(inputs):
                if hasattr(inp, "zeros_like"):
                    retval.append(inp.zeros_like().astype(theano.config.floatX))
                else:
                    retval.append(grad_undefined(self, ii, inp))
        else:
            retval = []
            for i in inputs:
                retval += [gz]
        return retval


add = Add(upcast_out, name="add")


class Mul(ScalarOp):
    identity = 1
    commutative = True
    associative = True
    nfunc_spec = ("multiply", 2, 1)
    nfunc_variadic = "product"

    def impl(self, *inputs):
        return np.product(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        op = " * "
        if node.outputs[0].type == bool:
            op = " && "
        if not inputs:
            return z + " = 1;"
        else:
            return z + " = " + op.join(inputs) + ";"

    def grad(self, inputs, gout):
        (gz,) = gout
        retval = []

        # The following 3 lines verify that gz is complex when the
        # output is complex. The rest of this function make this supposition.
        output_type = self.output_types([i.type for i in inputs])[0]
        if output_type in complex_types:
            if gz.type not in complex_types:
                raise TypeError(
                    "Mul with output_type "
                    + str(output_type)
                    + " expected gz type to be complex, got gz with type "
                    + str(gz.type)
                )

        if output_type in discrete_types:
            return [ipt.zeros_like().astype(theano.config.floatX) for ipt in inputs]

        for input in inputs:
            if gz.type in complex_types:
                # zr+zi = (xr + xi)(yr + yi)
                # zr+zi = (xr*yr - xi*yi) + (xr yi + xi yr )
                otherprod = mul(*(utils.difference(inputs, [input])))
                yr = real(otherprod)
                yi = imag(otherprod)
                if input.type in complex_types:
                    retval += [
                        complex(
                            yr * real(gz) + yi * imag(gz), yr * imag(gz) - yi * real(gz)
                        )
                    ]
                else:
                    retval += [yr * real(gz) + yi * imag(gz)]
            else:
                retval += [mul(*([gz] + utils.difference(inputs, [input])))]
        return retval


mul = Mul(upcast_out, name="mul")


class Sub(BinaryScalarOp):
    nfunc_spec = ("subtract", 2, 1)

    def impl(self, x, y):
        return x - y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = {x} - {y};"

    def L_op(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            return [
                x.zeros_like().astype(theano.config.floatX),
                y.zeros_like().astype(theano.config.floatX),
            ]

        first_part = gz
        second_part = -gz

        return first_part, second_part


sub = Sub(upcast_out_nobool, name="sub")


def int_or_true_div(x_discrete, y_discrete):
    """
    Return 'int' or 'true' depending on the type of division used for x / y.

    Parameters
    ----------
    x_discrete : bool
        True if `x` is discrete ([unsigned] integer).
    y_discrete : bool
        True if `y` is discrete ([unsigned] integer).

    Returns
    -------
    str
        'int' if `x / y` should be an integer division, or `true` if it
        should be a true division.

    Raises
    ------
    IntegerDivisionError
        If both `x_discrete` and `y_discrete` are True and `config.int_division`
        is set to 'raise'.

    Notes
    -----
    This function is used by both scalar/basic.py and tensor/basic.py.

    """
    if x_discrete and y_discrete:
        if config.int_division == "raise":
            raise IntegerDivisionError(
                "With `config.int_division` set to 'raise', dividing two "
                "integer types with '/' is forbidden to avoid confusion "
                "between integer and floating point divisions. Please "
                "use // for integer division, or if you want a float result "
                "either cast one of the arguments to a float or directly call "
                "`x.__truediv__(y)`."
            )
        elif config.int_division == "int":
            warnings.warn(
                "Division of two integer types with x / y is deprecated, "
                "please use x // y for an integer division.",
                DeprecationWarning,
                stacklevel=4,
            )
            return int_div
        elif config.int_division == "floatX":
            return true_div
        else:
            raise NotImplementedError(config.int_division)
    else:
        return true_div


def div_proxy(x, y):
    """
    Proxy for either true_div or int_div, depending on types of x, y.

    """
    f = int_or_true_div(
        as_scalar(x).type in discrete_types, as_scalar(y).type in discrete_types
    )
    return f(x, y)


class TrueDiv(BinaryScalarOp):
    nfunc_spec = ("true_divide", 2, 1)

    def output_types(self, types):
        if all(t in discrete_types for t in types):
            return [get_scalar_type(config.floatX)]
        else:
            return super().output_types(types)

    def impl(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if all(a.dtype in discrete_types for a in (x, y)):
            return np.sctypeDict[config.floatX](float(x) / y)
        else:
            return x / y

    def c_code(self, node, name, inputs, outputs, sub):
        # we generate good c code only when both are complex!
        (x, y) = inputs
        (z,) = outputs
        if (
            sum(
                [
                    node.inputs[0].type in complex_types,
                    node.inputs[1].type in complex_types,
                ]
            )
            == 1
        ):
            raise NotImplementedError("type not supported", type)
        if (
            node.inputs[0].type in discrete_types
            and node.inputs[1].type in discrete_types
        ):
            return f"{z} = ((double){x}) / {y};"
        return f"{z} = {x} / {y};"

    def grad(self, inputs, gout):

        (x, y) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()

        # If the output of this op is discrete, then it
        # it is locally flat everywhere, so the gradient
        # through it is 0.
        # This is different from it not being connected
        # to the output; x/y is still a function of x
        # and y; it's just a step function.
        if all(a.dtype in discrete_types for a in (x, y)):
            return [x.zeros_like(), y.zeros_like()]

        first_part = gz / y

        if y.type in complex_types:
            raise NotImplementedError()

        second_part = -(gz * x) / (y * y)

        return first_part, second_part


true_div = TrueDiv(upcast_out, name="true_div")


class IntDiv(BinaryScalarOp):
    nfunc_spec = ("floor_divide", 2, 1)
    complex_error = ComplexError(
        "Theano does not support integer division (//) on "
        "complex numbers, since numpy deprecated it."
    )

    def impl(self, x, y):
        return x // y

    def c_support_code(self):
        # We use a macro as python use % as a special string character,
        # and the output of c_code may be run through another level
        # of string formatting.
        return "#define THEANO_MACRO_MOD(x,y) (x % y)"

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        fail = sub["fail"]

        t = node.inputs[0].type.upcast(*[i.type for i in node.inputs[1:]])
        if t in map(str, discrete_types):
            x_div_y_pp = f"({x} / {y})"
            x_div_y_mp = f"((-{x}) / {y})"
            x_mod_y_mp = f"THEANO_MACRO_MOD((-{x}), {y})"
            x_div_y_pm = f"({x} / (-{y}))"
            x_mod_y_pm = f"THEANO_MACRO_MOD({x}, (-{y}))"
            x_div_y_mm = f"((-{x}) / (-{y}))"
            # If we are in a gpuarray kernel, %(fail)s exits the kernel,
            # and we do not have any error report, and we cannot set
            # Python error messages either, so for now we just call the
            # cuda function, which return a binary pattern of all 1s.
            div_zero = dedent(
                f"""
                #ifdef KERNEL
                    {z} = {x_div_y_pp};
                #else
                    PyErr_SetString(PyExc_ZeroDivisionError, "integer division by zero");
                    {fail}
                #endif
                """
            )
        elif t in map(str, float_types):
            # We need to call different functions of math.h
            # depending on the type
            if t == "float32":
                floor = "floorf"
                fmod = "fmodf"
            elif t == "float64":
                floor = "floor"
                fmod = "fmod"
            else:
                raise NotImplementedError("type not supported", t)

            x_div_y_pp = f"{floor}({x} / {y})"
            x_div_y_mp = f"{floor}((-{x}) / {y})"
            x_mod_y_mp = f"{fmod}((-{x}), {y})"
            x_div_y_pm = f"{floor}({x} / (-{y}))"
            x_mod_y_pm = f"{fmod}({x}, (-{y}))"
            x_div_y_mm = f"{floor}((-{x}) / (-{y}))"
            div_zero = f"{z} = {x_div_y_pp};"
        elif t in complex_types:
            raise self.complex_error
        else:
            raise NotImplementedError("type not supported", t)

        return dedent(
            f"""
            if ({y} == 0) {{
                {div_zero};
            }} else if ({y} < 0) {{
                if ({x} < 0) {{
                    {z} = {x_div_y_mm};
                }} else {{
                    {z} = - {x_div_y_pm} - (({x_mod_y_pm} == 0) ? 0 : 1);
                }}
            }} else {{
                if ({x} < 0) {{
                    {z} = - {x_div_y_mp} - (({x_mod_y_mp} == 0) ? 0 : 1);
                }} else {{
                    {z} = {x_div_y_pp};
                }}
            }}
            """
        )

    def c_code_cache_version(self):
        return (6,)

    def grad(self, inputs, g_output):
        return [inp.zeros_like(dtype=theano.config.floatX) for inp in inputs]


int_div = IntDiv(upcast_out, name="int_div")


floor_div = int_div


def mod_check(x, y):
    if as_scalar(x).type in complex_types or as_scalar(y).type in complex_types:
        # Currently forbidden.
        raise Mod.complex_error
    else:
        return mod(x, y)


class Mod(BinaryScalarOp):
    nfunc_spec = ("mod", 2, 1)
    complex_error = ComplexError(
        "Theano does not support the mod operator (%) on "
        "complex numbers, since numpy deprecated it."
    )

    def impl(self, x, y):
        if isinstance(x, complex) or isinstance(y, complex): # PMB changed np.complex to complex
            raise self.complex_error
        return x % y

    def c_code_cache_version(self):
        return (9,)

    def c_support_code(self):
        # We use a macro as python use % as a special string character,
        # and the output of c_code may be run through another level
        # of string formatting.
        return "#define THEANO_MACRO_MOD(x, y) (x % y)"

    def c_code(self, node, name, inputs, outputs, sub):
        """
        We want the result to have the same sign as Python, not the other
        implementation of mod.

        """
        (x, y) = inputs
        (z,) = outputs
        fail = sub["fail"]
        t = node.inputs[0].type.upcast(*[i.type for i in node.inputs[1:]])
        if (
            str(t) in map(str, discrete_types)
            or t in ["uint8", "int8", "uint16", "int16"]
            or t in ["uint32", "int32", "uint64", "int64"]
            or t in discrete_types
        ):
            # The above or's should not be needed anymore. However, for now we
            # keep them out of safety, and verify they are useless with an
            # assert.
            assert str(t) in map(str, discrete_types)
            x_mod_y = f"THEANO_MACRO_MOD({x}, {y})"
            x_mod_ymm = f"THEANO_MACRO_MOD(-{x}, -{y})"
            x_mod_ypm = f"THEANO_MACRO_MOD({x}, -{y})"
            x_mod_ymp = f"THEANO_MACRO_MOD(-{x}, {y})"
            # If we are in a gpuarray kernel, %(fail)s exits the kernel,
            # and we do not have any error report, and we cannot set
            # Python error messages either, so for now we just call the
            # cuda function, returning a binary pattern depending on dtype
            mod_zero = dedent(
                f"""
                #ifdef KERNEL
                    {z} = {x_mod_y};
                #else
                    PyErr_SetString(PyExc_ZeroDivisionError, "integer modulo by zero");
                    {fail}
                #endif
                """
            )
        elif (
            str(t) in map(str, float_types)
            or t in ["float32", "float64"]
            or t in float_types
        ):
            # The above or's should not be needed anymore. However, for now we
            # keep them out of safety, and verify they are useless with an
            # assert.
            assert str(t) in map(str, float_types)
            x_mod_y = f"fmod({x}, {y})"
            x_mod_ymm = f"fmod(-{x}, -{y})"
            x_mod_ypm = f"fmod({x}, -{y})"
            x_mod_ymp = f"fmod(-{x}, {y})"
            mod_zero = f"{z} = {x_mod_y};"
        elif str(t) in map(str, complex_types):
            raise self.complex_error
        else:
            raise NotImplementedError("type not supported", t)

        return dedent(
            f"""
            if ({y} == 0) {{
                {mod_zero};
            }} else if ({y} < 0){{
                if ({x} < 0){{
                    {z} = -({x_mod_ymm});
                }} else {{
                    {z} = ({x_mod_ypm}) + ({x_mod_ypm} != 0 ? {y} : 0);
                }}
            }} else {{
                if ({x} < 0){{
                    {z} = - {x_mod_ymp} + ({x_mod_ymp} != 0 ? {y} : 0);
                }} else {{
                    {z} = {x_mod_y};
                }}
            }}
            """
        )

    def L_op(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if outputs[0].type.dtype in discrete_types:
            # The gradient does not flow in if the output is discrete
            return [
                x.zeros_like(dtype=theano.config.floatX),
                y.zeros_like(dtype=theano.config.floatX),
            ]
        return [gz, -(x // y) * gz]


mod = Mod(upcast_out, name="mod")


class Pow(BinaryScalarOp):
    nfunc_spec = ("power", 2, 1)

    def impl(self, x, y):
        return x ** y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types or node.inputs[1].type in complex_types:
            raise NotImplementedError("type not supported", type)
        return f"{z} = pow({x}, {y});"

    def L_op(self, inputs, outputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()

        if outputs[0].type in discrete_types:
            return [
                x.zeros_like().astype(theano.config.floatX),
                y.zeros_like().astype(theano.config.floatX),
            ]

        first_part = gz * y * x ** (y - 1)

        second_part = gz * log(x) * x ** y
        second_part = switch(eq(x, 0), 0, second_part)

        return (first_part, second_part)

    def c_code_contiguous(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if not theano.config.lib.amdlibm:
            raise theano.gof.utils.MethodNotDefined()

        # We compare the dtype AND the broadcast flag
        # as this function do not broadcast
        if (
            node.inputs[0].type == node.outputs[0].type
            and node.inputs[1].type == node.outputs[0].type
            and
            # amdlibm 3.0 do not have a float64 version of this SIMD function
            node.inputs[0].dtype == "float32"
            and node.inputs[1].dtype == "float32"
        ):
            dtype = "float"
            fct = "amd_vrsa_powf"
            return f"""
        npy_intp n = PyArray_SIZE({z});
        {dtype} * x = ({dtype}*) PyArray_DATA({x});
        {dtype} * y = ({dtype}*) PyArray_DATA({y});
        {dtype} * z = ({dtype}*) PyArray_DATA({z});
        {fct}(n, x, y, z);
        """
        # We compare the dtype and check we broadcast a scalar
        elif (
            node.inputs[0].type == node.outputs[0].type
            and node.inputs[1].dtype == node.outputs[0].dtype
            and all(node.inputs[1].broadcastable)
            and
            # amdlibm 3.0 do not have a float64 version of this SIMD function
            node.inputs[0].dtype == "float32"
            and node.inputs[1].dtype == "float32"
        ):
            dtype = "float"
            fct = "amd_vrsa_powxf"
            return f"""
        npy_intp n = PyArray_SIZE({z});
        {dtype} * x = ({dtype}*) PyArray_DATA({x});
        {dtype} * y = ({dtype}*) PyArray_DATA({y});
        {dtype} * z = ({dtype}*) PyArray_DATA({z});
        {fct}(n, x, *y, z);
        """

        raise theano.gof.utils.MethodNotDefined()


pow = Pow(upcast_out_min8, name="pow")


class Clip(ScalarOp):
    nin = 3
    # The numpy.clip don't work correctly when the min is bigger then the max,
    # So we do not use nfunc_spec = ('clip', 3, 1)

    def impl(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        else:
            return x

    def c_code(self, node, name, inputs, outputs, sub):
        (x, min, max) = inputs
        (z,) = outputs
        return f"{z} = {x} < {min} ? {min} : {x} > {max} ? {max} : {x};"

    def L_op(self, inputs, outputs, gout):
        (x, mn, mx) = inputs
        (gz,) = gout
        assert gz.type not in complex_types
        gx = ((x >= mn) & (x <= mx)) * gz
        gmn = (x < mn) * gz
        gmx = (x > mx) * gz

        def handle_int(v):
            if outputs[0].type in int_types:
                return v.zeros_like().astype(config.floatX)
            return v

        return list(map(handle_int, [gx, gmn, gmx]))


# Don't allow complex even if numpy do
# As there is no mathematical reason for this function on complex
clip = Clip(upcast_out_no_complex, name="clip")


class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return f"{z} = {y};"

    def connection_pattern(self, node):

        # x is never connected because its elements are never used
        # y is connected because its elements are copied over

        return [[False], [True]]

    def grad(self, inputs, gout):

        (x, y) = inputs
        (gz,) = gout
        if y.type in continuous_types:
            # x is disconnected because the elements of x are not used
            return DisconnectedType()(), gz
        else:
            # when y is discrete, we assume the function can be extended
            # to deal with real-valued inputs by rounding them to the
            # nearest integer. f(x+eps) thus equals f(x) so the gradient
            # is zero, not disconnected or undefined
            return DisconnectedType()(), y.zeros_like()


second = Second(transfer_type(1), name="second")


class Identity(UnaryScalarOp):
    def impl(self, input):
        return input

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x};"

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in continuous_types:
            return (gz,)
        else:
            return (x.zeros_like(dtype=theano.config.floatX),)


identity = Identity(same_out, name="identity")


# CASTING OPERATIONS
class Cast(UnaryScalarOp):
    def __init__(self, o_type, name=None):
        if not isinstance(o_type, Scalar):
            raise TypeError(o_type)
        super().__init__(specific_out(o_type), name=name)
        self.o_type = o_type
        # PMB changed this
        #self.ctor = getattr(np, o_type.dtype)
        if o_type.dtype=='bool':  # PMB added for numpy > 1.23.5 
             self.ctor = np.bool_  # PMB added for numpy > 1.23.5 
        else:                     # PMB added for numpy > 1.23.5 
             self.ctor = getattr(np, o_type.dtype)  # PMB added for numpy > 1.23.5 

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.o_type.dtype}}}"

    def clone_float32(self):
        if self.o_type == float16:
            return convert_to_float32
        return self

    def make_new_inplace(self, output_types_preference=None, name=None):
        """
        This op.__init__ fct don't have the same parameter as other scalar op.
        This breaks the insert_inplace_optimizer optimization.
        This function is a fix to patch this, by ignoring the
        output_types_preference passed by the optimization, and replacing it
        by the current output type. This should only be triggered when
        both input and output have the same dtype anyway.

        """
        return self.__class__(self.o_type, name)

    def impl(self, input):
        return self.ctor(input)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type == bool:
            return f"{z} = ({x}) ? 1 : 0;"
        return f"{z} = ({node.outputs[0].type.dtype_specs()[1]}){x};"

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self.o_type in continuous_types:
            return [gz]
        else:
            return [x.zeros_like().astype(theano.config.floatX)]

    def c_code_cache_version(self):
        s = super().c_code_cache_version()
        if s:
            return (4,) + s
        else:
            return s


convert_to_bool = Cast(bool, name="convert_to_bool")
convert_to_int8 = Cast(int8, name="convert_to_int8")
convert_to_int16 = Cast(int16, name="convert_to_int16")
convert_to_int32 = Cast(int32, name="convert_to_int32")
convert_to_int64 = Cast(int64, name="convert_to_int64")
convert_to_uint8 = Cast(uint8, name="convert_to_uint8")
convert_to_uint16 = Cast(uint16, name="convert_to_uint16")
convert_to_uint32 = Cast(uint32, name="convert_to_uint32")
convert_to_uint64 = Cast(uint64, name="convert_to_uint64")
convert_to_float16 = Cast(float16, name="convert_to_float16")
convert_to_float32 = Cast(float32, name="convert_to_float32")
convert_to_float64 = Cast(float64, name="convert_to_float64")
convert_to_complex64 = Cast(complex64, name="convert_to_complex64")
convert_to_complex128 = Cast(complex128, name="convert_to_complex128")

_cast_mapping = {
    "bool": convert_to_bool,
    "int8": convert_to_int8,
    "int16": convert_to_int16,
    "int32": convert_to_int32,
    "int64": convert_to_int64,
    "uint8": convert_to_uint8,
    "uint16": convert_to_uint16,
    "uint32": convert_to_uint32,
    "uint64": convert_to_uint64,
    "float16": convert_to_float16,
    "float32": convert_to_float32,
    "float64": convert_to_float64,
    "complex64": convert_to_complex64,
    "complex128": convert_to_complex128,
}


def cast(x, dtype):
    """
    Symbolically cast `x` to a Scalar of given `dtype`.

    """
    if dtype == "floatX":
        dtype = config.floatX

    _x = as_scalar(x)
    if _x.type.dtype == dtype:
        return _x
    if _x.type.dtype.startswith("complex") and not dtype.startswith("complex"):
        raise TypeError(
            "Casting from complex to real is ambiguous: consider"
            " real(), imag(), angle() or abs()"
        )
    return _cast_mapping[dtype](_x)


class Abs(UnaryScalarOp):
    nfunc_spec = ("abs", 1, 1)

    def make_node(self, x):
        inputs = [as_scalar(input) for input in [x]]
        if inputs[0].type == complex64:
            outputs = [float32()]
        elif inputs[0].type == complex128:
            outputs = [float64()]
        else:
            outputs = [t() for t in self.output_types([input.type for input in inputs])]
        return Apply(self, inputs, outputs)

    def impl(self, x):
        return np.abs(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        if x.type in float_types:
            return (gz * sgn(x),)
        return (gz * x / abs(x),)  # formula works for complex and real

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in int_types:
            return f"{z} = abs({x});"
        if type in float_types:
            return f"{z} = fabs({x});"
        if type in complex_types:
            return f"{z} = sqrt({x}.real*{x}.real + {x}.imag*{x}.imag);"
        if node.outputs[0].type == bool:
            return f"{z} = ({x}) ? 1 : 0;"
        if type in uint_types:
            # uint are always already absolute value.
            return f"{z} = {x};"
        raise NotImplementedError("type not supported", type)


abs_ = Abs(same_out)


class Sgn(UnaryScalarOp):
    nfunc_spec = ("sign", 1, 1)

    @staticmethod
    def output_types_preference(x):
        if x == bool:
            raise TypeError(x)
        return same_out_nocomplex(x)

    def impl(self, x):
        # casting to output type is handled by filter
        return np.sign(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        # casting is done by compiler
        # TODO: use copysign
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in float_types:
            return (
                f"{z} = ({x} > 0) ? 1. : (({x} < 0) ? -1. : (isnan({x}) ? NAN : 0.));"
            )
        if type in int_types:
            return f"{z} = ({x} >= 0) ? ({x} == 0) ? 0 : 1 : -1;"
        raise ComplexError("complex has no sgn")

    def c_code_cache_version(self):
        s = super().c_code_cache_version()
        if s:
            return (4,) + s
        else:  # if parent is unversioned, we are too
            return s


sgn = Sgn(name="sgn")


class Ceil(UnaryScalarOp):
    nfunc_spec = ("ceil", 1, 1)

    def impl(self, x):
        return np.ceil(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = ceil(({cast}){x});"


ceil = Ceil(upgrade_to_float_no_complex, name="ceil")


class Floor(UnaryScalarOp):
    nfunc_spec = ("floor", 1, 1)

    def impl(self, x):
        return np.floor(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = floor(({cast}){x});"


floor = Floor(upgrade_to_float_no_complex, name="floor")


class Trunc(UnaryScalarOp):
    nfunc_spec = ("trunc", 1, 1)

    def impl(self, x):
        return np.trunc(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return [x.zeros_like().astype(theano.config.floatX)]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x} >= 0? floor({x}): -floor(-{x});"


trunc = Trunc(upgrade_to_float_no_complex, name="trunc")


class RoundHalfToEven(UnaryScalarOp):
    """
    This function implement the same rounding than numpy: Round half to even.

    c/c++ round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.

    """

    nfunc_spec = ("around", 1, 1)

    def impl(self, x):
        return np.round(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        typ = node.outputs[0].type.dtype
        if typ not in ["float32", "float64"]:
            raise NotImplementedError("The output should be float32 or float64")
        if typ == "float32":
            ctype = "float"
            floor_function = "floorf"
        else:
            ctype = "double"
            floor_function = "floor"
        return f"""
        /* Code inspired from NumPy npy_rint implementation. */
        {{
            {ctype} y, r;
            y = {floor_function}({x});
            r = {x} - y;
            if(r > 0.5) {{
                y += 1;
            }} else if(r == 0.5) {{
                r = y - 2.0*{floor_function}(0.5*y);
                /*
                If y is even, then r == 0
                If y is odd,  then r == 1
                So we can just add r to y, so that
                y will be incremented only if he's odd.
                */
                y += (int)r;
            }}
            {z} = y;
        }}
        """


round_half_to_even = RoundHalfToEven(same_out_float_only)


def round_half_away_from_zero_(a):
    if a > 0:
        return np.floor(a + 0.5)
    else:
        return np.ceil(a - 0.5)


round_half_away_from_zero_vec64 = np.vectorize(
    round_half_away_from_zero_, doc="round_half_away_from_zero_vec64"
)
round_half_away_from_zero_vec32 = np.vectorize(
    round_half_away_from_zero_,
    doc="round_half_away_from_zero_vec32",
    otypes=["float32"],
)


def round_half_away_from_zero_vec(a):
    if getattr(a, "dtype", None) == np.float32:
        return round_half_away_from_zero_vec32(a)
    return round_half_away_from_zero_vec64(a)


class RoundHalfAwayFromZero(UnaryScalarOp):
    """
    Implement the same rounding algo as c round() fct.

    numpy.round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.

    """

    def impl(self, x):
        return round_half_away_from_zero_vec(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type.dtype in ["float32", "float64"]:
            return f"{z} = round({x});"
        else:
            raise NotImplementedError("The output should be float32 or float64")


round_half_away_from_zero = RoundHalfAwayFromZero(same_out_float_only)


class Neg(UnaryScalarOp):
    # We can use numpy.negative here, because even if it gives unexpected
    # results on Boolean arrays, it will be passed other dtypes as Theano
    # does not have a Boolean type for tensors.
    nfunc_spec = ("negative", 1, 1)

    def impl(self, x):
        return -x

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz,)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = -{x};"


neg = Neg(same_out_nobool, name="neg")

pprint.assign(add, printing.OperatorPrinter("+", -2, "either"))
pprint.assign(mul, printing.OperatorPrinter("*", -1, "either"))
pprint.assign(sub, printing.OperatorPrinter("-", -2, "left"))
pprint.assign(neg, printing.OperatorPrinter("-", 0, "either"))
pprint.assign(true_div, printing.OperatorPrinter("/", -1, "left"))
pprint.assign(int_div, printing.OperatorPrinter("//", -1, "left"))
pprint.assign(pow, printing.OperatorPrinter("**", 1, "right"))
pprint.assign(mod, printing.OperatorPrinter("%", -1, "left"))


class Inv(UnaryScalarOp):
    """
    Multiplicative inverse. Also called reciprocal.

    """

    nfunc_spec = ("reciprocal", 1, 1)

    def impl(self, x):
        return np.float32(1.0) / x

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz / (x * x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return f"{z} = 1.0 / {x};"


inv = Inv(upgrade_to_float, name="inv")


class Log(UnaryScalarOp):
    """
    log base e.

    """

    nfunc_spec = ("log", 1, 1)
    amd_float32 = "amd_vrsa_logf"
    amd_float64 = "amd_vrda_log"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log(x, sig="f")
        return np.log(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / x,)

    def c_code(self, node, name, inputs, outputs, sub):
        # todo: the version using log2 seems to be very slightly faster
        # on some machines for some reason, check if it's worth switching
        # return f"{z} = log2({x}) * 0.69314718055994529;"
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log(({cast}){x});"


log = Log(upgrade_to_float, name="log")


class Log2(UnaryScalarOp):
    """
    log base 2.

    """

    nfunc_spec = ("log2", 1, 1)
    amd_float32 = "amd_vrsa_log2f"
    amd_float64 = "amd_vrda_log2"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log2 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log2(x, sig="f")
        return np.log2(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (x * np.asarray(math.log(2.0)).astype(x.dtype)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log2(({cast}){x});"


log2 = Log2(upgrade_to_float, name="log2")


class Log10(UnaryScalarOp):
    """
    log base 10.

    """

    nfunc_spec = ("log10", 1, 1)
    amd_float32 = "amd_vrsa_log10f"
    amd_float64 = "amd_vrda_log10"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log10 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log10(x, sig="f")
        return np.log10(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (x * np.log(10.0)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log10(({cast}){x});"


log10 = Log10(upgrade_to_float, name="log10")


class Log1p(UnaryScalarOp):
    """
    log(1+x).

    """

    nfunc_spec = ("log1p", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log1p will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.log1p(x, sig="f")
        return np.log1p(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz / (1 + x)]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = log1p(({cast}){x});"


log1p = Log1p(upgrade_to_float, name="log1p")


class Exp(UnaryScalarOp):
    nfunc_spec = ("exp", 1, 1)
    amd_float32 = "amd_vrsa_expf"
    amd_float64 = "amd_vrda_exp"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.exp(x, sig="f")
        return np.exp(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * exp(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = exp(({cast}){x});"


exp = Exp(upgrade_to_float, name="exp")


class Exp2(UnaryScalarOp):
    nfunc_spec = ("exp2", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp2 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.exp2(x, sig="f")
        return np.exp2(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * exp2(x) * log(np.cast[x.type](2)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = exp2(({cast}){x});"


exp2 = Exp2(upgrade_to_float, name="exp2")


class Expm1(UnaryScalarOp):
    nfunc_spec = ("expm1", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.expm1 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.expm1(x, sig="f")
        return np.expm1(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * exp(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = expm1(({cast}){x});"

    def c_code_cache_version(self):
        return (5,)


expm1 = Expm1(upgrade_to_float, name="expm1")


class Sqr(UnaryScalarOp):
    nfunc_spec = ("square", 1, 1)

    def impl(self, x):
        return x * x

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * x * 2,)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {x} * {x};"


sqr = Sqr(same_out, name="sqr")


class Sqrt(UnaryScalarOp):
    nfunc_spec = ("sqrt", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sqrt will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.sqrt(x, sig="f")
        return np.sqrt(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return ((gz * 0.5) / sqrt(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = sqrt(({cast}){x});"


sqrt = Sqrt(upgrade_to_float, name="sqrt")


class Deg2Rad(UnaryScalarOp):
    nfunc_spec = ("deg2rad", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.deg2rad will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.deg2rad(x, sig="f")
        return np.deg2rad(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * np.asarray(np.pi / 180, gz.type),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        return f"{z} = {x} * (M_PI / 180.0);"


deg2rad = Deg2Rad(upgrade_to_float, name="deg2rad")


class Rad2Deg(UnaryScalarOp):
    nfunc_spec = ("rad2deg", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.rad2deg will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.rad2deg(x, sig="f")
        return np.rad2deg(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * np.asarray(180.0 / np.pi, gz.type),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        return f"{z} = {x} * (180.0 / M_PI);"


rad2deg = Rad2Deg(upgrade_to_float, name="rad2deg")


class Cos(UnaryScalarOp):
    nfunc_spec = ("cos", 1, 1)
    amd_float32 = "amd_vrsa_cosf"
    amd_float64 = "amd_vrda_cos"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.cos will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.cos(x, sig="f")
        return np.cos(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz * sin(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = cos(({cast}){x});"


cos = Cos(upgrade_to_float, name="cos")


class ArcCos(UnaryScalarOp):
    nfunc_spec = ("arccos", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arccos will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arccos(x, sig="f")
        return np.arccos(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (-gz / sqrt(np.cast[x.type](1) - sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = acos(({cast}){x});"


arccos = ArcCos(upgrade_to_float, name="arccos")


class Sin(UnaryScalarOp):
    nfunc_spec = ("sin", 1, 1)
    amd_float32 = "amd_vrsa_sinf"
    amd_float64 = "amd_vrda_sin"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sin will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.sin(x, sig="f")
        return np.sin(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * cos(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = sin(({cast}){x});"


sin = Sin(upgrade_to_float, name="sin")


class ArcSin(UnaryScalarOp):
    nfunc_spec = ("arcsin", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arcsin will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arcsin(x, sig="f")
        return np.arcsin(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqrt(np.cast[x.type](1) - sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = asin(({cast}){x});"


arcsin = ArcSin(upgrade_to_float, name="arcsin")


class Tan(UnaryScalarOp):
    nfunc_spec = ("tan", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.tan will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.tan(x, sig="f")
        return np.tan(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqr(cos(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = tan(({cast}){x});"


tan = Tan(upgrade_to_float, name="tan")


class ArcTan(UnaryScalarOp):
    nfunc_spec = ("arctan", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arctan will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arctan(x, sig="f")
        return np.arctan(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (np.cast[x.type](1) + sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = atan(({cast}){x});"


arctan = ArcTan(upgrade_to_float, name="arctan")


class ArcTan2(BinaryScalarOp):
    nfunc_spec = ("arctan2", 2, 1)

    def impl(self, y, x):
        # If x and y are int8 or uint8, numpy.arctan2 will compute the result
        # in half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            y_dtype = str(getattr(x, "dtype", ""))
            if y_dtype in ("int8", "uint8"):
                return np.arctan2(y, x, sig="f")
        return np.arctan2(y, x)

    def L_op(self, inputs, outputs, gout):
        (y, x) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        else:
            if outputs[0].type in discrete_types:
                if x.type in discrete_types:
                    gx = x.zeros_like(dtype=theano.config.floatX)
                else:
                    gx = x.zeros_like()
                if y.type in discrete_types:
                    gy = y.zeros_like(dtype=theano.config.floatX)
                else:
                    gy = y.zeros_like()
                return [gx, gy]

            # If the output is float, the gradient should flow,
            # even if the inputs are ints
            return [gz * x / (sqr(x) + sqr(y)), gz * neg(y) / (sqr(x) + sqr(y))]

    def c_code(self, node, name, inputs, outputs, sub):
        (y, x) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types or node.inputs[1].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = atan2(({cast}){y}, ({cast}){x});"


arctan2 = ArcTan2(upgrade_to_float, name="arctan2")


class Cosh(UnaryScalarOp):
    """
    cosh(x) = (exp(x) + exp(-x)) / 2.

    """

    nfunc_spec = ("cosh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.cosh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.cosh(x, sig="f")
        return np.cosh(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * sinh(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = cosh(({cast}){x});"


cosh = Cosh(upgrade_to_float, name="cosh")


class ArcCosh(UnaryScalarOp):
    nfunc_spec = ("arccosh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arccosh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arccosh(x, sig="f")
        return np.arccosh(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqrt(sqr(x) - np.cast[x.type](1)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = acosh(({cast}){x});"


arccosh = ArcCosh(upgrade_to_float, name="arccosh")


class Sinh(UnaryScalarOp):
    """
    sinh(x) = (exp(x) - exp(-x)) / 2.

    """

    nfunc_spec = ("sinh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sinh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.sinh(x, sig="f")
        return np.sinh(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * cosh(x),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = sinh(({cast}){x});"


sinh = Sinh(upgrade_to_float, name="sinh")


class ArcSinh(UnaryScalarOp):
    nfunc_spec = ("arcsinh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arcsinh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arcsinh(x, sig="f")
        return np.arcsinh(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / sqrt(sqr(x) + np.cast[x.type](1)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = asinh(({cast}){x});"


arcsinh = ArcSinh(upgrade_to_float, name="arcsinh")


class Tanh(UnaryScalarOp):
    """
    tanh(x) = sinh(x) / cosh(x)
            = (exp(2*x) - 1) / (exp(2*x) + 1).

    """

    nfunc_spec = ("tanh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.tanh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.tanh(x, sig="f")
        return np.tanh(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * (1 - sqr(tanh(x))),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = tanh(({cast}){x});"


tanh = Tanh(upgrade_to_float, name="tanh")


class ArcTanh(UnaryScalarOp):
    nfunc_spec = ("arctanh", 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arctanh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, "dtype", ""))
        if x_dtype in ("int8", "uint8"):
            return np.arctanh(x, sig="f")
        return np.arctanh(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz / (np.cast[x.type](1) - sqr(x)),)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError("type not supported", type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return f"{z} = atanh(({cast}){x});"


arctanh = ArcTanh(upgrade_to_float, name="arctanh")


class Real(UnaryScalarOp):
    """
    Extract the real coordinate of a complex number.

    """

    # numpy.real(float32) return a view on the inputs.
    # nfunc_spec = ('real', 1, 1)

    def impl(self, x):
        return np.real(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return [complex(gz, 0)]


real = Real(real_out, name="real")


class Imag(UnaryScalarOp):
    nfunc_spec = ("imag", 1, 1)

    def impl(self, x):
        return np.imag(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            return [complex(0, gz)]
        elif x.type in float_types:
            return [second(x, 0)]
        else:
            return [x.zeros_like(dtype=theano.config.floatX)]


imag = Imag(real_out, name="imag")


class Angle(UnaryScalarOp):
    nfunc_spec = ("angle", 1, 1)

    def impl(self, x):
        return np.angle(x)

    def grad(self, inputs, gout):
        # y = x.imag
        # r = sqrt(y**2 + x.real**2)
        # g = y/r
        # if x == 0 and y == 0:
        #     theta = 0
        # elif x >= 0:
        #     theta = numpy.arcsin(g)
        # else:
        #     theta = -numpy.arcsin(g)+numpy.pi

        (c,) = inputs
        (gtheta,) = gout
        x = real(c)
        y = imag(c)
        r = abs(c)

        gr = -gtheta * y / (r ** 2 * sqrt(1 - (y / r) ** 2))
        gx = gr * x / r
        gy = gr * y / r
        if c in complex_types:
            return [cast(complex(gx, gy), x.type.dtype)]
        elif c in float_types:
            return [cast(second(x, 0), x.type.dtype)]
        else:
            return [c.zeros_like(dtype=theano.config.floatX)]


angle = Angle(specific_out(float64), name="angle")


class Complex(BinaryScalarOp):
    @staticmethod
    def output_types_preference(x, y):
        if x in complex_types:
            raise TypeError(x)
        if y in complex_types:
            raise TypeError(y)

        up = Scalar.upcast(x, y)
        if up in ("float64", "int64", "uint64", "int32", "uint32"):
            return [complex128]
        else:
            return [complex64]

    def impl(self, x, y):
        return complex(x, y) # PMB changed np.complex to complex

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        return [cast(real(gz), x.type.dtype), cast(imag(gz), y.type.dtype)]


complex = Complex(name="complex")


class Conj(UnaryScalarOp):
    nfunc_spec = ("conj", 1, 1)

    def impl(self, x):
        return np.conj(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            # For non complex, th
            raise NotImplementedError("type have no c code", node.inputs[0].type)
        return f"{z} = {x};"


conj = Conj(same_out_min8, name="conj")


class ComplexFromPolar(BinaryScalarOp):
    @staticmethod
    def output_types_preference(x, y):
        return Complex.output_types_preference(x, y)

    def impl(self, r, theta):
        if r < 0:
            raise ValueError("polar radius must be non-negative", r)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        if x.dtype == "float32":
            return np.complex64(complex(x, y))  # PMB changed np.complex to complex
        else:
            return np.complex128(complex(x, y))  # PMB changed np.complex to complex

    def grad(self, inputs, gout):
        (r, theta) = inputs
        (gz,) = gout
        gr = gz * complex_from_polar(1, theta)
        gtheta = gz * complex_from_polar(r, -theta)
        return [gr, gtheta]


complex_from_polar = ComplexFromPolar(name="complex_from_polar")


class Composite(ScalarOp):
    """
    Composite is an Op that takes a graph of scalar operations and
    produces c code for the whole graph. Its purpose is to implement loop
    fusion.

    Composite depends on all the Ops in its graph having C code.

    """

    init_param = ("inputs", "outputs")

    def __str__(self):
        if self.name is None:
            self.init_name()
        return self.name

    def make_new_inplace(self, output_types_preference=None, name=None):
        """
        This op.__init__ fct don't have the same parameter as other scalar op.
        This break the insert_inplace_optimizer optimization.
        This fct allow fix patch this.

        """
        d = {k: getattr(self, k) for k in self.init_param}
        out = self.__class__(**d)
        if name:
            out.name = name
        else:
            name = out.name
        super(Composite, out).__init__(output_types_preference, name)
        return out

    def init_c_code(self):
        """
        Assemble the C code for this Composite Op.

        The result is assigned to `self._c_code`.
        """
        # It was already called
        if hasattr(self, "_c_code"):
            return
        subd = dict(
            chain(
                ((e, f"%(i{int(i)})s") for i, e in enumerate(self.fgraph.inputs)),
                ((e, f"%(o{int(i)})s") for i, e in enumerate(self.fgraph.outputs)),
            )
        )

        for var in self.fgraph.variables:
            if var.owner is None:
                if var not in self.fgraph.inputs:
                    # This is an orphan
                    if isinstance(var, Constant):
                        subd[var] = var.type.c_literal(var.data)
                    else:
                        raise ValueError(
                            "All orphans in the fgraph to Composite must"
                            " be Constant instances."
                        )
            elif any(i.dtype == "float16" for i in var.owner.inputs) or any(
                o.dtype == "float16" for o in var.owner.outputs
            ):
                # flag for elemwise ops to check.
                self.inner_float16 = True

        _c_code = "{\n"
        self.nodenames = [
            f"%(nodename)s_subnode{int(j)}"
            for j, n in enumerate(self.fgraph.toposort())
        ]

        i = 0
        for j, node in enumerate(self.fgraph.toposort()):
            for output in node.outputs:
                if output not in subd:
                    i += 1
                    name = f"V%(id)s_tmp{int(i)}"
                    subd[output] = name
                    _c_code += f"{output.type.dtype_specs()[1]} {name};\n"
            s = node.op.c_code(
                node,
                self.nodenames[j],
                [subd[input] for input in node.inputs],
                [subd[output] for output in node.outputs],
                dict(fail="%(fail)s", id=f"%(id)s_{int(j)}"),
            )
            _c_code += s
            _c_code += "\n"
        _c_code += "}\n"
        self._c_code = _c_code

    def init_py_impls(self):
        """
        Return a list of functions that compute each output of self.

        """
        # In the case where the graph is a dag, but not a tree like:
        # add(*1 -> mul(x, y), *1)

        # We have an efficient way to build the executable (we build
        # and traverse each node only once).

        # But we don't have an efficient execution. We will execute
        # like a tree, so nodes that have more then 1 client will be
        # executed as many times as there number of clients. In the
        # example aboce, it will calculate *1 twice. Doing otherwise
        # imply making a complicated execution engine.

        # We need the fast creation of the executor as we always do it
        # even if we will use the c code. The Python implementation is
        # already slow, so it is not as much important to have a fast
        # execution there.

        memo = {}

        def compose_impl(r):
            if r in memo:
                return memo[r]
            if r in self.fgraph.inputs:
                idx = self.fgraph.inputs.index(r)

                def f(inputs):
                    return inputs[idx]

                memo[r] = f
                return f
            elif r.owner is None:  # in fgraph.orphans:

                def f(inputs):
                    return r.data

                memo[r] = f
                return f
            node = r.owner
            producers = [compose_impl(input) for input in node.inputs]

            def f(inputs):
                return node.op.impl(*[p(inputs) for p in producers])

            memo[r] = f
            return f

        self._impls = [compose_impl(r) for r in self.fgraph.outputs]

    def init_name(self):
        """
        Return a readable string representation of self.fgraph.

        """
        rval = self.name
        if rval is None:
            for i, r in enumerate(self.fgraph.inputs):
                r.name = f"i{int(i)}"
            for i, r in enumerate(self.fgraph.outputs):
                r.name = f"o{int(i)}"
            io = set(self.fgraph.inputs + self.fgraph.outputs)
            for i, r in enumerate(self.fgraph.variables):
                if r not in io and len(r.clients) > 1:
                    r.name = f"t{int(i)}"
            outputs_str = ", ".join([pprint(output) for output in self.fgraph.outputs])
            rval = f"Composite{{{outputs_str}}}"
            self.name = rval

    def init_fgraph(self):
        # The clone done by FunctionGraph is needed as we don't want
        # the fgraph to be set to the variable as we need to pickle
        # them for the cache of c module to work.
        fgraph = FunctionGraph(self.inputs, self.outputs)
        gof.MergeOptimizer().optimize(fgraph)
        for node in fgraph.apply_nodes:
            if not isinstance(node.op, ScalarOp):
                raise ValueError(
                    "The fgraph to Composite must be exclusively"
                    " composed of ScalarOp instances."
                )
        self.fgraph = fgraph

    def __init__(self, inputs, outputs):
        # We need to clone the graph as sometimes its nodes already
        # contain a reference to an fgraph. As we want the Composite
        # to be pickable, we can't have reference to fgraph.

        # Also, if there is Composite in the inner graph, we want to
        # remove them. In that case, we do a more complicated clone
        # that will flatten Composite. We don't need to do this
        # recusively, as the way the fusion optimizer work, we have
        # only 1 new Composite each time at the output.
        for i in inputs:
            assert i not in outputs  # This isn't supported, use identity
        if len(outputs) > 1 or not any(
            [isinstance(var.owner.op, Composite) for var in outputs]
        ):
            # No inner Composite
            inputs, outputs = gof.graph.clone(inputs, outputs)
        else:
            # Inner Composite that we need to flatten
            assert len(outputs) == 1
            # 1. Create a new graph from inputs up to the
            # Composite
            res = theano.compile.rebuild_collect_shared(
                inputs=inputs, outputs=outputs[0].owner.inputs, copy_inputs_over=False
            )  # Clone also the inputs
            # 2. We continue this partial clone with the graph in
            # the inner Composite
            res2 = theano.compile.rebuild_collect_shared(
                inputs=outputs[0].owner.op.inputs,
                outputs=outputs[0].owner.op.outputs,
                replace=dict(zip(outputs[0].owner.op.inputs, res[1])),
            )
            assert len(res2[1]) == len(outputs)
            assert len(res[0]) == len(inputs)
            assert res[0] != inputs
            inputs, outputs = res[0], res2[1]
            # Next assert comment just for speed
            # assert not any([isinstance(node.op, Composite) for node in
            #                theano.gof.graph.ops(inputs, outputs)])

        self.inputs = copy(inputs)
        self.outputs = copy(outputs)
        self.inputs_type = tuple([input.type for input in inputs])
        self.outputs_type = tuple([output.type for output in outputs])
        self.nin = len(inputs)
        self.nout = len(outputs)
        self.init_fgraph()  # self.fgraph
        # Postpone the creation in case it isn't needed.
        #  self.init_name()      # self.name
        self.name = None
        self.prepare_node_called = set()

    def prepare_node(self, node, storage_map, compute_map, impl):
        if impl == "py":
            self.init_py_impls()  # self._impls
        if impl not in self.prepare_node_called:
            for n in theano.gof.graph.list_of_nodes(self.inputs, self.outputs):
                n.op.prepare_node(n, None, None, impl)
            self.prepare_node_called.add(impl)

    def clone_float32(self):
        # This will not modify the fgraph or the nodes
        new_ins, new_outs = composite_f32.apply(self.fgraph)
        return Composite(new_ins, new_outs)

    def output_types(self, input_types):
        if tuple(input_types) != self.inputs_type:
            raise TypeError(
                f"Wrong types for Composite. Expected {self.inputs_type}, got {tuple(input_types)}."
            )
        return self.outputs_type

    def make_node(self, *inputs):
        if tuple([i.type for i in self.inputs]) == tuple([i.type for i in inputs]):
            return super().make_node(*inputs)
        else:
            # Make a new op with the right input type.
            assert len(inputs) == self.nin
            res = theano.compile.rebuild_collect_shared(
                self.outputs,
                replace=dict(zip(self.inputs, inputs)),
                rebuild_strict=False,
            )
            # After rebuild_collect_shared, the Variable in inputs
            # are not necessarily in the graph represented by res.
            # res[2][0] is a dict that map from the original variable to the
            # cloned variable.
            cloned_inputs = [res[2][0][i] for i in inputs]
            node = Composite(cloned_inputs, res[1]).make_node(*inputs)
            return node

    def perform(self, node, inputs, output_storage):
        for storage, impl in zip(output_storage, self._impls):
            storage[0] = impl(inputs)

    def impl(self, *inputs):
        output_storage = [[None] for i in range(self.nout)]
        self.perform(None, inputs, output_storage)
        ret = utils.to_return_values([storage[0] for storage in output_storage])
        if self.nout > 1:
            ret = tuple(ret)
        return ret

    def grad(self, inputs, output_grads):
        raise NotImplementedError("grad is not implemented for Composite")

    def c_code(self, node, nodename, inames, onames, sub):
        self.init_c_code()

        d = dict(
            chain(
                zip((f"i{int(i)}" for i in range(len(inames))), inames),
                zip((f"o{int(i)}" for i in range(len(onames))), onames),
            ),
            **sub,
        )
        d["nodename"] = nodename
        if "id" not in sub:
            # The use of a dummy id is safe as the code is in a separate block.
            # It won't generate conflicting variable name.
            d["id"] = "_DUMMY_ID_"

        return self._c_code % d

    def c_code_cache_version(self):
        rval = [3]
        for x in self.fgraph.toposort():
            xv = x.op.c_code_cache_version()
            if xv:
                rval.append(xv)
            else:
                return ()
        return tuple(rval)

    def c_support_code(self):
        rval = []
        for subnode in self.fgraph.toposort():
            try:
                rval.append(subnode.op.c_support_code().strip())
            except gof.utils.MethodNotDefined:
                pass
        # remove duplicate code blocks
        return "\n".join(sorted(set(rval)))

    def c_support_code_apply(self, node, name):
        self.init_c_code()
        rval = []
        for subnode, subnodename in zip(self.fgraph.toposort(), self.nodenames):
            try:
                subnode_support_code = subnode.op.c_support_code_apply(
                    subnode, subnodename % dict(nodename=name)
                )
                if subnode_support_code:
                    rval.append(subnode_support_code)
            except gof.utils.MethodNotDefined:
                pass
        # there should be no need to remove duplicate code blocks because
        # each block should have been specialized for the given nodename.
        # Any block that isn't specialized should be returned via
        # c_support_code instead of c_support_code_apply.
        return "\n".join(rval)

    def __eq__(self, other):
        if self is other:
            return True
        if (
            type(self) != type(other)
            or self.nin != other.nin
            or self.nout != other.nout
        ):
            return False
        # see __hash__ for comment on why there is no mention of fgraph
        # or module cache key here.
        self.init_c_code()  # self._c_code and self.nodenames
        other.init_c_code()
        return self._c_code == other._c_code

    def __hash__(self):
        self.init_c_code()  # self._c_code and self.nodenames
        rval = hash((type(self), self.nin, self.nout, self._c_code))
        # Note that in general, the configparser settings at the time
        # of code generation (__init__) affect the semantics of this Op.
        # This function assumes that all relevant info about the configparser
        # is embodied in _c_code.  So the _c_code, rather than self.fgraph,
        # is the signature of the semantics of this Op.
        # _c_code is preserved through unpickling, so the Op will not change
        # semantics when it is reloaded with different configparser
        # settings.
        return rval

    def __getstate__(self):
        rval = dict(self.__dict__)
        rval.pop("_impls", None)
        rval.pop("prepare_node_called", None)
        del rval["fgraph"]
        return rval

    def __setstate__(self, d):
        self.__dict__.update(d)
        # We must call init to set fgraph and _impls again, as otherwise
        # self.perform will not work.
        self.prepare_node_called = set()
        self.init_fgraph()
        self.init_py_impls()


class Compositef32:
    # This is a dict of scalar op classes that need special handling
    special = {}

    def apply(self, fgraph):
        mapping = {}
        topo = fgraph.toposort()
        for i in fgraph.inputs:
            if i.dtype == "float16":
                mapping[i] = get_scalar_type("float32")()
                if hasattr(i.tag, "test_value"):
                    mapping[i].tag.test_value = i.tag.test_value
            else:
                mapping[i] = i
        for node in topo:
            # Patch up for constants
            for i in node.inputs:
                if i not in mapping:
                    assert type(i) is ScalarConstant
                    if i.type == float16:
                        ni = ScalarConstant(float32, i.data)
                    else:
                        ni = i
                    mapping[i] = ni
            if type(node.op) in self.special:
                self.special[type(node.op)](node, mapping)
                continue
            new_node = node.clone_with_new_inputs(
                [mapping[inp] for inp in node.inputs], strict=False
            )
            # make sure we don't produce any float16.
            assert not any(o.dtype == "float16" for o in new_node.outputs)
            for o, no in zip(node.outputs, new_node.outputs):
                mapping[o] = no

        new_ins = [mapping[inp] for inp in fgraph.inputs]
        new_outs = [mapping[out] for out in fgraph.outputs]
        return new_ins, new_outs


composite_f32 = Compositef32()


def handle_cast(node, mapping):
    inp = mapping[node.inputs[0]]
    out = node.outputs[0]
    node_ok = False
    if node.op.o_type == float16:
        if node.inputs[0].type == float32:
            # cast f32 -> f16, remove
            mapping[out] = inp
            return
        else:
            # cast to f16, convert to f32
            new_out = cast(inp, "float32")
            # change the node for the following if
            node = new_out.owner
            mapping[out] = new_out
            node_ok = True
    if node.inputs[0].type == float16:
        if node.op.o_type == inp.type:
            # cast f16 to new input type, remove
            mapping[out] = inp
            return
    if not node_ok:
        new_node = node.clone_with_new_inputs([inp], strict=False)
        mapping[out] = new_node.outputs[0]


Compositef32.special[Cast] = handle_cast


def handle_composite(node, mapping):
    new_op = node.op.clone_float32()
    new_outs = new_op(*[mapping[i] for i in node.inputs], return_list=True)
    assert len(new_outs) == len(node.outputs)
    for o, no in zip(node.outputs, new_outs):
        mapping[o] = no


Compositef32.special[Composite] = handle_composite
