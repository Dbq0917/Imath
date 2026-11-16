..
  SPDX-License-Identifier: BSD-3-Clause
  Copyright Contributors to the OpenEXR Project.

.. _low-precision:

Low-precision floating point
############################

.. code-block::

   #include <Imath/ImathFP8.h>

Imath provides two 8-bit floating-point formats for quantized
intermediate values and packed weights:

* ``fp8_e4m3`` (also ``fp8``, ``quarter``, and ``fp8_e4m3fn``) is an
  OCP E4M3FN format with four exponent bits and three mantissa bits.
  It has the higher precision of the two formats and no infinities.
* ``fp8_e5m2`` (also ``quarter_e5m2``) is an OCP E5M2 format with five
  exponent bits and two mantissa bits. It has the wider dynamic range
  and IEEE-style infinities.

Both types have a one-byte, trivially-copyable representation and
support construction from ``float``, ``half``, and raw bits, conversion
back to ``float`` and ``half``, unary negation, compound addition,
subtraction, multiplication, and division, and the usual
``isFinite()``/``isNan()``/``isInfinity()`` classification methods.
Scalar arithmetic follows the same promotion model as ``half``:
expressions are evaluated as ``float`` and rounded once when assigned
to an FP8 object. For example:

.. code-block:: c++

   Imath::fp8_e4m3 a (1.0f);
   Imath::fp8_e4m3 b (0.1f);
   a += b;                 // one nearest-even rounding
   float result = a * 2;   // promoted to float; assign back to round

Precision and range
-------------------

The following table is the authoritative quick reference for the
formats. Values are exact at the stated endpoints.

.. list-table::
   :header-rows: 1

   * - Format
     - Sign bits
     - Exp bits
     - Mant bits
     - Min denorm
     - Min normal
     - Max finite
     - Epsilon
   * - ``fp8_e4m3``
     - 1
     - 4
     - 3
     - 0.001953125
     - 0.015625
     - 448
     - 0.125
   * - ``fp8_e5m2``
     - 1
     - 5
     - 2
     - 1.52587890625e-5
     - 6.103515625e-5
     - 57344
     - 0.25

Additional properties:

.. list-table::
   :header-rows: 1

   * - Format
     - Bias
     - NaN bits
     - Infinity
     - Overflow target
   * - ``fp8_e4m3``
     - 7
     - 0x7f
     - none
     - saturates to +/-448
   * - ``fp8_e5m2``
     - 15
     - 0x7f
     - +/-0x7c
     - infinity

``std::numeric_limits<fp8_e4m3>`` and
``std::numeric_limits<fp8_e5m2>`` are specialized with the corresponding
range, precision, and special-value properties.

Rounding
--------

Conversion from ``half``, ``float``, or ``double`` uses
**round-to-nearest, ties-to-even**. The tie rule is applied to the
retained significand: if the discarded bits are exactly half an ulp,
the retained significand is made even. This is the same rounding rule
used by Imath ``half``.

* Underflow rounds to the nearest subnormal. Values exactly halfway
  between zero and the smallest subnormal round to zero, and the sign
  of zero is preserved.
* Finite values at or below the maximum round normally.
* E4M3 values above 448, including input infinities, saturate to the
  finite maximum with the input sign.
* E5M2 values at or above the overflow threshold 61440 round to
  infinity. Values between 57344 and that threshold round to 57344.
* Every FP8 value is exactly representable as ``half``. Converting an
  FP8 value to ``half`` and back therefore preserves its bit pattern,
  except that NaN payloads are canonicalized as described below.
  Conversion from arbitrary ``half`` values to FP8 can round because FP8
  has fewer significand bits.

``half`` conversion
-------------------

The header provides scalar constructors, ``toHalf()``, and batch
conversion overloads. The following round trip is exact for every
finite FP8 value and for infinities:

.. code-block:: c++

   Imath::fp8_e4m3 q (some_half);
   Imath::half      h = q.toHalf ();

This property is checked exhaustively over all 256 codewords in the
test suite.

NaN and infinity
----------------

The handling of exceptional values is format-independent where
possible and explicitly defined where the formats differ:

* A NaN input converts to the canonical positive quiet NaN ``0x7f``.
  NaN sign and payload are discarded. Arithmetic involving a NaN
  produces the canonical NaN. There is no distinct signaling NaN in
  E4M3; ``fp8_e5m2`` reserves ``0x7e`` as its signaling NaN.
* ``fp8_e5m2`` preserves positive and negative infinity through
  conversion and arithmetic. ``inf - inf`` and ``0 * inf`` produce NaN.
* ``fp8_e4m3`` has no infinity encoding. An input infinity saturates to
  the maximum finite value with the source sign. Consequently,
  ``+inf - +inf`` applied to stored E4M3 values is ``448 - 448 == 0``,
  not NaN. Code requiring infinity semantics must use ``fp8_e5m2``.
* Division by zero follows IEEE ``float`` promotion and is then rounded
  according to the destination format.

Batch and vectorized use
------------------------

The following functions operate on contiguous packed arrays and round
the result of each element once:

.. code-block:: c++

   Imath::fp8_batch_convert (float_src, fp8_dst, count);
   Imath::fp8_batch_convert (fp8_src, float_dst, count);
   Imath::fp8_batch_convert (half_src, fp8_dst, count);
   Imath::fp8_batch_convert (fp8_src, half_dst, count);
   Imath::fp8_batch_add (a, b, dst, count);
   Imath::fp8_batch_sub (a, b, dst, count);
   Imath::fp8_batch_mul (a, b, dst, count);

The arithmetic entry points process 32 elements at a time through
contiguous, branch-free float staging blocks. This keeps conversion
bit-twiddling out of the arithmetic loop and allows the compiler's
normal auto-vectorization path to issue packed float adds,
subtractions, and multiplies. The remaining tail uses the same scalar
semantics, so no element takes a different code path for rounding or
exceptions.

FP8 is registered as float-like for ``Vec2``, ``Vec3``, and ``Vec4``,
so normalization and length operations are available. Convenience
aliases include ``V2q``/``V3q``/``V4q`` for the default E4M3 format and
``V2e4m3``/``V3e4m3``/``V4e4m3`` plus
``V2e5m2``/``V3e5m2``/``V4e5m2``.

Example:

.. code-block:: c++

   Imath::V3q packed_normal (0.6f, 0.8f, 0.0f);
   Imath::V3q unit_normal = packed_normal.normalized ();

Compatibility
-------------

The existing ``half``, ``float``, and ``double`` types and operations
are unchanged. FP8 support is additive and header-only; existing code
does not need to opt in. The regression suite includes representative
half-based normal compression and skinning paths in addition to the
existing Imath test suite.
