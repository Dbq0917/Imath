//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "testLowPrecision.h"

#include <ImathFP8.h>
#include <ImathMatrix.h>
#include <ImathVec.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

using namespace IMATH_NAMESPACE;

namespace
{

double
roundTiesToEven (double v)
{
    const double lower = std::floor (v);
    const double frac  = v - lower;

    if (frac < 0.5) return lower;
    if (frac > 0.5) return lower + 1.0;
    return std::fmod (lower, 2.0) == 0.0 ? lower : lower + 1.0;
}

template <class T>
bool
sameValue (float expected, T actual)
{
    const float value = float (actual);
    if (std::isnan (expected)) return actual.isNan ();
    if (std::isinf (expected)) return actual.isInfinity ();
    if (expected == 0.0f)
        return value == 0.0f &&
               std::signbit (value) == std::signbit (expected);
    return value == expected;
}

template <FP8Format Format>
float
referenceQuantize (float input)
{
    typedef fp8_traits<Format> traits;

    const bool negative = std::signbit (input);
    const float magnitude = std::fabs (input);

    if (std::isnan (input)) return std::numeric_limits<float>::quiet_NaN ();
    if (std::isinf (input))
        return traits::has_infinity
                   ? (negative ? -std::numeric_limits<float>::infinity ()
                               : std::numeric_limits<float>::infinity ())
                   : (negative ? -traits::max_value () : traits::max_value ());

    if (magnitude == 0.0f) return negative ? -0.0f : 0.0f;

    if (traits::has_infinity)
    {
        const double overflow_threshold = std::ldexp (
            2.0 - std::ldexp (1.0, -(traits::mantissa_bits + 1)),
            traits::max_normal_exponent);
        if (magnitude >= overflow_threshold)
            return negative ? -std::numeric_limits<float>::infinity ()
                            : std::numeric_limits<float>::infinity ();
    }
    else if (magnitude > traits::max_value ())
    {
        return negative ? -traits::max_value () : traits::max_value ();
    }

    double quantum;
    if (magnitude < traits::min_value ())
        quantum = traits::denorm_min ();
    else
    {
        int exponent;
        std::frexp (magnitude, &exponent);
        quantum = std::ldexp (1.0, exponent - 1 - traits::mantissa_bits);
    }

    double quantized =
        roundTiesToEven (static_cast<double> (magnitude) / quantum) * quantum;

    if (quantized > traits::max_value ())
        quantized = traits::max_value ();

    const float result = static_cast<float> (quantized);
    return negative ? -result : result;
}

template <FP8Format Format>
void
testReferenceConversions ()
{
    typedef lowp8<Format> fp8_type;

    for (std::uint32_t bits = 0; bits < (1u << 16); ++bits)
    {
        const half h (half::FromBits, static_cast<std::uint16_t> (bits));
        const float input  = float (h);
        const float expect = referenceQuantize<Format> (input);
        const fp8_type actual (input);
        if (!sameValue (expect, actual))
            std::cerr << "half bits 0x" << std::hex << bits
                      << " input " << std::dec << input << " expected "
                      << expect << " actual " << float (actual) << " bits 0x"
                      << std::hex << unsigned (actual.bits ()) << std::dec
                      << "\n";
        assert (sameValue (expect, actual));

        const fp8_type from_half (h);
        assert (from_half.bits () == actual.bits ());
    }

    std::uint32_t state = 0x12345678u;
    for (int i = 0; i < 100000; ++i)
    {
        state = state * 1664525u + 1013904223u;
        float input;
        std::memcpy (&input, &state, sizeof (input));
        const float expect = referenceQuantize<Format> (input);
        const fp8_type actual (input);
        assert (sameValue (expect, actual));
    }
}

template <FP8Format Format>
void
testExhaustiveRoundTrip ()
{
    typedef lowp8<Format> fp8_type;

    for (unsigned int i = 0; i < 256; ++i)
    {
        const fp8_type value (
            fp8_type::FromBits, static_cast<std::uint8_t> (i));
        const half h       = value.toHalf ();
        const fp8_type back (h);

        if (value.isNan ())
            assert (back.isNan ());
        else
            assert (back.bits () == value.bits ());
    }
}

void
testKnownBits ()
{
    const fp8_e4m3 e4_one (1.0f);
    assert (e4_one.bits () == 0x38);
    assert (fp8_e4m3 (1.0625f).bits () == 0x38);
    assert (fp8_e4m3 (1.1875f).bits () == 0x3a);
    assert (fp8_e4m3 (448.0f).bits () == 0x7e);
    assert (fp8_e4m3 (1000.0f).bits () == 0x7e);
    assert (fp8_e4m3 (-1000.0f).bits () == 0xfe);
    assert (fp8_e4m3 (0.001953125f).bits () == 0x01);
    assert (fp8_e4m3 (0.0009765625f).bits () == 0x00);
    assert (fp8_e4m3 (0.0029296875f).bits () == 0x02);
    assert (std::signbit (float (fp8_e4m3 (-0.0f))));

    const fp8_e5m2 e5_one (1.0f);
    assert (e5_one.bits () == 0x3c);
    assert (fp8_e5m2 (1.125f).bits () == 0x3c);
    assert (fp8_e5m2 (1.375f).bits () == 0x3e);
    assert (fp8_e5m2 (57344.0f).bits () == 0x7b);
    assert (fp8_e5m2 (60000.0f).bits () == 0x7b);
    assert (fp8_e5m2 (62000.0f).bits () == 0x7c);
    assert (fp8_e5m2 (-60000.0f).bits () == 0xfb);
    assert (fp8_e5m2 (0.0000152587890625f).bits () == 0x01);

    assert (std::numeric_limits<fp8_e4m3>::max ().bits () == 0x7e);
    assert (std::numeric_limits<fp8_e5m2>::max ().bits () == 0x7b);
    assert (std::numeric_limits<fp8_e4m3>::min ().bits () == 0x08);
    assert (std::numeric_limits<fp8_e5m2>::min ().bits () == 0x04);
    assert (std::numeric_limits<fp8_e4m3>::denorm_min ().bits () == 0x01);
    assert (std::numeric_limits<fp8_e4m3>::epsilon ().bits () == 0x20);
    assert (std::numeric_limits<fp8_e5m2>::epsilon ().bits () == 0x34);
    assert (std::numeric_limits<fp8_e4m3>::digits == 4);
    assert (std::numeric_limits<fp8_e5m2>::digits == 3);
    assert (std::numeric_limits<fp8_e4m3>::min_exponent == -5);
    assert (std::numeric_limits<fp8_e4m3>::max_exponent == 9);
    assert (std::numeric_limits<fp8_e5m2>::min_exponent == -13);
    assert (std::numeric_limits<fp8_e5m2>::max_exponent == 16);
    assert (!std::numeric_limits<fp8_e4m3>::has_infinity);
    assert (std::numeric_limits<fp8_e5m2>::has_infinity);

    const half via_conversion (fp8_e4m3 (1.0f));
    assert (via_conversion.bits () == 0x3c00);
}

template <FP8Format Format>
void
testFiniteBatch ()
{
    typedef lowp8<Format> fp8_type;
    const std::size_t count = 1031;

    float* source = new float[count];
    fp8_type* packed = new fp8_type[count];
    fp8_type* packed_b = new fp8_type[count];
    fp8_type* packed_c = new fp8_type[count];
    float* decoded = new float[count];

    for (std::size_t i = 0; i < count; ++i)
    {
        const float a = std::sin (float (i) * 0.17f) * 300.0f;
        const float b = std::cos (float (i) * 0.11f) * 250.0f;
        source[i]  = a;
        packed[i]  = fp8_type (a);
        packed_b[i] = fp8_type (b);
    }

    fp8_batch_convert (source, packed_c, count);
    for (std::size_t i = 0; i < count; ++i)
        assert (packed_c[i].bits () == packed[i].bits ());

    fp8_batch_add (packed, packed_b, packed_c, count);
    for (std::size_t i = 0; i < count; ++i)
        assert (
            packed_c[i].bits () ==
            fp8_type (float (packed[i]) + float (packed_b[i])).bits ());

    fp8_batch_sub (packed, packed_b, packed_c, count);
    for (std::size_t i = 0; i < count; ++i)
        assert (
            packed_c[i].bits () ==
            fp8_type (float (packed[i]) - float (packed_b[i])).bits ());

    fp8_batch_mul (packed, packed_b, packed_c, count);
    for (std::size_t i = 0; i < count; ++i)
        assert (
            packed_c[i].bits () ==
            fp8_type (float (packed[i]) * float (packed_b[i])).bits ());

    fp8_batch_convert (packed_c, decoded, count);
    for (std::size_t i = 0; i < count; ++i)
        assert (decoded[i] == float (packed_c[i]));

    delete[] source;
    delete[] packed;
    delete[] packed_b;
    delete[] packed_c;
    delete[] decoded;
}

void
testSpecialValues ()
{
    const float inf = std::numeric_limits<float>::infinity ();
    const float qnan = std::numeric_limits<float>::quiet_NaN ();

    const fp8_e4m3 e4_inf (inf);
    assert (!e4_inf.isNan ());
    assert (!e4_inf.isInfinity ());
    assert (e4_inf.bits () == fp8_e4m3::posInf ().bits ());
    assert (e4_inf.bits () == 0x7e);
    assert (fp8_e4m3 (-inf).bits () == 0xfe);

    const fp8_e5m2 e5_inf (inf);
    const fp8_e5m2 e5_ninf (-inf);
    assert (e5_inf.isInfinity () && !e5_inf.isNegative ());
    assert (e5_ninf.isInfinity () && e5_ninf.isNegative ());
    assert (e5_inf.bits () == 0x7c);
    assert (e5_ninf.bits () == 0xfc);

    assert (fp8_e4m3 (qnan).isNan ());
    assert (fp8_e5m2 (qnan).isNan ());
    assert (!fp8_e4m3 (qnan).isNormalized ());
    assert (!fp8_e5m2 (qnan).isNormalized ());
    assert (!fp8_e5m2 (inf).isNormalized ());
    assert (!fp8_e4m3 (qnan).isFinite ());
    assert (!fp8_e5m2 (inf).isFinite ());
    assert (fp8_e4m3 (qnan).bits () == 0x7f);
    assert (fp8_e5m2 (qnan).bits () == 0x7f);

    fp8_e5m2 nan_value (qnan);
    nan_value += 1.0f;
    assert (nan_value.isNan ());

    fp8_e5m2 zero = fp8_e5m2 (inf);
    zero -= fp8_e5m2 (inf);
    assert (zero.isNan ());

    fp8_e4m3 e4_zero = fp8_e4m3 (inf);
    e4_zero -= fp8_e4m3 (inf);
    assert (e4_zero.isZero ());
}

void
testVectorPath ()
{
    const Vec3<fp8> v (3.0f, 4.0f, 0.0f);
    const fp8 length = v.length ();
    assert (float (length) == 5.0f);

    const Vec3<fp8> n = v.normalized ();
    assert (std::fabs (float (n.x) - 0.6f) <= 0.0625f);
    assert (std::fabs (float (n.y) - 0.8f) <= 0.0625f);
    assert (float (n.z) == 0.0f);
}

void
octEncode (const Vec3<float>& n, Vec2<half>& packed)
{
    const float l1 =
        std::fabs (n.x) + std::fabs (n.y) + std::fabs (n.z);
    Vec3<float> p = n / l1;

    if (p.z < 0.0f)
    {
        const float oldX = p.x;
        p.x = (1.0f - std::fabs (p.y)) * (oldX >= 0.0f ? 1.0f : -1.0f);
        p.y = (1.0f - std::fabs (oldX)) * (p.y >= 0.0f ? 1.0f : -1.0f);
    }

    packed.x = half ((p.x + 1.0f) * 0.5f);
    packed.y = half ((p.y + 1.0f) * 0.5f);
}

Vec3<float>
octDecode (const Vec2<half>& packed)
{
    const float x = 2.0f * float (packed.x) - 1.0f;
    const float y = 2.0f * float (packed.y) - 1.0f;
    const float z = 1.0f - std::fabs (x) - std::fabs (y);

    if (z < 0.0f)
    {
        const float oldX = x;
        const float nx =
            (1.0f - std::fabs (y)) * (oldX >= 0.0f ? 1.0f : -1.0f);
        const float ny =
            (1.0f - std::fabs (oldX)) * (y >= 0.0f ? 1.0f : -1.0f);
        return Vec3<float> (nx, ny, z).normalized ();
    }

    return Vec3<float> (x, y, z).normalized ();
}

Vec4<half>
skinVertex (
    const Vec3<half>& position,
    const half* weights,
    const Matrix44<float>* bones,
    int boneCount)
{
    Vec4<float> result (0.0f);
    for (int bone = 0; bone < boneCount; ++bone)
    {
        const Vec3<float> p (position);
        const Vec4<float> transformed =
            Vec4<float> (p.x, p.y, p.z, 1.0f) * bones[bone];
        result += transformed * float (weights[bone]);
    }
    return Vec4<half> (half (result.x), half (result.y), half (result.z),
                       half (result.w));
}

} // namespace

void
testLowPrecisionConversion ()
{
    std::cout << "FP8 conversion and rounding\n";

    testKnownBits ();
    testReferenceConversions<FP8Format::E4M3> ();
    testReferenceConversions<FP8Format::E5M2> ();
    testExhaustiveRoundTrip<FP8Format::E4M3> ();
    testExhaustiveRoundTrip<FP8Format::E5M2> ();
    testSpecialValues ();
    testFiniteBatch<FP8Format::E4M3> ();
    testFiniteBatch<FP8Format::E5M2> ();

    std::cout << "ok\n\n";
}

void
testLowPrecisionArithmetic ()
{
    std::cout << "FP8 scalar arithmetic\n";

    fp8_e4m3 a (1.0f);
    const fp8_e4m3 b (0.09375f);
    a += b;
    assert (a.bits () == 0x39);

    a *= fp8_e4m3 (2.0f);
    assert (a.bits () == 0x41);

    a -= fp8_e4m3 (2.25f);
    assert (a.isZero ());

    assert (fp8_e4m3 (2.0f) + fp8_e4m3 (3.0f) == 5.0f);
    assert (fp8_e4m3 (2.0f) * fp8_e4m3 (3.0f) == 6.0f);
    assert (fp8_e4m3 (7.0f) - fp8_e4m3 (3.0f) == 4.0f);

    testVectorPath ();

    std::cout << "ok\n\n";
}

void
testLowPrecisionRegression ()
{
    std::cout << "existing half-based algorithm regression\n";

    const Vec3<float> normals[] = {
        Vec3<float> (0.0f, 0.0f, 1.0f),
        Vec3<float> (1.0f, 0.0f, 0.0f),
        Vec3<float> (-1.0f, 0.0f, 0.0f),
        Vec3<float> (0.0f, 1.0f, 0.0f)};
    const std::uint16_t expectedX[] = {0x3800, 0x3c00, 0x0000, 0x3800};
    const std::uint16_t expectedY[] = {0x3800, 0x3800, 0x3800, 0x3c00};

    for (int i = 0; i < 4; ++i)
    {
        Vec2<half> packed;
        octEncode (normals[i], packed);
        assert (packed.x.bits () == expectedX[i]);
        assert (packed.y.bits () == expectedY[i]);
        assert (octDecode (packed).equalWithAbsError (normals[i], 1e-6f));
    }

    Matrix44<float> bones[2];
    bones[0] = Matrix44<float> (1.0f);
    bones[1] = Matrix44<float> (1.0f);
    bones[0][3][0] = 1.0f;
    bones[0][3][1] = 2.0f;
    bones[0][3][2] = 3.0f;
    bones[1][3][0] = 4.0f;
    bones[1][3][1] = 5.0f;
    bones[1][3][2] = 6.0f;

    const Vec3<half> position (0.0f, 0.0f, 0.0f);
    const half weights[2] = {half (0.25f), half (0.75f)};
    const Vec4<half> result =
        skinVertex (position, weights, bones, 2);
    assert (result.x.bits () == 0x4280);
    assert (result.y.bits () == 0x4440);
    assert (result.z.bits () == 0x4540);
    assert (result.w.bits () == 0x3c00);

    std::cout << "ok\n\n";
}
