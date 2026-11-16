//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMATHFP8_H
#define INCLUDED_IMATHFP8_H

#include "ImathExport.h"
#include "ImathNamespace.h"
#include "ImathPlatform.h"
#include "ImathTypeTraits.h"
#include "half.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

IMATH_STABLE_3_2_ENTER

/// The two 8-bit floating-point interchange formats supported by Imath.
///
/// `E4M3` follows the OCP FP8 E4M3FN layout: 4 exponent bits, 3
/// mantissa bits, no infinities, and one NaN encoding. `E5M2` follows
/// the OCP FP8 E5M2 layout: 5 exponent bits, 2 mantissa bits, and
/// IEEE-style infinities and NaNs.
enum class FP8Format
{
    E4M3,
    E5M2
};

/// Properties of an FP8 interchange format.
template <FP8Format Format> struct fp8_traits;

template <> struct fp8_traits<FP8Format::E4M3>
{
    static constexpr int exponent_bits       = 4;
    static constexpr int mantissa_bits       = 3;
    static constexpr int exponent_bias       = 7;
    static constexpr int exponent_field_max  = 15;
    static constexpr int max_normal_exponent = 8;
    static constexpr int max_digits10        = 3;
    static constexpr int digits              = mantissa_bits + 1;
    static constexpr int min_exponent        = -5;
    static constexpr int max_exponent_10     = 2;
    static constexpr int min_exponent_10     = -1;
    static constexpr bool has_infinity       = false;
    static constexpr bool has_signaling_nan = false;

    static constexpr std::uint8_t sign_mask       = 0x80;
    static constexpr std::uint8_t exponent_mask   = 0x78;
    static constexpr std::uint8_t mantissa_mask   = 0x07;
    static constexpr std::uint8_t positive_max    = 0x7e;
    static constexpr std::uint8_t positive_inf    = 0x7e; // saturates
    static constexpr std::uint8_t positive_nan    = 0x7f;
    static constexpr std::uint8_t positive_snan   = 0x7f;
    static constexpr std::uint8_t positive_min    = 0x08;
    static constexpr std::uint8_t positive_denorm = 0x01;
    static constexpr std::uint8_t positive_eps    = 0x20;
    static constexpr std::uint8_t positive_half   = 0x30;

    static float max_value () IMATH_NOEXCEPT { return 448.0f; }
    static float min_value () IMATH_NOEXCEPT { return 0.015625f; }
    static float denorm_min () IMATH_NOEXCEPT { return 0.001953125f; }
    static float epsilon () IMATH_NOEXCEPT { return 0.125f; }
};

template <> struct fp8_traits<FP8Format::E5M2>
{
    static constexpr int exponent_bits       = 5;
    static constexpr int mantissa_bits       = 2;
    static constexpr int exponent_bias       = 15;
    static constexpr int exponent_field_max  = 31;
    static constexpr int max_normal_exponent = 15;
    static constexpr int max_digits10        = 2;
    static constexpr int digits              = mantissa_bits + 1;
    static constexpr int min_exponent        = -13;
    static constexpr int max_exponent_10     = 4;
    static constexpr int min_exponent_10     = -4;
    static constexpr bool has_infinity       = true;
    static constexpr bool has_signaling_nan = true;

    static constexpr std::uint8_t sign_mask       = 0x80;
    static constexpr std::uint8_t exponent_mask   = 0x7c;
    static constexpr std::uint8_t mantissa_mask   = 0x03;
    static constexpr std::uint8_t positive_max    = 0x7b;
    static constexpr std::uint8_t positive_inf    = 0x7c;
    static constexpr std::uint8_t positive_nan    = 0x7f;
    static constexpr std::uint8_t positive_snan   = 0x7e;
    static constexpr std::uint8_t positive_min    = 0x04;
    static constexpr std::uint8_t positive_denorm = 0x01;
    static constexpr std::uint8_t positive_eps    = 0x34;
    static constexpr std::uint8_t positive_half   = 0x38;

    static float max_value () IMATH_NOEXCEPT { return 57344.0f; }
    static float min_value () IMATH_NOEXCEPT { return 0.00006103515625f; }
    static float denorm_min () IMATH_NOEXCEPT
    {
        return 0.0000152587890625f;
    }
    static float epsilon () IMATH_NOEXCEPT { return 0.25f; }
};

namespace detail
{

#if defined(__GNUC__) && !defined(__clang__)
#    define IMATH_FP8_IVDEP _Pragma ("GCC ivdep")
#elif defined(__clang__)
#    define IMATH_FP8_IVDEP _Pragma ("clang loop vectorize(enable)")
#else
#    define IMATH_FP8_IVDEP
#endif

inline std::uint32_t
fp8_round_shift (std::uint32_t value, unsigned int shift) IMATH_NOEXCEPT
{
    if (shift == 0) return value;
    if (shift >= 32) return 0;

    const std::uint32_t truncated = value >> shift;
    const std::uint32_t mask      = (std::uint32_t (1) << shift) - 1;
    const std::uint32_t remainder = value & mask;
    const std::uint32_t halfway  = std::uint32_t (1) << (shift - 1);

    if (remainder > halfway ||
        (remainder == halfway && (truncated & std::uint32_t (1)) != 0))
        return truncated + 1;

    return truncated;
}

inline std::uint32_t
fp8_float_bits (float f) IMATH_NOEXCEPT
{
    std::uint32_t bits;
    std::memcpy (&bits, &f, sizeof (bits));
    return bits;
}

inline float
fp8_bits_float (std::uint32_t bits) IMATH_NOEXCEPT
{
    float f;
    std::memcpy (&f, &bits, sizeof (f));
    return f;
}

template <FP8Format Format>
inline std::uint8_t
fp8_from_float_bits (float f) IMATH_NOEXCEPT
{
    typedef fp8_traits<Format> traits;

    const std::uint32_t u   = fp8_float_bits (f);
    const std::uint8_t  sign =
        static_cast<std::uint8_t> ((u >> 24) & traits::sign_mask);
    const std::uint32_t abs = u & 0x7fffffffu;

    if (abs > 0x7f800000u)
        return traits::positive_nan;

    if (abs == 0x7f800000u)
    {
        if (traits::has_infinity)
            return static_cast<std::uint8_t> (sign | traits::positive_inf);
        return static_cast<std::uint8_t> (sign | traits::positive_max);
    }

    if (abs == 0) return sign;

    // E4M3 cannot represent infinity, so values above its finite
    // maximum saturate. E5M2 uses the normal nearest-even path below;
    // rounding a value at or above the overflow threshold produces an
    // infinity there.
    const std::uint32_t max_bits =
        fp8_float_bits (traits::max_value ()) & 0x7fffffffu;
    if (!traits::has_infinity && abs > max_bits)
    {
        return static_cast<std::uint8_t> (sign | traits::positive_max);
    }

    const std::uint32_t float_exponent = (abs >> 23) & 0xffu;
    const std::uint32_t float_mantissa = abs & 0x7fffffu;
    const std::uint32_t min_normal_bits =
        fp8_float_bits (traits::min_value ()) & 0x7fffffffu;

    if (abs >= min_normal_bits)
    {
        std::uint32_t significand = float_mantissa | 0x800000u;
        std::uint32_t exponent    = float_exponent;
        const unsigned int shift =
            static_cast<unsigned int> (23 - traits::mantissa_bits);

        significand = fp8_round_shift (significand, shift);
        if (significand == (std::uint32_t (1)
                            << (traits::mantissa_bits + 1)))
        {
            significand >>= 1;
            ++exponent;
        }

        const int encoded_exponent =
            static_cast<int> (exponent) - 127 + traits::exponent_bias;
        if (traits::has_infinity &&
            encoded_exponent >= traits::exponent_field_max)
        {
            return static_cast<std::uint8_t> (
                sign | traits::positive_inf);
        }

        const std::uint8_t mantissa = static_cast<std::uint8_t> (
            significand &
            ((std::uint32_t (1) << traits::mantissa_bits) - 1));
        return static_cast<std::uint8_t> (
            sign |
            (static_cast<std::uint8_t> (encoded_exponent)
             << traits::mantissa_bits) |
            mantissa);
    }

    // Round into the signed-magnitude subnormal field. If rounding
    // reaches 1 << mantissa_bits, that bit pattern is exactly the
    // smallest normal value.
    const int subnormal_shift =
        151 - traits::exponent_bias - traits::mantissa_bits -
        static_cast<int> (float_exponent);
    if (subnormal_shift >= 32) return sign;

    const std::uint32_t significand = float_mantissa | 0x800000u;
    const std::uint32_t rounded     = fp8_round_shift (
        significand, static_cast<unsigned int> (subnormal_shift));
    return static_cast<std::uint8_t> (sign | rounded);
}

template <FP8Format Format>
inline float
fp8_to_float_bits (std::uint8_t bits) IMATH_NOEXCEPT
{
    typedef fp8_traits<Format> traits;

    const std::uint32_t sign = (bits & traits::sign_mask) != 0
                                   ? 0x80000000u
                                   : 0u;
    const int exponent =
        (bits & traits::exponent_mask) >> traits::mantissa_bits;
    const std::uint32_t mantissa = bits & traits::mantissa_mask;

    if (exponent == 0)
    {
        if (mantissa == 0) return fp8_bits_float (sign);

        int leading_bit = 0;
        for (std::uint32_t m = mantissa; m > 1; m >>= 1) ++leading_bit;

        const int float_exponent =
            127 + 1 - traits::exponent_bias - traits::mantissa_bits +
            leading_bit;
        const std::uint32_t float_mantissa =
            (mantissa << (23 - leading_bit)) &
            std::uint32_t (0x7fffff);
        return fp8_bits_float (
            sign | (static_cast<std::uint32_t> (float_exponent) << 23) |
            float_mantissa);
    }

    if (traits::has_infinity && exponent == traits::exponent_field_max)
    {
        if (mantissa == 0)
            return fp8_bits_float (sign | 0x7f800000u);
        return fp8_bits_float (0x7fc00000u);
    }

    if (!traits::has_infinity &&
        (bits & ~traits::sign_mask) ==
            static_cast<std::uint8_t> (traits::positive_nan))
        return fp8_bits_float (0x7fc00000u);

    const std::uint32_t float_exponent =
        static_cast<std::uint32_t> (
            exponent - traits::exponent_bias + 127);
    const std::uint32_t float_mantissa =
        mantissa << (23 - traits::mantissa_bits);
    return fp8_bits_float (
        sign | (float_exponent << 23) | float_mantissa);
}

} // namespace detail

/// An 8-bit floating-point value.
///
/// Arithmetic follows the same promotion model as `half`: scalar
/// expressions are evaluated as `float` and rounded once when the
/// result is stored in an FP8 object. Compound assignment performs the
/// operation in `float` and rounds once.
template <FP8Format Format> class lowp8
{
public:
    enum FromBitsTag
    {
        FromBits
    };

    lowp8 () IMATH_NOEXCEPT = default;

    lowp8 (float f) IMATH_NOEXCEPT
        : _bits (detail::fp8_from_float_bits<Format> (f))
    {}

    lowp8 (half h) IMATH_NOEXCEPT : lowp8 (float (h)) {}

    constexpr lowp8 (FromBitsTag, std::uint8_t bits) IMATH_NOEXCEPT
        : _bits (bits)
    {}

    constexpr lowp8 (const lowp8&) IMATH_NOEXCEPT = default;
    constexpr lowp8 (lowp8&&) IMATH_NOEXCEPT      = default;
    ~lowp8 () IMATH_NOEXCEPT                      = default;

    lowp8& operator= (const lowp8&) IMATH_NOEXCEPT = default;
    lowp8& operator= (lowp8&&) IMATH_NOEXCEPT      = default;

    lowp8& operator= (float f) IMATH_NOEXCEPT
    {
        _bits = detail::fp8_from_float_bits<Format> (f);
        return *this;
    }

    lowp8& operator= (half h) IMATH_NOEXCEPT
    {
        return *this = float (h);
    }

    operator float () const IMATH_NOEXCEPT
    {
        return detail::fp8_to_float_bits<Format> (_bits);
    }

    half toHalf () const IMATH_NOEXCEPT { return half (float (*this)); }

    constexpr lowp8 operator- () const IMATH_NOEXCEPT
    {
        return lowp8 (FromBits, _bits ^ fp8_traits<Format>::sign_mask);
    }

    lowp8& operator+= (lowp8 v) IMATH_NOEXCEPT
    {
        return *this = float (*this) + float (v);
    }
    lowp8& operator+= (float f) IMATH_NOEXCEPT
    {
        return *this = float (*this) + f;
    }
    lowp8& operator-= (lowp8 v) IMATH_NOEXCEPT
    {
        return *this = float (*this) - float (v);
    }
    lowp8& operator-= (float f) IMATH_NOEXCEPT
    {
        return *this = float (*this) - f;
    }
    lowp8& operator*= (lowp8 v) IMATH_NOEXCEPT
    {
        return *this = float (*this) * float (v);
    }
    lowp8& operator*= (float f) IMATH_NOEXCEPT
    {
        return *this = float (*this) * f;
    }
    lowp8& operator/= (lowp8 v) IMATH_NOEXCEPT
    {
        return *this = float (*this) / float (v);
    }
    lowp8& operator/= (float f) IMATH_NOEXCEPT
    {
        return *this = float (*this) / f;
    }

    constexpr bool isFinite () const IMATH_NOEXCEPT
    {
        return !isNan () && !isInfinity ();
    }

    constexpr bool isNormalized () const IMATH_NOEXCEPT
    {
        const std::uint8_t e =
            (_bits & fp8_traits<Format>::exponent_mask) >>
            fp8_traits<Format>::mantissa_bits;
        return !isNan () && e != 0 &&
               (!fp8_traits<Format>::has_infinity ||
                e != fp8_traits<Format>::exponent_field_max);
    }

    constexpr bool isDenormalized () const IMATH_NOEXCEPT
    {
        return (_bits & fp8_traits<Format>::exponent_mask) == 0 &&
               (_bits & fp8_traits<Format>::mantissa_mask) != 0;
    }

    constexpr bool isZero () const IMATH_NOEXCEPT
    {
        return (_bits & ~fp8_traits<Format>::sign_mask) == 0;
    }

    constexpr bool isNan () const IMATH_NOEXCEPT
    {
        return fp8_traits<Format>::has_infinity
                   ? ((_bits & fp8_traits<Format>::exponent_mask) ==
                          fp8_traits<Format>::exponent_mask &&
                      (_bits & fp8_traits<Format>::mantissa_mask) != 0)
                   : ((_bits & ~fp8_traits<Format>::sign_mask) ==
                      fp8_traits<Format>::positive_nan);
    }

    constexpr bool isInfinity () const IMATH_NOEXCEPT
    {
        return fp8_traits<Format>::has_infinity
                   ? ((_bits & ~fp8_traits<Format>::sign_mask) ==
                      fp8_traits<Format>::positive_inf)
                   : false;
    }

    constexpr bool isNegative () const IMATH_NOEXCEPT
    {
        return (_bits & fp8_traits<Format>::sign_mask) != 0;
    }

    static constexpr lowp8 posInf () IMATH_NOEXCEPT
    {
        return lowp8 (FromBits, fp8_traits<Format>::positive_inf);
    }

    static constexpr lowp8 negInf () IMATH_NOEXCEPT
    {
        return lowp8 (
            FromBits,
            static_cast<std::uint8_t> (
                fp8_traits<Format>::sign_mask |
                fp8_traits<Format>::positive_inf));
    }

    static constexpr lowp8 qNan () IMATH_NOEXCEPT
    {
        return lowp8 (FromBits, fp8_traits<Format>::positive_nan);
    }

    static constexpr lowp8 sNan () IMATH_NOEXCEPT
    {
        return lowp8 (FromBits, fp8_traits<Format>::positive_snan);
    }

    constexpr std::uint8_t bits () const IMATH_NOEXCEPT { return _bits; }

    IMATH_CONSTEXPR14 void setBits (std::uint8_t bits) IMATH_NOEXCEPT
    {
        _bits = bits;
    }

private:
    std::uint8_t _bits;
};

static_assert (sizeof (lowp8<FP8Format::E4M3>) == 1, "FP8 must be 8 bits");
static_assert (sizeof (lowp8<FP8Format::E5M2>) == 1, "FP8 must be 8 bits");
static_assert (
    std::is_trivially_copyable<lowp8<FP8Format::E4M3>>::value,
    "FP8 must be trivially copyable");
static_assert (
    std::is_trivially_copyable<lowp8<FP8Format::E5M2>>::value,
    "FP8 must be trivially copyable");

/// OCP E4M3FN: best precision for quantized inference, no infinities.
using fp8_e4m3 = lowp8<FP8Format::E4M3>;
/// OCP E5M2: wider dynamic range with IEEE-style infinities.
using fp8_e5m2 = lowp8<FP8Format::E5M2>;

/// Shorter names for the default 8-bit format.
using fp8          = fp8_e4m3;
using quarter      = fp8_e4m3;
using fp8_e4m3fn   = fp8_e4m3;
using quarter_e4m3 = fp8_e4m3;
using quarter_e5m2 = fp8_e5m2;

template <> struct is_float_like<fp8_e4m3> : public std::true_type
{};
template <> struct is_float_like<fp8_e5m2> : public std::true_type
{};

template <FP8Format Format>
inline std::ostream&
operator<< (std::ostream& os, lowp8<Format> v)
{
    os << float (v);
    return os;
}

template <FP8Format Format>
inline std::istream&
operator>> (std::istream& is, lowp8<Format>& v)
{
    float f;
    is >> f;
    v = lowp8<Format> (f);
    return is;
}

/// Convert a contiguous float array to FP8 using nearest-even rounding.
template <FP8Format Format>
inline void
fp8_batch_convert (const float* src, lowp8<Format>* dst, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) dst[i] = lowp8<Format> (src[i]);
}

/// Convert a contiguous FP8 array to float.
template <FP8Format Format>
inline void
fp8_batch_convert (const lowp8<Format>* src, float* dst, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) dst[i] = float (src[i]);
}

/// Convert half values to FP8. Every finite FP8 value converts back to
/// the same half value without loss.
template <FP8Format Format>
inline void
fp8_batch_convert (const half* src, lowp8<Format>* dst, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) dst[i] = lowp8<Format> (src[i]);
}

/// Convert FP8 values to half.
template <FP8Format Format>
inline void
fp8_batch_convert (const lowp8<Format>* src, half* dst, std::size_t count)
{
    for (std::size_t i = 0; i < count; ++i) dst[i] = src[i].toHalf ();
}

namespace detail
{

template <FP8Format Format, int Operation>
inline void
fp8_batch_binary (
    const lowp8<Format>* a,
    const lowp8<Format>* b,
    lowp8<Format>*       dst,
    std::size_t          count)
{
    const std::size_t block_size = 32;
    std::size_t       i          = 0;

    for (; i + block_size <= count; i += block_size)
    {
        float af[block_size];
        float bf[block_size];

        for (std::size_t j = 0; j < block_size; ++j)
            af[j] = float (a[i + j]);
        for (std::size_t j = 0; j < block_size; ++j)
            bf[j] = float (b[i + j]);
        IMATH_FP8_IVDEP
        for (std::size_t j = 0; j < block_size; ++j)
        {
            if (Operation == 0)
                af[j] += bf[j];
            else if (Operation == 1)
                af[j] -= bf[j];
            else
                af[j] *= bf[j];
        }
        for (std::size_t j = 0; j < block_size; ++j)
            dst[i + j] = lowp8<Format> (af[j]);
    }

    for (; i < count; ++i)
    {
        const float av = float (a[i]);
        const float bv = float (b[i]);
        if (Operation == 0)
            dst[i] = lowp8<Format> (av + bv);
        else if (Operation == 1)
            dst[i] = lowp8<Format> (av - bv);
        else
            dst[i] = lowp8<Format> (av * bv);
    }
}

} // namespace detail

/// Elementwise FP8 addition. The result of each element is rounded once.
template <FP8Format Format>
inline void
fp8_batch_add (
    const lowp8<Format>* a,
    const lowp8<Format>* b,
    lowp8<Format>*       dst,
    std::size_t          count)
{
    detail::fp8_batch_binary<Format, 0> (a, b, dst, count);
}

/// Elementwise FP8 subtraction. The result of each element is rounded once.
template <FP8Format Format>
inline void
fp8_batch_sub (
    const lowp8<Format>* a,
    const lowp8<Format>* b,
    lowp8<Format>*       dst,
    std::size_t          count)
{
    detail::fp8_batch_binary<Format, 1> (a, b, dst, count);
}

/// Elementwise FP8 multiplication. The result of each element is rounded once.
template <FP8Format Format>
inline void
fp8_batch_mul (
    const lowp8<Format>* a,
    const lowp8<Format>* b,
    lowp8<Format>*       dst,
    std::size_t          count)
{
    detail::fp8_batch_binary<Format, 2> (a, b, dst, count);
}

#undef IMATH_FP8_IVDEP

IMATH_STABLE_3_2_EXIT

namespace std
{

template <IMATH_NAMESPACE::FP8Format Format>
class numeric_limits<IMATH_NAMESPACE::lowp8<Format>>
{
public:
    typedef IMATH_NAMESPACE::fp8_traits<Format> traits;
    typedef IMATH_NAMESPACE::lowp8<Format>      value_type;

    static const bool is_specialized = true;

    static constexpr value_type min () IMATH_NOEXCEPT
    {
        return value_type (value_type::FromBits, traits::positive_min);
    }
    static constexpr value_type max () IMATH_NOEXCEPT
    {
        return value_type (value_type::FromBits, traits::positive_max);
    }
    static constexpr value_type lowest () IMATH_NOEXCEPT
    {
        return value_type (
            value_type::FromBits,
            static_cast<std::uint8_t> (
                traits::sign_mask | traits::positive_max));
    }

    static constexpr int digits       = traits::digits;
    static constexpr int digits10     = 0;
    static constexpr int max_digits10 = traits::max_digits10;
    static constexpr bool is_signed   = true;
    static constexpr bool is_integer  = false;
    static constexpr bool is_exact    = false;
    static constexpr int  radix       = 2;

    static constexpr value_type epsilon () IMATH_NOEXCEPT
    {
        return value_type (value_type::FromBits, traits::positive_eps);
    }
    static constexpr value_type round_error () IMATH_NOEXCEPT
    {
        return value_type (value_type::FromBits, traits::positive_half);
    }

    static constexpr int min_exponent   = traits::min_exponent;
    static constexpr int min_exponent10 = traits::min_exponent_10;
    static constexpr int max_exponent   = traits::max_normal_exponent + 1;
    static constexpr int max_exponent10 = traits::max_exponent_10;

    static constexpr bool has_infinity      = traits::has_infinity;
    static constexpr bool has_quiet_NaN     = true;
    static constexpr bool has_signaling_NaN = traits::has_signaling_nan;

#if (defined(__clang__) || defined(__GNUC__)) && (__cplusplus >= 202302L)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER) && defined(_MSVC_LANG) && (_MSVC_LANG >= 202302L)
#    pragma warning(push)
#    pragma warning(disable : 4996)
#endif
    static constexpr float_denorm_style has_denorm = denorm_present;
#if (defined(__clang__) || defined(__GNUC__)) && (__cplusplus >= 202302L)
#    pragma GCC diagnostic pop
#elif defined(_MSC_VER) && defined(_MSVC_LANG) && (_MSVC_LANG >= 202302L)
#    pragma warning(pop)
#endif

    static constexpr bool has_denorm_loss = false;

    static constexpr value_type infinity () IMATH_NOEXCEPT
    {
        return value_type (
            value_type::FromBits, traits::positive_inf);
    }
    static constexpr value_type quiet_NaN () IMATH_NOEXCEPT
    {
        return value_type (
            value_type::FromBits, traits::positive_nan);
    }
    static constexpr value_type signaling_NaN () IMATH_NOEXCEPT
    {
        return value_type (
            value_type::FromBits, traits::positive_snan);
    }
    static constexpr value_type denorm_min () IMATH_NOEXCEPT
    {
        return value_type (value_type::FromBits, traits::positive_denorm);
    }

    static constexpr bool is_iec559  = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo  = false;

    static constexpr bool              traps           = false;
    static constexpr bool              tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};

} // namespace std

#endif // INCLUDED_IMATHFP8_H
