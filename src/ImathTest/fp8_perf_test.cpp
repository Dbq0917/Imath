//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#include <ImathFP8.h>
#include <ImathRandom.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace IMATH_NAMESPACE;

namespace
{

typedef std::chrono::steady_clock clock_type;

template <class Work>
std::uint64_t
bestTime (Work work, int repetitions)
{
    work ();
    std::uint64_t best = ~std::uint64_t (0);
    for (int i = 0; i < repetitions; ++i)
    {
        const clock_type::time_point start = clock_type::now ();
        work ();
        const clock_type::time_point end = clock_type::now ();
        const std::uint64_t elapsed =
            static_cast<std::uint64_t> (
                std::chrono::duration_cast<std::chrono::nanoseconds> (
                    end - start)
                    .count ());
        best = std::min (best, elapsed);
    }
    return best;
}

double
millionValuesPerSecond (std::size_t count, std::uint64_t nanoseconds)
{
    if (nanoseconds == 0) return 0.0;
    return static_cast<double> (count) * 1.0e3 /
           static_cast<double> (nanoseconds);
}

struct ConversionWork
{
    const float*        src;
    fp8_e4m3*           dst;
    std::size_t         count;
    bool                large;

    void operator() () const
    {
        if (large)
            fp8_batch_convert (src, dst, count);
        else
            for (std::size_t i = 0; i < count; ++i)
                dst[i] = fp8_e4m3 (src[i]);
    }
};

struct MultiplyWork
{
    const fp8_e4m3* a;
    const fp8_e4m3* b;
    fp8_e4m3*       dst;
    std::size_t     count;
    bool            large;

    void operator() () const
    {
        if (large)
            fp8_batch_mul (a, b, dst, count);
        else
            for (std::size_t i = 0; i < count; ++i)
                dst[i] = fp8_e4m3 (float (a[i]) * float (b[i]));
    }
};

double
report (
    const char*     label,
    std::size_t     count,
    std::uint64_t   libraryTime,
    std::uint64_t   scalarTime)
{
    const double libraryRate = millionValuesPerSecond (count, libraryTime);
    const double scalarRate  = millionValuesPerSecond (count, scalarTime);
    const double ratio       = libraryRate / scalarRate;

    std::printf (
        "%-24s library %8.1f Mvalues/s  scalar %8.1f Mvalues/s  "
        "library/scalar %.2fx\n",
        label,
        libraryRate,
        scalarRate,
        ratio);
    return ratio;
}

} // namespace

int
main (int argc, char* argv[])
{
    std::size_t count = 1u << 20;
    if (argc > 1)
    {
        const long requested = std::atol (argv[1]);
        if (requested <= 0)
        {
            std::fprintf (stderr, "Invalid element count '%s'\n", argv[1]);
            return 1;
        }
        count = static_cast<std::size_t> (requested);
    }

    std::vector<float> source (count);
    std::vector<fp8_e4m3> a (count);
    std::vector<fp8_e4m3> b (count);
    std::vector<fp8_e4m3> libraryResult (count);
    std::vector<fp8_e4m3> scalarResult (count);

    Rand48 random (0x5eedu);
    for (std::size_t i = 0; i < count; ++i)
    {
        source[i] = float (random.nextf (-448.0f, 448.0f));
        a[i] = fp8_e4m3 (source[i]);
        b[i] = fp8_e4m3 (float (random.nextf (-16.0f, 16.0f)));
    }

    fp8_batch_convert (source.data (), libraryResult.data (), count);
    for (std::size_t i = 0; i < count; ++i)
        scalarResult[i] = fp8_e4m3 (source[i]);
    if (libraryResult != scalarResult)
    {
        std::fprintf (stderr, "FP8 batch conversion mismatch\n");
        return 1;
    }

    fp8_batch_mul (a.data (), b.data (), libraryResult.data (), count);
    for (std::size_t i = 0; i < count; ++i)
        scalarResult[i] = fp8_e4m3 (float (a[i]) * float (b[i]));
    if (libraryResult != scalarResult)
    {
        std::fprintf (stderr, "FP8 batch multiply mismatch\n");
        return 1;
    }

    const int repetitions = 12;
    const std::uint64_t libraryConvert = bestTime (
        ConversionWork {
            source.data (), libraryResult.data (), count, true},
        repetitions);
    const std::uint64_t scalarConvert = bestTime (
        ConversionWork {
            source.data (), scalarResult.data (), count, false},
        repetitions);

    const std::uint64_t libraryMultiply = bestTime (
        MultiplyWork {
            a.data (), b.data (), libraryResult.data (), count, true},
        repetitions);
    const std::uint64_t scalarMultiply = bestTime (
        MultiplyWork {
            a.data (), b.data (), scalarResult.data (), count, false},
        repetitions);

    std::printf (
        "FP8 E4M3 batch throughput, %zu elements, best of %d:\n",
        count,
        repetitions);
    const double convertRatio =
        report ("float -> fp8", count, libraryConvert, scalarConvert);
    const double multiplyRatio =
        report ("fp8 multiply", count, libraryMultiply, scalarMultiply);

    // The batch path must not fall back to a materially slower
    // per-element implementation. This is intentionally conservative
    // to tolerate noisy shared CI machines.
    if (convertRatio < 0.50 || multiplyRatio < 0.50)
    {
        std::fprintf (
            stderr,
            "FP8 batch path is more than 2x slower than the scalar "
            "reference\n");
        return 1;
    }

    return 0;
}
