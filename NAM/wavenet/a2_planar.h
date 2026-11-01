#pragma once

// Planar kernels for the A2 fast path (AArch64, ARMv7, and x86_64 AVX2+FMA).
//
// These are drop-in replacements for A2FastModel<3> and A2FastModel<8>. On ARM
// they produce **bit-identical** output: not "within a tolerance", not "below
// the noise floor" -- the same float32 bits, sample for sample. On AVX2 the
// match is the same scalar-order reduction as the ARM kernels (order A), which
// lands ~133–139 dB below the signal versus Eigen's gebp-backed a2_fast — see
// the AVX2 section below.
//
// The idea in one line: a2_fast keeps the channels of a frame adjacent in
// memory and vectorises across channels; these kernels keep each channel in its
// own plane and vectorise across *frames*, so one SIMD lane runs a2_fast's
// per-frame scalar reduction verbatim. Nothing is reassociated, which is what
// makes the bit-identity claim hold on ARM rather than being a lucky accident.
//
// -----------------------------------------------------------------------------
// Where this is active, and where it is not
//
// NAM_A2_PLANAR is defined when the A2 fast path is built for an ARM target
// with fused multiply-add, or for x86_64 with AVX2 and FMA. Elsewhere this
// header declares nothing, a2_planar.cpp / a2_planar_avx2.cpp compile to
// objects with no symbols, the call site in a2_fast.cpp is preprocessed away,
// and the A2 path is byte for byte the code that is there today.
//
// The gate was __APPLE__ && __aarch64__ at first, because Apple Silicon was the
// only place these had been built and measured. It has since been widened
// three times, each time on evidence rather than optimism: AArch64 generally,
// ARMv7 with NEON+FMA, then x86_64 AVX2+FMA.
//
// -----------------------------------------------------------------------------
// At a glance
//
// The promoted kernel is a2_planar in every case: one engine, six
// configurations. What varies is the tile width, how a weight reaches the
// multiplier, and -- at C=8 on ARM -- the shape of the conv loop.
//
//   target     submodel      C  tile  weights    C=8 conv loop
//   ------------------------------------------------------------------
//   AArch64    A2 nano       3    32  by-lane    --
//   AArch64    A2 standard   8     8  by-lane    fold over lambdas
//   ARMv7      A2 nano       3     8  broadcast  --
//   ARMv7      A2 standard   8     8  broadcast  plain loop nest
//   AVX2+FMA   A2 nano       3    32  broadcast  --
//   AVX2+FMA   A2 standard   8     8  broadcast  plain loop nest
//
// "by-lane" is vfmaq_laneq_f32, which encodes the lane in the instruction;
// "broadcast" is vld1q_dup_f32 (ARM) or _mm256_broadcast_ss (AVX2) at the
// point of use, because ARMv7 and AVX2 have no by-element FMA. Both sections
// below say why the remaining two columns differ, and neither difference was
// inherited -- each was measured on the part.
//
// The speed (ARM rows bit-identical to a2_fast over a full render; AVX2 rows
// from the NAMBench full protocol on the named part):
//
//   part                  submodel      C  vs a2_fast  conditions
//   ------------------------------------------------------------------
//   Apple M2              A2 nano       3       2.00x  see note
//   Apple M2              A2 standard   8       2.47x  see note
//   Cortex-A76 (Pi 500)   A2 nano       3       2.87x  block 32
//   Cortex-A76 (Pi 500)   A2 standard   8       2.40x  block 32
//   Cortex-A17 (RK3288)   A2 nano       3       1.47x  block 32, 1416 MHz
//   Cortex-A17 (RK3288)   A2 standard   8       1.42x  block 32, 1416 MHz
//   i5-9500T (pro)        A2 nano       3       5.52x  block 64, ~139 dB
//   i5-9500T (pro)        A2 standard   8       2.66x  block 64, ~133 dB
//   i7-4770HQ (vbox)      A2 nano       3       7.56x  Ampeg, 8 s window
//   i7-4770HQ (vbox)      A2 standard   8       3.19x  Ampeg, 8 s window
//
// The M2 rows are carried from the Apple Silicon campaign these kernels came
// out of, and have not been re-measured since; every other row is a direct
// measurement of the code as it stands. The A76 rows are at a 32-frame block,
// which is why they do not match the 2.13x/2.94x quoted just below -- same
// code, different block size, not a discrepancy.
//
// Note the inversion. On the M2 the wide model wins bigger; on both ARM parts
// and on the i5-9500T the narrow one does. That is the same architectural
// story as the tile widths further down, and the reason none of this travels
// between targets by assumption.
// -----------------------------------------------------------------------------
//
// AArch64 generally:
//
//   * Bit-identity holds off Apple. The property it leans on is the compiler
//     contracting a*b+c into an FMA inside a2_fast's *own* 3-channel branch,
//     which is a toolchain behaviour, not an architectural one. Checked on a
//     Cortex-A76 (Raspberry Pi 500, Ubuntu 24.04) under GCC 13, and on Neoverse
//     N2 under GCC 14 and Clang 18: both submodels bit-identical to a2_fast,
//     max|diff| exactly zero, over a full render.
//
//   * The speed holds too, though the shape of the win is not the same. On an
//     M2: 2.47x on A2 standard and 2.00x on A2 nano. On a Cortex-A76: 2.13x and
//     2.94x. Faster on both parts, on both submodels. (At a 32-frame block the
//     A76 reads 2.40x and 2.87x; see the table above.)
//
// ARMv7-A with NEON and VFPv4 (see the caveats below, which are load-bearing):
//
//   * Measured on a Rockchip RK3288 (quad Cortex-A17, ASUS Tinker Board), GCC
//     13.3, clock-pinned to 1416 MHz, over the same 523,808-frame render. Both
//     submodels bit-identical to a2_fast, max|diff| exactly zero.
//
//   * A2 standard 78.79% -> 55.63% of one core and A2 nano 12.29% -> 8.36% at
//     32-frame blocks (1.416x and 1.470x). Smaller than the AArch64 wins, and
//     on this part that is the difference between one instance per core with
//     nothing left over and one with room for the rest of a signal chain.
//
// x86_64 with AVX2 and FMA (body in a2_planar_avx2.cpp):
//
//   * Measured on an Intel Core i5-9500T (NAMBench host `pro`), GCC with
//     -mavx2 -mfma -ffp-contract=fast, performance governor, full protocol
//     (5 s warm-up / 30 s timing, block 64) over the same 523,808-frame
//     render. A2 standard 6.507% -> 2.446% of one core (2.660x); A2 nano
//     1.875% -> 0.340% (5.517x). Tiles and switches were swept on this part
//     in the X64 lab (Sources/X64Engines); winners are tile 8 / tile 32 with
//     ring-direct, skip-last 1x1, and a linear-rewind ring — not inherited
//     from ARM or from the M2.
//
//   * The same body on a Core i7-4770HQ in VirtualBox (NAMBench host `mp`,
//     testbed `i7-4770HQ-vbox`, one vCPU pinned). The X64 lab ladder picks the
//     same compositions: C=8 tile 8 and C=3 tile 32, both with ring-direct,
//     skip-last, and the linear ring. On the Ampeg capture, an 8-second
//     window gives A2 standard 12.36% -> 3.88% of one core (3.19x) and A2
//     nano 4.51% -> 0.60% (7.56x). Pass-to-pass spread on that guest is about
//     5–16%, so the run does not meet the protocol's 3% agreement gate and
//     is not a Bencher upload. a2_planar and the lab's stacked winners sit
//     inside that spread, which is why this part does not get its own body.
//
//   * Parity versus Eigen-backed a2_fast is ~133 dB (C=8) / ~139 dB (C=3),
//     not bit-identical. Eigen's AVX2 gebp for N≥4 does not match the
//     per-tap FMA order these kernels share with the NEON path (order A).
//     Within the X64 lab the same kernels are bit-identical to that lab's
//     order-A baseline; the cross-library gap is the Eigen packing, not a
//     promotion bug. See the Eigen order probe in the NAMBench X64 campaign.
//
// __aarch64__ specifically, rather than a spelling that would also catch MSVC's
// _M_ARM64. That is deliberate and is the one part of the old gate worth
// keeping: MSVC at /fp:precise does not contract a*b+c into an FMA, so the
// reference branch it would be compared against computes something else, and
// bit-identity -- the whole claim -- would not hold. clang-cl on ARM64 defines
// __aarch64__ and is fine.
//
// The ARMv7 arm of the gate additionally requires __ARM_NEON and
// __ARM_FEATURE_FMA, and both are load-bearing rather than defensive:
//
//   * Without NEON there is no kernel at all.
//   * Without FMA, Eigen stops defining EIGEN_VECTORIZE_FMA and selects the
//     non-fused vmlaq_f32 for every pmadd, which changes a2_fast's *own* C=8
//     arithmetic. The kernels would then be bit-identical to a reference that
//     is no longer there. -mfpu=neon alone reaches that state; -mfpu=neon-vfpv4
//     is what makes the comparison meaningful.
//
// -----------------------------------------------------------------------------
// Three caveats on ARMv7, stated here rather than buried
//
//   1. **Bit-identity rests on FPSCR.FZ.** AArch32 Advanced SIMD is
//      *unconditionally* flush-to-zero for single precision, while the VFP
//      scalar code these kernels are compared against honours the FZ bit. The
//      two agree bit-for-bit only when the host has FZ set -- which audio hosts
//      usually do, and which a host application is under no obligation to do. If
//      FZ is clear and a denormal reaches a layer, the NEON kernel and the
//      scalar reference will differ. This is the one condition under which the
//      claim above is false. AArch64 has no such caveat: there FPCR.FZ applies
//      to scalar and vector alike, so both sides move together.
//
//   2. **The C=8 claim is a GCC claim.** Under clang 18.1.3 on this target
//      Eigen's gebp_kernel emits non-fused vmla.f32, so a2_fast itself computes
//      different bits (reference checksum -17.478711597881365 under GCC against
//      -17.478718637490147 under clang) and these kernels land 127.5 dB from it
//      rather than at zero. The general honest form of the claim is
//      "bit-identical wherever the reference contracts its own FMAs".
//
//   3. **The C=3 speed is a GCC claim**, for an unrelated reason: clang declines
//      to inline the tile helpers and spills the accumulators the kernel exists
//      to keep in registers, which costs most of the win. Correctness is
//      unaffected; only the speed is.
//
// -----------------------------------------------------------------------------
// What is per-architecture here, and why none of it was inherited
//
// Two things vary by target, and neither changes the output -- only the speed.
//
// **Tile widths**, and the difference is not a small one. They were swept
// separately on each architecture, and the AArch64 values are actively wrong on
// ARMv7: the C=3 ladder peaks at 32 frames on an M2 and at 8 on a Cortex-A17,
// with tile 32 running *slower than a2_fast* there. ARMv7 has 16 Q registers
// against AArch64's 32, and tile 8 is the last rung whose accumulators fit.
//
// **The shape of the C=8 conv loop**, which is the less obvious one and cost
// more to find. The AArch64 form unrolls the input and output channels with a
// fold over generic lambdas, because it needs each index as a compile-time
// value for the by-lane FMA encoding. ARMv7 has no by-lane FMA, so it gains
// nothing from that -- and loses a great deal: with 16 registers the
// accumulators cannot stay resident, and GCC schedules the unavoidable spill
// traffic far better for a plain loop nest than for the fold. The two forms are
// 68.7% and 55.4% of one core on the same part. See the comment at the branch
// itself for the instruction and memory-access counts.
//
// The general lesson, stated because it was learned the expensive way: porting
// these kernels is not a matter of widening the gate and re-sweeping the tiles.
// The first ARMv7 build of this file was bit-identical, passed every
// conformance check, and was still slower than the kernel it was ported from by
// enough to lose most of the win -- and nothing but a measurement on the part
// said so. Re-measure against the reference on the target itself, and do not
// trust a figure carried over from a lab kernel of the same shape.
//
// See A32-PATH.md in the NAMBench repository for the full ladder and the spill
// counts underneath it.
//
// NAM_DISABLE_A2_PLANAR opts out anywhere, which is what makes an A/B
// measurement against the reference a one-flag change.
// -----------------------------------------------------------------------------

#if defined(NAM_ENABLE_A2_FAST)

  #if !defined(NAM_DISABLE_A2_PLANAR)
    #if defined(__aarch64__)
      #define NAM_A2_PLANAR 1
    #elif defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
      #define NAM_A2_PLANAR 1
      /// 32-bit ARM: 16 Q registers, no by-element FMA, narrower NEON datapath.
      /// Selects the ARMv7 weight delivery and tile widths in a2_planar.cpp.
      #define NAM_A2_PLANAR_A32 1
    #elif defined(__AVX2__) && defined(__FMA__)
      #define NAM_A2_PLANAR 1
      /// x86_64 AVX2+FMA: 16 YMM registers, broadcast weight delivery (no
      /// by-element FMA). Selects the AVX2 body in a2_planar_avx2.cpp.
      #define NAM_A2_PLANAR_AVX2 1
    #endif
  #endif

  #if defined(NAM_A2_PLANAR)

    #include <memory>
    #include <vector>

    #include "../dsp.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

/// \brief Build the planar model for an A2 submodel (NEON or AVX2 body).
/// \param channels 3 (A2 nano) or 8 (A2 standard); anything else yields nullptr.
/// \param weights The A2 weight stream, consumed in A2FastModel's order.
/// \param expected_sample_rate Passed through to DSP.
/// \return The model, or nullptr when this channel count has no planar kernel
///         (the caller then falls back to A2FastModel).
std::unique_ptr<DSP> create_a2_planar_model(int channels, std::vector<float> weights, double expected_sample_rate);

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

  #endif // NAM_A2_PLANAR
#endif // NAM_ENABLE_A2_FAST
