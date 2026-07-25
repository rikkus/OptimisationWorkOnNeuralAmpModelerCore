// Direct vs partitioned-FFT parity for dsp::PartitionedConvolution /
// dsp::ImpulseResponse (cab FIR path).

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ImpulseResponse.h"
#include "PartitionedConvolution.h"

#include "allocation_tracking.h"

namespace test_ir_convolution
{
namespace
{

std::vector<float> MakeSyntheticIR(size_t numTaps, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> ir(numTaps);
  for (size_t i = 0; i < numTaps; i++)
  {
    // Slight decay so tails are quieter (more IR-like).
    const float envelope = std::exp(-3.0f * static_cast<float>(i) / static_cast<float>(numTaps));
    ir[i] = dist(rng) * envelope;
  }
  return ir;
}

std::vector<float> MakeNoise(size_t numSamples, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> x(numSamples);
  for (size_t i = 0; i < numSamples; i++)
    x[i] = dist(rng);
  return x;
}

void AssertClose(const std::vector<float>& a, const std::vector<float>& b, float absTol, const char* label)
{
  assert(a.size() == b.size());
  double maxAbs = 0.0;
  double sumSqErr = 0.0;
  double sumSqRef = 0.0;
  for (size_t i = 0; i < a.size(); i++)
  {
    const double err = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    maxAbs = std::max(maxAbs, std::abs(err));
    sumSqErr += err * err;
    sumSqRef += static_cast<double>(a[i]) * static_cast<double>(a[i]);
  }
  const double rmsErr = std::sqrt(sumSqErr / static_cast<double>(a.size()));
  const double rmsRef = std::sqrt(sumSqRef / static_cast<double>(a.size()));
  const double relDb = rmsRef > 1e-12 ? 20.0 * std::log10(rmsErr / rmsRef) : -300.0;
  if (maxAbs > absTol)
  {
    std::cerr << label << " failed: maxAbs=" << maxAbs << " rmsErr=" << rmsErr << " relDb=" << relDb << "\n";
  }
  assert(maxAbs <= absTol);
  // Also require roughly -90 dB relative when signal is present.
  if (rmsRef > 1e-4)
    assert(relDb < -90.0);
}

void CompareDirectVsFft(size_t numTaps, size_t blockSize, size_t numBlocks, float absTol, const char* label)
{
  const auto ir = MakeSyntheticIR(numTaps, 1);
  const auto input = MakeNoise(blockSize * numBlocks, 2);

  dsp::PartitionedConvolution direct;
  dsp::PartitionedConvolution fft;
  direct.SetKernel(ir.data(), ir.size(), dsp::ConvolutionImplementation::Direct);
  fft.SetKernel(ir.data(), ir.size(), dsp::ConvolutionImplementation::FFT);

  assert(direct.GetActiveImplementation() == dsp::ConvolutionImplementation::Direct);
  assert(fft.GetActiveImplementation() == dsp::ConvolutionImplementation::FFT);

  std::vector<float> outDirect(input.size());
  std::vector<float> outFft(input.size());

  for (size_t b = 0; b < numBlocks; b++)
  {
    const float* in = input.data() + b * blockSize;
    direct.Process(in, outDirect.data() + b * blockSize, blockSize);
    fft.Process(in, outFft.data() + b * blockSize, blockSize);
  }

  AssertClose(outDirect, outFft, absTol, label);
}

void CompareImpulseResponseDirectVsFft(size_t numTaps, size_t blockSize, size_t numBlocks, float absTol,
                                       const char* label)
{
  const auto ir = MakeSyntheticIR(numTaps, 7);
  const auto input = MakeNoise(blockSize * numBlocks, 8);

  dsp::ImpulseResponse::IRData data;
  data.mRawAudio = ir;
  data.mRawAudioSampleRate = 48000.0;

  dsp::ImpulseResponse direct(data, 48000.0, dsp::ConvolutionImplementation::Direct);
  dsp::ImpulseResponse fft(data, 48000.0, dsp::ConvolutionImplementation::FFT);
  assert(direct.GetWavState() == dsp::wav::LoadReturnCode::SUCCESS);
  assert(fft.GetWavState() == dsp::wav::LoadReturnCode::SUCCESS);
  assert(direct.GetActiveImplementation() == dsp::ConvolutionImplementation::Direct);
  assert(fft.GetActiveImplementation() == dsp::ConvolutionImplementation::FFT);
  assert(direct.GetNumTaps() == numTaps);
  assert(fft.GetNumTaps() == numTaps);

  std::vector<double> inBuf(blockSize);
  std::vector<double> outDirect(blockSize * numBlocks);
  std::vector<double> outFft(blockSize * numBlocks);
  double* inPtr = inBuf.data();

  for (size_t b = 0; b < numBlocks; b++)
  {
    for (size_t i = 0; i < blockSize; i++)
      inBuf[i] = input[b * blockSize + i];
    double** dOut = direct.Process(&inPtr, 1, blockSize);
    double** fOut = fft.Process(&inPtr, 1, blockSize);
    for (size_t i = 0; i < blockSize; i++)
    {
      outDirect[b * blockSize + i] = dOut[0][i];
      outFft[b * blockSize + i] = fOut[0][i];
    }
  }

  std::vector<float> a(outDirect.size()), bb(outFft.size());
  for (size_t i = 0; i < a.size(); i++)
  {
    a[i] = static_cast<float>(outDirect[i]);
    bb[i] = static_cast<float>(outFft[i]);
  }
  AssertClose(a, bb, absTol, label);
}

} // namespace

void test_auto_selects_direct_for_short_ir()
{
  auto ir = MakeSyntheticIR(64, 3);
  dsp::PartitionedConvolution conv;
  conv.SetKernel(ir.data(), ir.size(), dsp::ConvolutionImplementation::Auto);
  assert(conv.GetActiveImplementation() == dsp::ConvolutionImplementation::Direct);
  std::cout << "  test_ir_convolution::test_auto_selects_direct_for_short_ir\n";
}

void test_auto_selects_fft_for_long_ir()
{
  auto ir = MakeSyntheticIR(512, 4);
  dsp::PartitionedConvolution conv;
  conv.SetKernel(ir.data(), ir.size(), dsp::ConvolutionImplementation::Auto);
  assert(conv.GetActiveImplementation() == dsp::ConvolutionImplementation::FFT);
  std::cout << "  test_ir_convolution::test_auto_selects_fft_for_long_ir\n";
}

void test_direct_matches_fft_block_sizes()
{
  // Tolerances: float FFT reordering vs direct FIR.
  CompareDirectVsFft(64, 16, 32, 2e-4f, "taps64/block16");
  CompareDirectVsFft(512, 64, 40, 5e-4f, "taps512/block64");
  CompareDirectVsFft(8192, 64, 40, 1e-3f, "taps8192/block64");
  CompareDirectVsFft(8192, 23, 60, 1e-3f, "taps8192/block23");
  CompareDirectVsFft(8192, 256, 20, 1e-3f, "taps8192/block256");
  std::cout << "  test_ir_convolution::test_direct_matches_fft_block_sizes\n";
}

void test_impulse_response_direct_matches_fft()
{
  CompareImpulseResponseDirectVsFft(512, 64, 30, 5e-4f, "IR taps512");
  CompareImpulseResponseDirectVsFft(8192, 64, 30, 1e-3f, "IR taps8192");
  std::cout << "  test_ir_convolution::test_impulse_response_direct_matches_fft\n";
}

void test_ya_ampg_ir_file_parity()
{
  const char* path = "example_models/YA AMPG 410 HS 52-CNT.wav";
  dsp::ImpulseResponse direct(path, 48000.0, dsp::ConvolutionImplementation::Direct);
  dsp::ImpulseResponse fft(path, 48000.0, dsp::ConvolutionImplementation::FFT);
  if (direct.GetWavState() != dsp::wav::LoadReturnCode::SUCCESS)
  {
    std::cout << "  test_ir_convolution::test_ya_ampg_ir_file_parity SKIPPED (wav not found)\n";
    return;
  }
  assert(fft.GetWavState() == dsp::wav::LoadReturnCode::SUCCESS);
  assert(direct.GetNumTaps() == 8192);
  assert(fft.GetNumTaps() == 8192);
  assert(fft.GetActiveImplementation() == dsp::ConvolutionImplementation::FFT);

  constexpr size_t blockSize = 64;
  constexpr size_t numBlocks = 50;
  auto input = MakeNoise(blockSize * numBlocks, 11);
  std::vector<double> inBuf(blockSize);
  std::vector<float> outD, outF;
  outD.reserve(input.size());
  outF.reserve(input.size());
  double* inPtr = inBuf.data();
  for (size_t b = 0; b < numBlocks; b++)
  {
    for (size_t i = 0; i < blockSize; i++)
      inBuf[i] = input[b * blockSize + i];
    double** d = direct.Process(&inPtr, 1, blockSize);
    double** f = fft.Process(&inPtr, 1, blockSize);
    for (size_t i = 0; i < blockSize; i++)
    {
      outD.push_back(static_cast<float>(d[0][i]));
      outF.push_back(static_cast<float>(f[0][i]));
    }
  }
  AssertClose(outD, outF, 2e-3f, "YA AMPG direct vs FFT");
  std::cout << "  test_ir_convolution::test_ya_ampg_ir_file_parity\n";
}

void test_fft_process_realtime_safe()
{
  auto ir = MakeSyntheticIR(8192, 5);
  dsp::PartitionedConvolution conv;
  conv.SetKernel(ir.data(), ir.size(), dsp::ConvolutionImplementation::FFT);
  auto input = MakeNoise(64, 6);
  std::vector<float> output(64);

  allocation_tracking::run_allocation_test_no_allocations(
    [&]() { conv.Process(input.data(), output.data(), 64); },
    [&]() {
      for (int i = 0; i < 100; i++)
        conv.Process(input.data(), output.data(), 64);
    },
    nullptr, "PartitionedConvolution FFT Process real-time safe");
  std::cout << "  test_ir_convolution::test_fft_process_realtime_safe\n";
}

} // namespace test_ir_convolution
