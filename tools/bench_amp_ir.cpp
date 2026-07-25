#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ImpulseResponse.h"
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
#include "wav.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

namespace
{
constexpr size_t kBufferSize = 64;
constexpr size_t kIrMaxLength = 8192; // matches ImpulseResponse::mMaxLength

enum class Pass
{
  All,
  Model,
  Ir,
  Combined
};

struct TimingResult
{
  const char* name;
  double wall_ms;
  size_t num_buffers;
  double sample_rate;
  size_t buffer_size;
};

void PrintTiming(const TimingResult& r)
{
  const double audio_ms = (static_cast<double>(r.num_buffers) * static_cast<double>(r.buffer_size) / r.sample_rate)
                          * 1000.0;
  const double rtf = audio_ms / r.wall_ms;
  const double per_block_us = (r.wall_ms * 1000.0) / static_cast<double>(r.num_buffers);
  const double deadline_us = (static_cast<double>(r.buffer_size) / r.sample_rate) * 1e6;
  const double deadline_pct = 100.0 * per_block_us / deadline_us;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  " << r.name << ":\n";
  std::cout << "    wall:      " << r.wall_ms << " ms  (audio " << audio_ms << " ms)\n";
  std::cout << "    RTF:       " << rtf << "x\n";
  std::cout << "    per-block: " << per_block_us << " us  (deadline " << deadline_us << " us, " << deadline_pct
            << "% of deadline)\n";
}

Pass ParsePass(const std::string& s)
{
  if (s == "all")
    return Pass::All;
  if (s == "model")
    return Pass::Model;
  if (s == "ir")
    return Pass::Ir;
  if (s == "combined")
    return Pass::Combined;
  return Pass::All;
}
} // namespace

int main(int argc, char* argv[])
{
  double slimValue = 1.0;
  bool hasSlim = true; // default full A2 size for slimmable models
  bool useFastTanh = true;
  double seconds = 2.0;
  Pass pass = Pass::All;
  std::vector<char*> positionalArgs;
  positionalArgs.push_back(argv[0]);

  for (int i = 1; i < argc; i++)
  {
    std::string arg(argv[i]);
    if (arg == "--seconds")
    {
      if (i + 1 >= argc)
      {
        std::cerr << "Error: --seconds requires a positive number\n";
        return 1;
      }
      seconds = std::strtod(argv[i + 1], nullptr);
      if (seconds <= 0.0)
      {
        std::cerr << "Error: --seconds must be positive\n";
        return 1;
      }
      i++;
    }
    else if (arg == "--slim")
    {
      if (i + 1 >= argc)
      {
        std::cerr << "Error: --slim requires a value between 0.0 and 1.0\n";
        return 1;
      }
      char* end = nullptr;
      slimValue = std::strtod(argv[i + 1], &end);
      if (end == argv[i + 1] || *end != '\0' || slimValue < 0.0 || slimValue > 1.0)
      {
        std::cerr << "Error: --slim value must be a number between 0.0 and 1.0\n";
        return 1;
      }
      hasSlim = true;
      i++;
    }
    else if (arg == "--no-slim")
    {
      hasSlim = false;
    }
    else if (arg == "--no-fast-tanh")
    {
      useFastTanh = false;
    }
    else if (arg == "--pass")
    {
      if (i + 1 >= argc)
      {
        std::cerr << "Error: --pass requires model|ir|combined|all\n";
        return 1;
      }
      pass = ParsePass(argv[i + 1]);
      if (std::string(argv[i + 1]) != "all" && std::string(argv[i + 1]) != "model" && std::string(argv[i + 1]) != "ir"
          && std::string(argv[i + 1]) != "combined")
      {
        std::cerr << "Error: --pass must be model|ir|combined|all\n";
        return 1;
      }
      i++;
    }
    else
    {
      positionalArgs.push_back(argv[i]);
    }
  }

  if (positionalArgs.size() < 3)
  {
    std::cerr << "Usage: bench_amp_ir [--slim <0.0-1.0>] [--no-slim] [--seconds N] [--pass model|ir|combined|all]\n"
              << "                    [--no-fast-tanh] <model_path> <ir_wav_path>\n";
    return 1;
  }

  const char* modelPath = positionalArgs[1];
  const char* irPath = positionalArgs[2];

  if (useFastTanh)
  {
    nam::activations::Activation::enable_fast_tanh();
    std::cout << "Fast tanh: enabled\n";
  }
  else
  {
    nam::activations::Activation::disable_fast_tanh();
    std::cout << "Fast tanh: disabled\n";
  }

  std::cout << "Loading model " << modelPath << "\n";
  std::unique_ptr<nam::DSP> model = nam::get_dsp(std::filesystem::path(modelPath));
  if (model == nullptr)
  {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  if (hasSlim)
  {
    auto* slimmable = dynamic_cast<nam::SlimmableModel*>(model.get());
    if (!slimmable)
    {
      std::cerr << "Warning: --slim ignored; model is not SlimmableModel\n";
    }
    else
    {
      std::cout << "Setting slimmable size to " << slimValue << "\n";
      slimmable->SetSlimmableSize(slimValue);
    }
  }

  const double sampleRate = model->GetExpectedSampleRate() > 0.0 ? model->GetExpectedSampleRate() : 48000.0;
  model->Reset(sampleRate, kBufferSize);

  std::cout << "Loading IR " << irPath << "\n";
  dsp::ImpulseResponse ir(irPath, sampleRate);
  if (ir.GetWavState() != dsp::wav::LoadReturnCode::SUCCESS)
  {
    std::cerr << "Failed to load IR: " << dsp::wav::GetMsgForLoadReturnCode(ir.GetWavState()) << "\n";
    return 1;
  }

  const auto irData = ir.GetData();
  const size_t rawLen = irData.mRawAudio.size();
  const double irRawSr = irData.mRawAudioSampleRate;
  size_t effectiveTaps = rawLen;
  if (std::abs(irRawSr - sampleRate) > 1e-3)
  {
    // ImpulseResponse resamples then truncates; approximate post-resample length.
    effectiveTaps = static_cast<size_t>(std::llround(static_cast<double>(rawLen) * sampleRate / irRawSr));
  }
  effectiveTaps = std::min(effectiveTaps, kIrMaxLength);

  std::cout << "IR raw samples: " << rawLen << " @ " << irRawSr << " Hz\n";
  std::cout << "IR effective taps (after resample/cap " << kIrMaxLength << "): " << effectiveTaps << "\n";
  {
    const char* eng = "unknown";
    switch (ir.GetActiveImplementation())
    {
      case dsp::ConvolutionImplementation::Direct: eng = "direct"; break;
      case dsp::ConvolutionImplementation::FFT: eng = "partitioned-fft"; break;
      case dsp::ConvolutionImplementation::Auto: eng = "auto"; break;
    }
    std::cout << "IR engine: " << eng << " (" << ir.GetNumTaps() << " taps)\n";
  }
  std::cout << "Plugin sample rate: " << sampleRate << " Hz, buffer: " << kBufferSize << "\n";
  std::cout << "Audio duration: " << seconds << " s\n";

  const int inChannels = model->NumInputChannels();
  const int outChannels = model->NumOutputChannels();
  if (inChannels < 1 || outChannels < 1)
  {
    std::cerr << "Error: model must have at least 1 in/out channel\n";
    return 1;
  }

  std::vector<std::vector<double>> inputBuffers(static_cast<size_t>(inChannels));
  std::vector<std::vector<double>> modelOutBuffers(static_cast<size_t>(outChannels));
  std::vector<double*> inputPtrs(static_cast<size_t>(inChannels));
  std::vector<double*> modelOutPtrs(static_cast<size_t>(outChannels));

  for (int ch = 0; ch < inChannels; ch++)
  {
    inputBuffers[static_cast<size_t>(ch)].assign(kBufferSize, 0.0);
    // Mild noise so IR/conv work on non-zero data (avoids denormal-only paths).
    for (size_t i = 0; i < kBufferSize; i++)
      inputBuffers[static_cast<size_t>(ch)][i] = 0.01 * std::sin(0.1 * static_cast<double>(i + ch));
    inputPtrs[static_cast<size_t>(ch)] = inputBuffers[static_cast<size_t>(ch)].data();
  }
  for (int ch = 0; ch < outChannels; ch++)
  {
    modelOutBuffers[static_cast<size_t>(ch)].assign(kBufferSize, 0.0);
    modelOutPtrs[static_cast<size_t>(ch)] = modelOutBuffers[static_cast<size_t>(ch)].data();
  }

  // IR path is mono in the plugin (kNumChannelsInternal = 1).
  std::vector<double> irIn(kBufferSize, 0.0);
  double* irInPtr = irIn.data();

  const size_t numBuffers = static_cast<size_t>((sampleRate / static_cast<double>(kBufferSize)) * seconds);

  // Warmup
  for (size_t i = 0; i < 8; i++)
  {
    model->process(inputPtrs.data(), modelOutPtrs.data(), kBufferSize);
    std::memcpy(irIn.data(), modelOutBuffers[0].data(), kBufferSize * sizeof(double));
    ir.Process(&irInPtr, 1, kBufferSize);
  }

  std::cout << "\n=== Timings (" << numBuffers << " blocks) ===\n";

  auto run_model = [&]() {
    auto t1 = high_resolution_clock::now();
    for (size_t i = 0; i < numBuffers; i++)
      model->process(inputPtrs.data(), modelOutPtrs.data(), kBufferSize);
    auto t2 = high_resolution_clock::now();
    return duration<double, std::milli>(t2 - t1).count();
  };

  auto run_ir = [&]() {
    // Feed the (already-filled) model-out-style buffer; refresh from input each block.
    auto t1 = high_resolution_clock::now();
    for (size_t i = 0; i < numBuffers; i++)
    {
      std::memcpy(irIn.data(), inputBuffers[0].data(), kBufferSize * sizeof(double));
      ir.Process(&irInPtr, 1, kBufferSize);
    }
    auto t2 = high_resolution_clock::now();
    return duration<double, std::milli>(t2 - t1).count();
  };

  auto run_combined = [&]() {
    auto t1 = high_resolution_clock::now();
    for (size_t i = 0; i < numBuffers; i++)
    {
      model->process(inputPtrs.data(), modelOutPtrs.data(), kBufferSize);
      std::memcpy(irIn.data(), modelOutBuffers[0].data(), kBufferSize * sizeof(double));
      ir.Process(&irInPtr, 1, kBufferSize);
    }
    auto t2 = high_resolution_clock::now();
    return duration<double, std::milli>(t2 - t1).count();
  };

  if (pass == Pass::All || pass == Pass::Model)
  {
    PrintTiming(TimingResult{"model-only", run_model(), numBuffers, sampleRate, kBufferSize});
  }
  if (pass == Pass::All || pass == Pass::Ir)
  {
    PrintTiming(TimingResult{"IR-only", run_ir(), numBuffers, sampleRate, kBufferSize});
  }
  if (pass == Pass::All || pass == Pass::Combined)
  {
    PrintTiming(TimingResult{"model+IR", run_combined(), numBuffers, sampleRate, kBufferSize});
  }

  return 0;
}
