// What does a call larger than the maximum buffer size cost, and does it match
// the same audio processed as consecutive calls of at most that size?
//
// Built to compare "grow the buffers to fit the call" (what process() used to
// do) against "split the call into reservation-sized pieces" (what it does
// now). The first allocates on the audio thread and resets every history, so
// the output stops matching what a host that respects the reservation gets.
//
// Usage: bench_oversized [--reserve N] [--call N] [--calls N] [--passes P] <model.nam>
//
// --reserve is the maximum buffer size handed to Reset(); --call is the size of
// every process() call. When --call <= --reserve the oversized path is never
// taken, which is the control: the two columns should be the same work.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

#include "test/allocation_tracking.h"

namespace
{

using clock_type = std::chrono::steady_clock;

struct Options
{
  int reserve = 32;
  int call = 100;
  int calls = 480; // Calls per pass; calls * call is the audio per pass.
  int passes = 60;
  double slimming = -1.0; // Sends SetSlimmableSize() before anything else when >= 0.
  std::string model = "example_models/A2.nam";
};

Options parse(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; i++)
  {
    const std::string arg = argv[i];
    const bool has_value = (i + 1) < argc;
    if (arg == "--reserve" && has_value)
      o.reserve = std::atoi(argv[++i]);
    else if (arg == "--call" && has_value)
      o.call = std::atoi(argv[++i]);
    else if (arg == "--calls" && has_value)
      o.calls = std::atoi(argv[++i]);
    else if (arg == "--passes" && has_value)
      o.passes = std::atoi(argv[++i]);
    else if (arg == "--slimming" && has_value)
      o.slimming = std::atof(argv[++i]);
    else if (arg == "--help")
    {
      std::cout << "Usage: bench_oversized [--reserve N] [--call N] [--calls N] [--passes P] <model.nam>\n";
      std::exit(0);
    }
    else
      o.model = arg;
  }
  return o;
}

std::unique_ptr<nam::DSP> load(const std::string& path, double slimming)
{
  nam::DspLoadOptions options;
  options.prewarm = false; // The timing loop is the warm-up; skip the load-time one.
  auto dsp = nam::get_dsp(std::filesystem::path(path), options);
  if (dsp == nullptr)
    throw std::runtime_error("could not load " + path);
  if (slimming >= 0.0)
  {
    // A container .nam (like the example A2) picks its submodel from this:
    // below the first breakpoint is A2 nano (3 channels), at the top the
    // standard model (8 channels).
    auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
    if (slimmable == nullptr)
      throw std::runtime_error("--slimming given but " + path + " is not slimmable");
    slimmable->SetSlimmableSize(slimming);
  }
  dsp->SetPrewarmOnReset(false);
  return dsp;
}

std::vector<NAM_SAMPLE> make_input(size_t frames, double sample_rate)
{
  std::vector<NAM_SAMPLE> in(frames);
  for (size_t i = 0; i < frames; i++)
  {
    const double t = static_cast<double>(i) / sample_rate;
    in[i] = static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * t) + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t));
  }
  return in;
}

void process_chunked(nam::DSP& dsp, NAM_SAMPLE* in, NAM_SAMPLE* out, int frames, int max_chunk)
{
  for (int offset = 0; offset < frames; offset += max_chunk)
  {
    const int n = std::min(max_chunk, frames - offset);
    NAM_SAMPLE* in_arr[1] = {in + offset};
    NAM_SAMPLE* out_arr[1] = {out + offset};
    dsp.process(in_arr, out_arr, n);
  }
}

void process_one(nam::DSP& dsp, NAM_SAMPLE* in, NAM_SAMPLE* out, int frames)
{
  NAM_SAMPLE* in_arr[1] = {in};
  NAM_SAMPLE* out_arr[1] = {out};
  dsp.process(in_arr, out_arr, frames);
}

/// One pass over the same `calls` calls, timed as a whole. Returns nanoseconds per call.
template <typename Process>
double time_one_pass(int calls, Process&& process)
{
  const auto start = clock_type::now();
  process();
  const auto elapsed = std::chrono::duration<double, std::nano>(clock_type::now() - start).count();
  return elapsed / calls;
}

double median(const std::vector<double>& sorted)
{
  if (sorted.empty())
    return 0.0;
  const size_t mid = sorted.size() / 2;
  return (sorted.size() % 2 == 1) ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

/// Allocations made while the oversized path runs, counted with tracking on.
/// Kept out of the timed passes so the counter cannot colour the timing.
uint64_t count_allocations(nam::DSP& dsp, NAM_SAMPLE* in, NAM_SAMPLE* out, int frames)
{
  allocation_tracking::g_allocation_count = 0;
  allocation_tracking::g_deallocation_count = 0;
  allocation_tracking::g_tracking_enabled = true;
  process_one(dsp, in, out, frames);
  allocation_tracking::g_tracking_enabled = false;
  return static_cast<uint64_t>(allocation_tracking::g_allocation_count);
}

} // namespace

int main(int argc, char** argv)
{
  const Options o = parse(argc, argv);
  if (o.reserve <= 0 || o.call <= 0 || o.calls <= 0 || o.passes <= 0)
  {
    std::cerr << "reserve, call, calls and passes must all be positive\n";
    return 1;
  }

  auto model = load(o.model, o.slimming);
  const double sample_rate = model->GetExpectedSampleRate() > 0.0 ? model->GetExpectedSampleRate() : 48000.0;
  const int frames = o.call * o.calls;
  const auto input = make_input(static_cast<size_t>(frames), sample_rate);
  std::vector<NAM_SAMPLE> out_reference(frames, 0.0);
  std::vector<NAM_SAMPLE> out_oversized(frames, 0.0);

  std::cout << "model:   " << o.model << "  (sample rate " << sample_rate << ")";
  if (o.slimming >= 0.0)
    std::cout << "  [SetSlimmableSize(" << o.slimming << ")]";
  std::cout << "\n"
            << "reserve: " << o.reserve << " frames (maximum buffer size)\n"
            << "call:    " << o.call << " frames, " << o.calls << " calls per pass, " << o.passes << " passes\n"
            << "\n";

  // --- Does an oversized call give what the host would get by chunking? -----
  {
    auto reference = load(o.model, o.slimming);
    auto oversized = load(o.model, o.slimming);
    reference->Reset(sample_rate, o.reserve);
    oversized->Reset(sample_rate, o.reserve);

    // Give both models real history first. A call that grows the buffers resets
    // every history, and a model that was just Reset() has none to lose -- so a
    // prologue of reservation-sized calls is what makes the loss visible.
    const int prologue = o.reserve * 16;
    {
      const auto warm = make_input(static_cast<size_t>(prologue), sample_rate);
      std::vector<NAM_SAMPLE> scratch(static_cast<size_t>(prologue), 0.0);
      process_chunked(*reference, const_cast<NAM_SAMPLE*>(warm.data()), scratch.data(), prologue, o.reserve);
      process_chunked(*oversized, const_cast<NAM_SAMPLE*>(warm.data()), scratch.data(), prologue, o.reserve);
    }

    uint64_t allocations = 0;
    double max_diff = 0.0;
    long first_diff = -1;
    for (int call = 0; call < o.calls; call++)
    {
      NAM_SAMPLE* in = const_cast<NAM_SAMPLE*>(input.data()) + static_cast<size_t>(call) * o.call;
      NAM_SAMPLE* ref = out_reference.data() + static_cast<size_t>(call) * o.call;
      NAM_SAMPLE* big = out_oversized.data() + static_cast<size_t>(call) * o.call;
      process_chunked(*reference, in, ref, o.call, o.reserve);
      allocations += count_allocations(*oversized, in, big, o.call);
      for (int f = 0; f < o.call; f++)
      {
        const double diff = std::fabs(static_cast<double>(big[f]) - static_cast<double>(ref[f]));
        if (diff > max_diff)
          max_diff = diff;
        if (diff != 0.0 && first_diff < 0)
          first_diff = call * o.call + f;
      }
    }

    std::cout << "parity (oversized call vs the same audio as consecutive " << o.reserve << "-frame calls)\n"
              << "  bit-identical:  " << (max_diff == 0.0 ? "yes" : "no") << "\n";
    if (max_diff != 0.0)
      std::cout << "  max |diff|:     " << std::setprecision(6) << max_diff << "\n"
                << "  first sample:   " << first_diff << "\n";
    std::cout << "  allocations:    " << allocations << " over " << o.calls << " calls\n\n";
  }

  // --- What does each way of handling the same audio cost? ------------------
  auto reference = load(o.model, o.slimming);
  auto oversized = load(o.model, o.slimming);
  reference->Reset(sample_rate, o.reserve);
  oversized->Reset(sample_rate, o.reserve);

  const auto chunked_pass = [&]() {
    for (int call = 0; call < o.calls; call++)
      process_chunked(*reference, const_cast<NAM_SAMPLE*>(input.data()) + static_cast<size_t>(call) * o.call,
                      out_reference.data() + static_cast<size_t>(call) * o.call, o.call, o.reserve);
  };
  const auto oversized_pass = [&]() {
    for (int call = 0; call < o.calls; call++)
      process_one(*oversized, const_cast<NAM_SAMPLE*>(input.data()) + static_cast<size_t>(call) * o.call,
                  out_oversized.data() + static_cast<size_t>(call) * o.call, o.call);
  };

  chunked_pass(); // Warm up both models before anything is timed.
  oversized_pass();

  // Interleaved, so a change in the machine's state between the two scenarios
  // cannot be mistaken for a difference between them.
  std::vector<double> chunked;
  std::vector<double> big;
  for (int pass = 0; pass < o.passes; pass++)
  {
    chunked.push_back(time_one_pass(o.calls, chunked_pass));
    big.push_back(time_one_pass(o.calls, oversized_pass));
  }
  std::sort(chunked.begin(), chunked.end());
  std::sort(big.begin(), big.end());

  const auto print = [](const char* name, const std::vector<double>& per_call) {
    std::cout << "  " << std::left << std::setw(15) << name << std::right << std::fixed << std::setprecision(2)
              << "best " << std::setw(9) << per_call.front() / 1000.0 << " us/call"
              << "   median " << std::setw(9) << median(per_call) / 1000.0 << " us/call\n";
  };
  std::cout << "time per " << o.call << "-frame call\n";
  print("host-chunked:", chunked);
  print("oversized:", big);
  std::cout << "  ratio:          " << std::setprecision(3) << (big.front() / chunked.front()) << "x of host-chunked\n";

  return 0;
}
