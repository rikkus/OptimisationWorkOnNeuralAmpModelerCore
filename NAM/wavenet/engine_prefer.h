#pragma once

// Force WaveNet engine selection for A/B benches and diagnostics.
// Thread-local so SlimmableContainer nested get_dsp() calls inherit the preference.

namespace nam
{
namespace wavenet
{

enum class EnginePrefer
{
  Auto = 0, // fused → a2_fast → generic (production default)
  Generic, // Eigen / generic WaveNet only
  FusedNeon // fused engine, NEON conv kernels
};

void set_engine_prefer(EnginePrefer prefer);
EnginePrefer get_engine_prefer();

struct ScopedEnginePrefer
{
  explicit ScopedEnginePrefer(EnginePrefer prefer)
  : previous(get_engine_prefer())
  {
    set_engine_prefer(prefer);
  }
  ~ScopedEnginePrefer() { set_engine_prefer(previous); }
  ScopedEnginePrefer(const ScopedEnginePrefer&) = delete;
  ScopedEnginePrefer& operator=(const ScopedEnginePrefer&) = delete;
  EnginePrefer previous;
};

} // namespace wavenet
} // namespace nam
