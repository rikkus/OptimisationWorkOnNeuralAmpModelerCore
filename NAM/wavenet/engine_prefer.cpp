#include "engine_prefer.h"

namespace nam
{
namespace wavenet
{
namespace
{
thread_local EnginePrefer g_engine_prefer = EnginePrefer::Auto;
}

void set_engine_prefer(EnginePrefer prefer)
{
  g_engine_prefer = prefer;
}

EnginePrefer get_engine_prefer()
{
  return g_engine_prefer;
}

} // namespace wavenet
} // namespace nam
