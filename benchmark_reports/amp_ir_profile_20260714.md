# Amp + IR profile — ORNG full A2 + YA AMPG cab

Date:       2026-07-14 21:39 UTC  
Machine:    Apple M2 (arm64), macOS Darwin 27.0.0  
Core:       `09d46b0` (fused engine)  
Plugin:     `9451ecc`  
Build:      Release (`NAM_ENABLE_FUSED` + `NAM_ENABLE_A2_FAST` ON)  
Tool:       `tools/bench_amp_ir`

## Workload

| Item | Value |
|---|---|
| Model | `example_models/ORNG-V30-e609-Center.nam` |
| Slim | **1.0** (full 8-ch A2-standard submodel; not nano) |
| IR | `example_models/YA AMPG 410 HS 52-CNT.wav` |
| IR raw | 24000 samples @ 48 kHz (0.5 s) |
| IR used | **8192 taps** (`ImpulseResponse::mMaxLength` cap) |
| Rate / buffer | 48 kHz / 64 frames (deadline 1333.3 µs/block) |
| Note | ORNG has **no embedded IR**; cab is the separate WAV |

Engine path (from Instruments symbols): **`fused::FusedWaveNet`** (`conv_block<2>` → 8 channels).

## Wall timings (2 s audio, 1500 blocks)

| Pass | Wall | RTF | µs/block | % of deadline |
|---|---:|---:|---:|---:|
| model-only | 40.3 ms | 49.6× | 26.9 | 2.0% |
| IR-only | 59.1 ms | 33.9× | 39.4 | 3.0% |
| **model+IR** | **100.6 ms** | **19.9×** | **67.1** | **5.0%** |

Model-only matches prior fused A2-standard benches (~40 ms / 2 s). IR alone is **~1.5×** the fused amp. Combined RTF drops from ~50× to ~20× — amp and cab are additive (no interaction).

## Instruments flat profile (combined path, 10 s `sample`)

Raw log: [`sample_orng_amp_ir_20260714.txt`](sample_orng_amp_ir_20260714.txt)

Sort-by-top-of-stack (8486 main-thread samples during `model+IR`):

| Symbol | Samples | Share |
|---|---:|---:|
| `dsp::ImpulseResponse::Process` | 4953 | **58.4%** |
| `fused::conv_block<2>` | 2658 | 31.3% |
| `FusedWaveNet::process` (other) | 375 | 4.4% |
| `fused::tail_block<2>` | 339 | 4.0% |
| `_platform_memmove` | 145 | 1.7% |
| other | ~16 | ~0.2% |

Cab IR (direct Eigen `dot` per sample × 8192 taps) is the largest single cost in a realistic amp+cab session after the fused WaveNet win.

## Conclusion

1. Full A2 (ORNG @ slim 1.0) is on the fused NEON path and is already cheap vs the RT deadline.
2. With a typical long cab IR (capped at 8192 taps), **IR dominates (~58% of combined CPU)** and roughly matches or exceeds amp cost in wall time.
3. Highest-ROI next CPU target for amp+cab sessions: replace direct FIR in `ImpulseResponse` with partitioned FFT / Accelerate `vDSP` (same idea as NAM `Linear`), not further WaveNet silicon work.

## Reproduce

```sh
cd NeuralAmpModelerCore
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target bench_amp_ir -j$(sysctl -n hw.ncpu)

./tools/bench_amp_ir --slim 1.0 --seconds 2 \
  ../example_models/ORNG-V30-e609-Center.nam \
  "../example_models/YA AMPG 410 HS 52-CNT.wav"

# Flat profile (~10 s wall): long enough audio that sample can attach
./tools/bench_amp_ir --slim 1.0 --seconds 300 --pass combined \
  ../example_models/ORNG-V30-e609-Center.nam \
  "../example_models/YA AMPG 410 HS 52-CNT.wav" &
sample $! 10 -file /tmp/bench_amp_ir_sample.txt
```
