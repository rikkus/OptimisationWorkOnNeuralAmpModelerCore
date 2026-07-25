# Cab FIR partitioned FFT — before/after

Date:       2026-07-14 21:46 UTC  
Machine:    Apple M2 (arm64)  
Core:       `09d46b0`  
Plugin:     `9451ecc` (+ local AudioDSPTools partitioned-FFT changes)  
Build:      Release, fused + a2_fast ON  
Workload:   `ORNG-V30-e609-Center.nam --slim 1.0` + `YA AMPG 410 HS 52-CNT.wav`  
            48 kHz, buffer 64, IR capped at **8192** taps  

## Change

Replaced per-sample Eigen `dot` FIR in `dsp::ImpulseResponse` with zero-latency
**partitioned FFT** (`dsp::PartitionedConvolution`, NAM `Linear`-style
`Eigen::FFT<float>`). Auto: direct if taps ≤ 256, else FFT.

## Wall timings (2 s audio, median of 3 warm runs)

| Pass | Before (direct) | After (FFT) | Speedup |
|---|---:|---:|---:|
| model-only | 40.3 ms (49.6×) | 40.4 ms (49.5×) | ~1.0× |
| IR-only | 59.1 ms (33.9×) | **5.2 ms (386×)** | **~11×** |
| model+IR | 100.6 ms (19.9×) | **45.7 ms (43.7×)** | **~2.2×** |

Combined real-time factor moves from ~20× toward the model-only ~50× floor.
IR is no longer the dominant cost.

## Instruments flat profile (combined, 10 s `sample`)

Before ([`sample_orng_amp_ir_20260714.txt`](sample_orng_amp_ir_20260714.txt)):

| Symbol | Share |
|---|---:|
| `ImpulseResponse::Process` (direct FIR) | **58.4%** |
| `fused::conv_block<2>` | 31.3% |

After ([`sample_orng_amp_ir_fft_20260714.txt`](sample_orng_amp_ir_fft_20260714.txt)):

| Symbol | Samples (top-of-stack) |
|---|---:|
| `fused::conv_block<2>` | **3952** (dominant) |
| `FusedWaveNet::process` (other) | 580 |
| `fused::tail_block<2>` | 553 |
| `PartitionedConvolution::_ProcessFft` | 491 |
| Eigen kiss FFT helpers | ~166 |

Cab convolution is a small fraction of amp+cab again; fused WaveNet is the main cost.

## Correctness

`test_ir_convolution` in `run_tests` (from `NeuralAmpModelerCore/`):

- Auto selects Direct (≤256 taps) / FFT (>256)
- Direct vs FFT parity for synthetic IRs (64 / 512 / 8192 taps; blocks 16 / 23 / 64 / 256)
- `ImpulseResponse` wrapper parity + YA AMPG file parity
- Zero allocations in FFT `Process` after warm-up

## Reproduce

```sh
cd NeuralAmpModelerCore
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_amp_ir run_tests -j$(sysctl -n hw.ncpu)
./build/tools/run_tests
./build/tools/bench_amp_ir --slim 1.0 --seconds 2 \
  example_models/ORNG-V30-e609-Center.nam \
  "example_models/YA AMPG 410 HS 52-CNT.wav"
```
