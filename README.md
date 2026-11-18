# RDNA3.5 WMMA arithmetic and standalone validation

This directory is self-contained. [rdna35_wmma.hip](rdna35_wmma.hip) contains
CPU and HIP/CUDA GPU software emulators, a separate FP64 reference, real WMMA
hardware oracles, input generators, stress tests, failure replay, and JSON
reporting. No Python, PyTorch, rocBLAS, or other repository code is needed.
Both BF16→FP32 and FP16→FP32 are implemented and tested.

The arithmetic below specifies the empirical model measured on RDNA3.5
`gfx1151`, for wave32 `v_wmma_f32_16x16x16_bf16` and
`v_wmma_f32_16x16x16_f16`. All outputs are compared as raw FP32 bits by
default, including zero/infinity signs and NaN payloads/signs. The measured
`gfx1151` NaN selection rules are modeled explicitly. Floating-point exception
flags are outside the model. `--nan-classification` opts into a relaxed
comparison that ignores payload/sign only when both outputs are NaN.

This is an exact definition of the implemented model, supported by hardware
comparisons. It is not an AMD architectural guarantee, a description of the
physical datapath, or a proof over every possible input on every RDNA3.5 device.
Only `gfx1151` has been directly validated here.

## 1. Instruction shape and accumulation order

For `A[16][16]`, `B[16][16]`, and FP32 `C[16][16]`, each result element is:

```text
c0 = C[m,n]
c1 = dot2(A[m,0],  B[0,n],  A[m,1],  B[1,n],  c0)
c2 = dot2(A[m,2],  B[2,n],  A[m,3],  B[3,n],  c1)
...
c8 = dot2(A[m,14], B[14,n], A[m,15], B[15,n], c7)
D[m,n] = c8
```

Each dot2 emits an FP32 result before the next pair is processed. The model
therefore has eight rounding points, in increasing `k` order. It is not an
exact 16-term sum followed by one rounding, a generic matrix multiplication,
or a sequence of ordinary FP32 multiply/add operations.

The standalone APIs store A row-major, B column-major, and C/D row-major:

```text
A[m,k] = a[m*16+k]
B[k,n] = bt[n*16+k]
C[m,n] = c[m*16+n]
D[m,n] = d[m*16+n]
```

A/B use raw `uint16_t` encodings; C/D use raw FP32 `uint32_t` encodings.
No conversion to a host floating-point type is required by the integer emulator.
An output buffer must be distinct from the input buffers.

## 2. BF16 inputs: decode and input flushing

BF16 has one sign bit `s`, eight exponent bits `h`, and seven fraction bits `f`.
Its exponent bias is 127.

| Encoding | Value used by the operation | Alignment exponent `e` |
|---|---|---|
| `1 <= h <= 254` | `(-1)^s * (128+f) * 2^(h-127-7)` | `h-127` |
| `h = 0`, any `f` | Signed zero, preserving `s` | `-126` |
| `h = 255, f = 0` | Signed infinity | Not used for finite alignment |
| `h = 255, f != 0` | NaN, quiet or signaling | Not used for finite alignment |

Thus **all BF16 input subnormals are flushed to signed zero** before
multiplication. Their signs still matter in the correction described below.
For example, negative BF16 minimum subnormal `0x8001` becomes negative zero,
not positive zero.

After decoding/flushing, finite products are formed exactly. They are not
rounded back to BF16 or FP32. In particular, a finite product outside the FP32
range is not immediately replaced with infinity: alignment, cancellation, and
the signed correction occur before the step's output conversion.

## 3. FP16 inputs: decode and subnormal retention

FP16 has one sign bit `s`, five exponent bits `h`, and ten fraction bits `f`.
Its exponent bias is 15.

| Encoding | Value used by the operation | Alignment exponent `e` |
|---|---|---|
| `1 <= h <= 30` | `(-1)^s * (1024+f) * 2^(h-15-10)` | `h-15` |
| `h = 0` | `(-1)^s * f * 2^-24`, including signed zero | **`-14`** |
| `h = 31, f = 0` | Signed infinity | Not used for finite alignment |
| `h = 31, f != 0` | NaN, quiet or signaling | Not used for finite alignment |

Unlike BF16, **FP16 subnormals are retained**. Their alignment exponent stays
at `-14`; it is not recomputed by normalizing their nonzero significand. This
can change the truncation and signed correction considerably. For instance,
`dot2(-2^-24, 1, +0, +0, +0)` produces `-(2^-24 + 2^-38)`.

Finite FP16 products are also exact before alignment. Apart from input decoding
and the subnormal rule, the BF16 and FP16 operations use the same recurrence.

## 4. FP32 accumulator decoding

For an FP32 accumulator with sign `s`, exponent field `h`, and fraction `f`:

- Normal finite values use significand `2^23+f` and exponent `ec=h-127`.
- Zero and every FP32 subnormal become signed zero, preserving `s`, and use
  `ec=-126`.
- Exponent field 255 denotes infinity or NaN in the usual way.

The FP32 accumulator is decoded/flushed at **every** dot2 step, including the
initial C input. Input flushing to signed zero differs from output underflow,
which emits **positive** zero.

## 5. Common finite dot2 arithmetic

Let the decoded products be `p0=a0*b0` and `p1=a1*b1`. Define their alignment
exponents from the **input** exponents:

```text
ep0 = ea0 + eb0
ep1 = ea1 + eb1
```

Do not renormalize these exponents when the product significand is at least 2,
or when an FP16 operand is subnormal. Exclude a zero product from the maximum;
its sign will still participate in the correction.

```text
E = max(ec - 2, ep0 if p0 != 0, ep1 if p1 != 0)
q = 2^(E - 24)
```

If both products are zero, the maximum is simply `ec-2`.
Align and truncate **toward zero**, separately for each term:

```text
u0 = trunc(p0 / q)
u1 = trunc(p1 / q)
uc = trunc(c  / q)
```

Then apply the signed correction and add exactly:

```text
S = u0 + u1 + uc - signbit(p0) - signbit(p1) + 2*signbit(c)
x = S*q
```

`signbit` is 0 or 1 and is also evaluated for signed zeros. A product's sign is
the XOR of its operand signs. Consequently `-0` products contribute `-1` even
though they do not set `E`, and a `-0` accumulator contributes `+2`.
The constant 24 specifies fractional bits relative to the **unnormalized**
alignment exponent; it is not a claim that every intermediate is an ordinary
FP32 value with 24 significant bits.

### Output underflow, rounding, and overflow

Let the cutoff be:

```text
T = 2^-126 - 2^-151 = (2^25 - 1) * 2^-151
```

1. If `abs(x) < T`, return raw FP32 `0x00000000`, regardless of the sign of x.
2. Otherwise round x to FP32 with round to nearest, ties to even, carrying into
   the exponent when necessary.
3. Overflow of that output conversion produces signed infinity.

The strict inequality is intentional: **at the cutoff**, the result is the
signed minimum normal FP32 value. The cutoff is one quarter of one FP32
subnormal ULP below the minimum normal. This differs from both flushing every
exact value below `2^-126` and rounding to FP32 before flushing subnormals.
The halfway value `2^-126 - 2^-150` flushes to +0.
An exact zero result is +0.

The standalone integer code compares with T exactly, then explicitly encodes
normal FP32 significand rounding, ties-to-even, exponent carry, and infinity.
It uses no host/device floating-point conversion for this path. Alignment
significands and their sum fit in fewer than 29 magnitude bits; `int64_t` is
ample. Even very large finite BF16 products need only their integer
significands and exponents, not an FP32 intermediate.

### Nonfinite inputs and exact NaN selection

At **each dot2 step**, after the input flushing rules, use this precedence:

1. If any of the four 16-bit operands is NaN, return raw `0xffc00a3d`,
   independently of its sign/payload and of C. This wins over an invalid
   multiplication in the other product of the same pair.
2. If either product is infinity times zero, or the products/accumulator include
   both positive and negative infinities, return raw `0xffc00000`. This also
   wins over a NaN C. BF16 subnormals flushed to zero count as zero here.
3. If C is NaN, preserve its sign/payload and set the quiet bit:
   `c_bits | 0x00400000`. Thus a signaling NaN becomes quiet.
4. Otherwise, any infinity determines the result with its sign.
5. Otherwise, use the finite alignment, correction, and rounding rules above.

These are measured `gfx1151` output encodings, not a claim about unspecified
NaN propagation on every AMD architecture. Apply them per pair, not once to the
whole 16-term operation: an invalid product in a later pair can replace an
existing accumulator NaN with `0xffc00000`, and a later 16-bit NaN can replace it
with `0xffc00a3d`. Finite products outside the FP32 range are not themselves
classified as infinities before alignment/cancellation.

## 6. Distinguishing examples

These are single dot2 steps. Inputs in the first four rows are representable
in both BF16 and FP16.

| `(a0,b0,a1,b1,c)` | Result |
|---|---|
| `(1,1,0,0,-1)` | `2^-23` |
| `(-1,1,0,0,1)` | `-2^-24` |
| `(1,1,-1,1,0)` | `-2^-24` |
| `(1,1,2^-12,2^-12,-1)` | `3*2^-24` |
| BF16 `(-2^-133,1,1,1,0)` | `1-2^-24`, because the first product is negative zero |
| FP16 `(-2^-24,1,0,0,0)` | `-(2^-24+2^-38)` |
| BF16 `(2^-100,2^-51,-0,1,-2^-126)` | `+0` |
| BF16 `(2^-100,2^-52,-0,1,-2^-126)` | `-2^-126` |

The last two cases distinguish the measured underflow cutoff. A directed sweep
found this distinction after more than 220 million broader output checks had
passed. The program retains these cases and the finer sweep permanently.

## 7. Build and run

For the GPU build, install ROCm with `hipcc`. No Python environment is involved:

```sh
cd standalone/rdna35_wmma
make gpu ARCH=gfx1151
./rdna35_wmma --trials 1000 --report validation.json
```

The default HIP run checks, for both input formats:

- CPU integer emulation against the separate CPU FP64 reference.
- CPU integer emulation against actual WMMA hardware.
- GPU integer emulation against actual WMMA hardware.
- Two native dot2 replacements against actual WMMA hardware, bit for bit.

The integer GPU emulator contains **no WMMA or dot-product intrinsic**. The
separate `hardware_wmma` oracle and WMMA benchmark invoke WMMA; the two native
dot2 kernels explicitly invoke dot2 instructions. Both GPU paths run on a nonblocking
HIP stream, with explicit error checks and synchronization before comparison.
The hardware oracle uses one 32-thread block per tile: lanes 0–15 and 16–31
replicate the input fragments; each lane's eight outputs occupy rows
`lane/16 + 2*i` of column `lane%16`.

### NVIDIA / direct CUDA build

The same `.hip` source builds directly with `nvcc`; the NVIDIA machine needs
only the CUDA toolkit, not HIP or ROCm:

```sh
make cuda
./rdna35_wmma_cuda --trials 100000 --seed 981713 --report cuda-validation.json
./rdna35_wmma_cuda --benchmark-only --bench-inner 128 --report cuda-perf.json
```

Set `CUDA_ARCH` explicitly if needed, for example `make cuda CUDA_ARCH=sm_80`;
otherwise nvcc uses its default architecture. `NVCC`, `CUDAFLAGS`, and
`CUDA_OUT` can also be overridden. A direct build without Make is:

```sh
nvcc -O2 -std=c++17 -Xcompiler=-pthread,-fno-fast-math,-ffp-contract=off \
  -x cu rdna35_wmma.hip -o /tmp/wmma-check-cuda
```

CUDA runs the same integer RDNA arithmetic, including BF16/FP16 decoding and
RDNA NaN encodings. No native dot2 or tensor-core instruction is required.
The default compares GPU output bit for bit against CPU integer emulation and
checks the CPU model against the FP64 reference. All stress suites, strict NaN
checks, and failure replay remain available; `--no-reference` omits only the
FP64 cross-check. This is emulation of RDNA3.5, not of NVIDIA MMA arithmetic.

AMD WMMA/dot2 kernels are excluded from CUDA compilation. A CUDA run reports
its NVIDIA device architecture as `sm_XX` and `hardware_verified: false`: no
real RDNA instruction was checked. CUDA benchmarks only `integer_emulation`
in the two existing timing modes, with `relative_to_wmma: null` and no WMMA
speed ratio. HIP/CUDA runtime names use preprocessor aliases; there are no
additional runtime wrapper functions. CUDA compilation and execution require
validation on the NVIDIA machine; nvcc is not installed on the development host.

For a genuinely CPU-only build, including on a machine without ROCm:

```sh
make cpu
./rdna35_wmma_cpu --trials 1000 --report cpu-validation.json
```

This uses a C++17 compiler and the standard library, with GPU code removed by
`WMMA_CPU_ONLY`. It checks the integer emulator against the FP64 reference and
fixed golden results. Its report explicitly says hardware was not checked.
A HIP- or CUDA-built executable can also run with `--cpu-only` without querying
a GPU, although it still needs its linked GPU runtime installed.

To build the single source directly:

```sh
hipcc -O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off \
  --offload-arch=gfx1151 -mno-wavefrontsize64 rdna35_wmma.hip -o /tmp/wmma-check
c++ -O2 -std=c++17 -pthread -fno-fast-math -ffp-contract=off \
  -DWMMA_CPU_ONLY -x c++ rdna35_wmma.hip -o /tmp/wmma-check-cpu
```

The Makefile additionally embeds a source SHA-256 in each executable/report.
Do not enable fast-math for the independent FP64 reference. Its host rounding
mode is set to nearest-even. The integer emulators are independent of that mode.

### Stress and directed coverage

`--trials N` means N random **16x16 tiles per family per dtype**. The default
is 1000; the larger validation command is:

```sh
./rdna35_wmma --trials 100000 --seed 492031 \
  --report validation-large.json --failure failing-tile.txt
```

Every invocation also runs the directed suites; `--trials 0` runs just those.
`--dtype bf16` or `--dtype f16` restricts the matrix suites.
`--batch` controls GPU batch size; `--threads` controls CPU workers.
`--no-reference` skips the extra FP64 cross-check in GPU runs, but CPU and GPU
integer results are still compared separately against hardware on AMD. CUDA
always compares GPU output against the CPU integer model.

| Suite | Coverage |
|---|---|
| Golden cases | Fixed expected cancellation, signed-zero, subnormal, and cutoff results |
| Lane mapping | All k selections and output lanes, against independently known exact results |
| `bounded` | Finite values with random signs, significands, and moderate exponents |
| `bits` | Uniform raw 16-bit operands and raw 32-bit accumulators |
| `cancellation` | Opposite products in every pair, with zero C |
| `exponents` | Products and accumulators concentrated around alignment, underflow, and overflow boundaries |
| `special_values` | All 485,537 combinations of 13 operand representatives and 17 accumulator representatives, per dtype |
| `normal_boundary` | 353,736 BF16 and 78,336 FP16 directed small-product/normal-boundary cases |
| `all_encodings_role_0..3` | Every one of the 65,536 input encodings in each dot2 multiplicand role, per dtype |

Directed scalar cases are packed onto matrix diagonals; all off-diagonal
cross-combinations are compared too. Random draws use specified SplitMix64
integer arithmetic with explicitly sequenced draws, so seeds produce the same
corpus in the GCC CPU build and the Clang HIP build. Random family seeds are
reported separately. Data generation does not require floating-point randomness.

### Failure handling and replay

A mismatch fails the process, prints the format, suite, tile, output coordinate,
and CPU/reference/GPU/hardware raw bits, and writes all A, B, and C encodings
for the failing tile. Replay it without regenerating any random inputs:

```sh
./rdna35_wmma --replay failing-tile.txt --report replay.json
```

The default comparison requires exact bits, **including NaNs**; `--strict-nan`
explicitly selects the same default. `--nan-classification` is an opt-in relaxed
mode that ignores payload/sign only when both outputs are NaN. It never discards
a finite-vs-NaN mismatch. The strict setting also applies to failure replay.

Exit status is 0 for success, 1 for validation failure, and 2 for argument or
initialization errors. A failed report has `passed: false`; previously completed
suites may still show zero mismatches. The failure's details are in `error`.

## 8. Performance: WMMA versus eight native dot2 steps

Run the GPU benchmark after the stress suites, or on its own:

```sh
./rdna35_wmma --trials 100000 --seed 981713 --benchmark \
  --bench-inner 128 --bench-repeats 10 --bench-samples 9 --report strict-and-perf.json
./rdna35_wmma --benchmark-only --bench-inner 128 --report perf.json
```

On AMD, four implementations perform exactly the same complete tile operation
(CUDA offers only the integer emulator):

| Backend | Work distribution |
|---|---|
| `wmma` | One wave32 WMMA instruction produces 256 outputs |
| `native_dot2_per_output` | Eight waves per tile, one output/lane, eight dependent dot2 steps/output |
| `native_dot2_per_wave` | One wave per tile, eight outputs/lane, A rows gathered with shuffles before the recurrence |
| `integer_emulation` | The portable arithmetic emulator, one output/thread |

Both native dot2 implementations execute 64 **wave-wide** dot2 instructions per
complete tile: either eight steps in each of eight waves, or 64 steps in one
wave with eight accumulators. Eight dot2 instructions in a single wave produce
only 32 outputs; WMMA produces 256. Instruction counts are not cycle counts:
WMMA may itself take multiple cycles, and GPU scheduling can overlap work.

The numerical eight-pair recurrence constrains each output's rounding order;
it does not serialize all outputs or prescribe WMMA's physical implementation.
WMMA packages the outputs with hardware operand routing/reuse and a much more
compact instruction stream. Native dot2 replacements need additional lanes or
instructions, operand gathering, and registers. There is no general promise of
a performance-free replacement. A memory/launch-limited workload can nevertheless
show nearly identical kernel time, because arithmetic is not its bottleneck.

The program measures two modes:

- `tile_kernel`: one tile operation per kernel invocation, including input
  loads, routing, and output stores.
- `register_resident`: `--bench-inner` dependent tile recurrences per invocation,
  with inputs loaded/gathered before the inner loop. This amortizes setup and
  memory costs; it is not a pure single-instruction latency measurement.

Data are finite positive values and every backend's final result is checked
against CPU integer emulation **before** timing. Three warmup launches precede
measurements. HIP/CUDA events measure GPU stream execution over multiple launches;
CPU reference work and host/device transfers are excluded. Backend order rotates
between samples. JSON records median/min/max microseconds per launch, effective
matrix TFLOP/s (`8192*tiles*iterations / time`), and time relative to WMMA on AMD.
The integer path's effective TFLOP/s is a workload metric, not its physical
instruction throughput. `--bench-tiles`, `--bench-inner`, `--bench-repeats`, and
`--bench-samples` control the experiment.

Results apply to these kernels, launch sizes, device, and sampled clock/load
state. They do not establish optimal native-dot2 performance or GEMM performance
in an application. Run without another GPU workload when comparing timings.
Benchmark-only reports distinguish benchmark-verified output positions from
stress-suite positions; they do not claim the stress suites ran.

The recorded run used 2,048 tiles/launch, 128 resident recurrences, 10 launches
per sample, and nine samples on an AMD Radeon 8060S (`gfx1151`). Times below
are medians per launch, with no other task using the GPU during measurement:

| Implementation | BF16 tile pass (µs) | BF16 resident (µs) | FP16 resident (µs) |
|---|---:|---:|---:|
| WMMA | 11.18 | 58.06 | 58.89 |
| Native dot2, output/lane | 11.54 | 245.65 | 250.04 |
| Native dot2, eight outputs/lane | 11.01 | 133.17 | 135.68 |
| Integer emulator | 201.09 | 22,666.90 | 24,015.20 |

Both native replacements were close to WMMA in the single-pass experiment.
In the resident experiment, the cooperative native replacement took about
2.3× WMMA time and the per-output replacement about 4.2×, for both formats.
These measurements show why numerical equivalence does not imply performance
equivalence; they do not rule out further tuning of the replacements.
