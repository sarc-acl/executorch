## Purpose

Define how an ExecuTorch prefill result is compared with another inference runtime on the same GPU, so that the
comparison measures the same work, can be repeated by someone else, and states every way in which the two sides
differ.

## ADDED Requirements

### Requirement: One workload for every arm

Every arm on a device SHALL process the same prompt text, tokenized to the same number of tokens, on the same
model weights revision, with a context length equal to that of the ExecuTorch model being compared. The
reported quantity SHALL be prefill tokens per second: prompt tokens divided by the time the runtime spends
evaluating the prompt, excluding model load, pipeline or shader compilation, tokenization and any warm-up
pass.

#### Scenario: Token counts agree

- **WHEN** a run of any arm reports a prompt token count
- **THEN** it equals the count of the ExecuTorch arm, apart from a beginning-of-sequence token whose presence is
  recorded per arm, and a run with any other count is invalid

#### Scenario: Warm-up is excluded

- **WHEN** a run is taken
- **THEN** the process first performs a warm-up pass that is not timed, and the reported time covers prompt
  evaluation only

### Requirement: Weights come from one source

The weights of every arm SHALL be derived from the same upstream model revision, and the derivation of each
model file (source revision, tool version, exact command) SHALL be recorded beside the results.

#### Scenario: A model file without provenance

- **WHEN** a model file has no recorded source revision and conversion command
- **THEN** no result measured with it is reported

### Requirement: Arms are fixed before measuring

The set of arms per device SHALL be written down before the first timed run on that device: the ExecuTorch
builds with their commits and selection settings, and for the other runtime its commit, backend, quantization
and settings tier. Results SHALL be reported for every arm that was written down.

#### Scenario: An arm turns out slower or faster than expected

- **WHEN** a written-down arm has been measured validly
- **THEN** its result appears in the results table regardless of its value

### Requirement: The other runtime is measured at its best as well as at its default

For the other runtime, each backend and quantization SHALL be measured in three settings tiers: out of the box,
aligned to the ExecuTorch workload (the whole prompt in one batch), and the best documented settings. The
settings of the third tier SHALL be chosen by a screen that is recorded, and fixed before the timed session.

#### Scenario: The headline comparison

- **WHEN** a single number per runtime is quoted for a device
- **THEN** the other runtime's number is its best valid tier, and the tier is named

### Requirement: Versions are pinned and builds are recorded

Every arm SHALL be built from a recorded commit with recorded build options, and the other runtime SHALL be at
the same commit on every device.

#### Scenario: A build cannot be reproduced

- **WHEN** an arm's commit or build options are not recorded
- **THEN** its results are not reported

### Requirement: Runs are taken on an exclusive device and judged by fixed rules

A device SHALL run nothing else on its GPU while it is measured. Each cell SHALL be the median of at least five
valid runs, each in a fresh process, with the arms interleaved within a session and clock and temperature
sampled during each run. The rules that make a run invalid SHALL be fixed before the session.

#### Scenario: A foreign GPU process appears

- **WHEN** another process uses the GPU during a run
- **THEN** that run is invalid and is taken again

#### Scenario: Throttled run

- **WHEN** a run's clock or temperature record violates the device's validity rule
- **THEN** that run is invalid and is taken again, and the number of invalid runs is reported

### Requirement: Cells that cannot be run are reported as such

When an arm cannot run a model or scheme on a device, the cell SHALL say so with the reason. A result of a
different configuration SHALL NOT be put in its place.

#### Scenario: A backend does not support a quantization scheme

- **WHEN** a backend cannot execute a scheme natively
- **THEN** the cell reads "not supported" with the reason, and a run that silently computes a different scheme
  is not reported under that scheme's name

### Requirement: Differences between the runtimes are disclosed with the numbers

Every results table SHALL be accompanied by the known differences between the arms: quantization format and
bits per weight, quantization group size, which layers are quantized, compute precision, batching, and the
export path of each ExecuTorch arm.

#### Scenario: Bits per weight differ

- **WHEN** two arms compared in one row store a different number of bits per weight
- **THEN** both values are stated beside that row

### Requirement: An unmerged ExecuTorch arm is labelled

An ExecuTorch arm built from a branch that is not merged into the maintained branch SHALL carry its commit and
the words "unmerged development branch", and SHALL state how it was accepted when acceptance was not a
bit-exact or next-token-identical gate.

#### Scenario: A tuned profile accepted by the reference-error rule

- **WHEN** a tuned profile's next token differs from its parent's in some cell
- **THEN** that cell is marked and the acceptance rule is named

### Requirement: Results stay private until the owner releases them

Raw data and result tables of a comparison SHALL be kept out of any published location until the owner has
seen them and released them.

#### Scenario: A session finishes

- **WHEN** a device's comparison is complete
- **THEN** its numbers are delivered to the owner privately, and nothing is pushed, posted or added to a
  published report without the owner's release

### Requirement: No driver-level tracing

A comparison SHALL NOT use driver-level profiler capture or tracing on any device.

#### Scenario: A phase breakdown is wanted

- **WHEN** an arm's time is to be broken down
- **THEN** only the runtime's own timing output and the existing in-process event traces are used
