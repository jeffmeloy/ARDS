## ARDS (Adaptive Rate-Distortion Scaling): A Framework for Dynamic Control of Attention Channel Capacity

**Top-Level Goal: Intelligence as Learned, Adaptive Compression**

This system implements a framework for transformer control based on a core concept: high-level
reasoning is not just learned representation, but **learned, adaptive compression**. The model's
intelligence emerges from its ability to dynamically apply optimal, context-dependent strategies
for the lossy compression of information.

**The Core Realization: Attention as a Variable-Rate Compression Channel**

The computational locus of reasoning is the pre-softmax attention score matrix. The subsequent
softmax function acts as a non-linear compression algorithm, selectively routing information.
The central thesis of ARDS is that the *aggressiveness* of this compression should not be a
static architectural property, but a learned, dynamic variable.

**Genuine Control: Unbounded, Multiplicative Scaling of Information Density**

This system intervenes by multiplying the pre-softmax scores by a learned `info_density`
parameter. This directly and powerfully controls the compression ratio of the attention head.
The control mechanism is designed to be as simple and unconstrained as possible:

- **Multiplicative Scaling:** The controller learns a scaling factor that multiplies the
  raw attention scores.
- **Neutral Baseline of 1.0:** The controller learns a value in log-space (`log_density`).
  This is converted to a scaler via `exp(log_density)`, making `1.0` the natural,
  "do-nothing" baseline. A learned output of `0` results in a scaler of `1.0`.
- **Unbounded Control:** There are no artificial hyperparameters like `min/max` ranges or `gain`
  factors. The controller has full authority to set any positive scaling factor. We trust the
  main training loop (e.g., through weight decay and gradient clipping) to provide emergent
  regularization, forcing the controller to learn sensible policies that improve performance.

This approach treats each attention head as a variable-rate compression channel, where the
`info_density` scaler dictates the Rate-Distortion trade-off:

- **High `info_density` (> 1.0):** Aggressive compression. Creates a high-distortion, low-rate
  channel essential for filtering noise and focusing on logical signals.
- **Low `info_density` (< 1.0):** High-fidelity preservation. Creates a low-distortion, high-rate
  channel essential for creative association and capturing broad context.


