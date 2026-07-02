# Quantization-Aware Distillation: Recovering the Quality of Quantized Models

## 1. Introduction & Motivation

Quantization shrinks a model by storing weights (and sometimes activations)
in fewer bits. Naive post-training quantization (PTQ) rounds a trained
full-precision model down to the target format in one shot. This is fast and
often good enough at 8 bits, but as the bit-width drops (4-bit, 3-bit, 2-bit)
the rounding error grows large enough that accuracy, perplexity, and
downstream task quality degrade noticeably.

The key insight behind quantization-aware training (QAT) and
quantization-aware distillation is that the model does not have to passively
absorb this damage. If the network is allowed to *keep learning while it
experiences quantization noise*, it can adjust its remaining degrees of
freedom to compensate. The rounding error becomes just another form of noise
the optimizer learns to route around.

This document describes a practical recipe built from four ingredients:

1. **Self-distillation** — use the original full-precision model as a teacher
   and train a quantized copy of *the same model* to match it.
2. **Fake quantization** — during the forward pass, simulate the exact
   numerical format that will be used at inference time, so the loss reflects
   the real deployment error.
3. **The straight-through estimator (STE)** — make the non-differentiable
   quantization step trainable by passing gradients through it.
4. **Selective freezing** — train only the parameters that quantization
   actually perturbs, and freeze everything else.

Together these make the training signal *quantization-noise-aware*: the model
is optimized under precisely the conditions it will face when served.

## 2. Problem Setup

We start from a trained full-precision checkpoint. Conceptually we hold two
copies of it:

- The **teacher**: the untouched full-precision model, kept frozen.
- The **student**: a copy whose target layers are quantized during the
  forward pass.

Notation:

- `W` — the underlying high-precision ("latent") weights that we continue to
  optimize.
- `Q(W)` — the quantized-then-dequantized version of `W` produced by the fake
  quantizer.
- `z_T` — teacher logits (produced from `W` at full precision, no gradient).
- `z_S` — student logits (produced using `Q(W)`).

The objective, at a high level, is to make the student's output distribution
match the teacher's *despite* the student running through quantized weights:

```
minimize  Distill(z_S, z_T)   over the trainable subset of W
```

"Recovering quality" means closing the gap between the quantized model and the
full-precision baseline. We measure it the same way we evaluate any language
model — validation perplexity and/or downstream task accuracy — comparing
three points: the full-precision baseline (upper bound), naive PTQ (lower
bound), and the distilled quantized model (what we are trying to push toward
the baseline).

## 3. Self-Distillation from the Same Model

### Teacher and student are the same checkpoint

Classical knowledge distillation trains a small student to imitate a large,
separately trained teacher. Here we use a variant: the teacher and the student
are the *same model*. The teacher is the frozen full-precision version; the
student is the quantized version of that identical checkpoint.

This is attractive for several reasons:

- The teacher is guaranteed to be a strong, well-matched target — it is
  literally the model we are trying to preserve.
- The student starts extremely close to the teacher, so training is stable and
  converges quickly; we are recovering lost quality, not learning from scratch.
- No separate teacher training or teacher/student architecture mismatch to
  manage.

### The distillation objective

The primary loss is a distributional match between the two logit sets. A
standard choice is the temperature-scaled KL divergence on the soft targets:

```
p_T = softmax(z_T / τ)
p_S = softmax(z_S / τ)
L_distill = τ² · KL(p_T || p_S)
```

The temperature `τ` softens the distributions so the student learns from the
teacher's full ranking of tokens, not just the top-1. The `τ²` factor keeps
the gradient magnitude comparable across temperatures.

Optionally, a hard-label cross-entropy term against the ground-truth next
token can be mixed in:

```
L = α · L_distill + (1 - α) · L_CE(z_S, y)
```

In pure quality-recovery settings the soft-target term usually dominates or is
used alone, because the teacher's soft distribution already encodes everything
we want to preserve.

### Practical mechanics

Each training step needs two forward passes over the same input batch:

1. A **teacher forward** at full precision, under `no_grad`, to produce `z_T`.
2. A **student forward** using fake-quantized weights to produce `z_S`, with
   gradients enabled.

The teacher forward is pure inference — no backward, no optimizer state — so
its main cost is memory (a second set of weights) and one extra forward's
worth of compute. When memory is tight, the teacher and student can share the
same latent weights and differ only in whether the fake quantizer is applied,
which avoids storing two full weight copies. The essential requirement is that
the teacher path never sees the quantizer and never receives gradients.

## 4. Fake Quantization in the Forward Pass

### What fake quantization is

"Fake" (or simulated) quantization runs the network at full precision but
forces the target tensors through a quantize→dequantize round trip inside the
forward pass:

```
W_q   = quantize(W, scale, zero_point, format)   # to the low-bit grid
Q(W)  = dequantize(W_q, scale, zero_point)         # back to float
y     = x @ Q(W)ᵀ                                  # matmul in float
```

The matmul still executes in a normal floating-point kernel, but its operands
carry exactly the values the real low-bit kernel would use. The model
therefore *experiences* the quantization error during training without needing
a specialized low-bit kernel in the training loop.

### Match the inference format exactly

This is the entire point of the technique, and the part most easily gotten
wrong. The fake quantizer must reproduce the numerical behavior of the format
that will actually be served. If they diverge, the loss is measuring a
different model than the one you deploy, and any "recovery" is illusory.

Dimensions that must match the target format:

- **Data format family.** Integer (INT) formats use a uniform grid defined by
  a scale and (optionally) a zero-point. Floating-point (FP) low-bit formats
  use a non-uniform grid with an exponent/mantissa split. The
  quantize/dequantize math is different for each; use the one the serving stack
  uses.
- **Weight-only vs. weight-and-activation.** If only weights are quantized at
  inference, only weights are fake-quantized in training. If activations are
  also quantized at serving time, they must be fake-quantized in the forward
  pass too, at the same points in the graph.
- **Granularity.** Per-tensor, per-channel, or per-group (block) scales change
  the error distribution substantially. Use the same grouping the deployment
  format uses, including the group size.
- **Scale/zero-point derivation.** How the scale (and zero-point) are chosen
  from the weights — the range they cover, and any rounding of the scale
  itself — should follow the deployment quantizer's rules.

When the forward mirrors served precision this way, the distillation loss is
genuinely *quantization-noise-aware*: every unit of loss corresponds to error
the deployed model will really incur.

### Where fake quant goes

Fake quantization is inserted at the layers that get quantized for serving —
in transformer LLMs this is overwhelmingly the linear projections (attention
Q/K/V/O and the MLP matrices), which hold the vast majority of parameters.
Layers that remain in higher precision at inference are left untouched in the
forward pass, so they are simulated exactly as they will be served: at full
precision.

## 5. Straight-Through Estimator (STE) for the Backward Pass

### The gradient problem

Quantization is built from rounding and clamping. Rounding is a step function:
its derivative is zero almost everywhere and undefined at the steps. If we
differentiated it honestly, no gradient would ever reach the latent weights
and training would stall immediately.

### The STE fix

The straight-through estimator resolves this by using different functions on
the forward and backward passes:

- **Forward:** use the true quantized value `Q(W)`.
- **Backward:** pretend the quantizer was the identity, i.e. let
  `∂Q(W)/∂W ≈ 1`, so the gradient flows straight through to `W`.

In effect the optimizer updates the high-precision latent weights `W` using
gradients computed as if the quantizer were transparent, while the loss it is
descending was produced by the genuinely quantized `Q(W)`. Over many steps the
latent weights drift to positions whose quantized projections yield lower loss.

### Handling the clamp region

There is one important refinement. Values pushed outside the representable
range are clamped, and gradients for those elements are meaningless — nudging
a saturated weight further out does nothing at the output. The standard
practice is to *mask* the pass-through gradient to zero for elements that fall
outside the quantization range, and pass it through unchanged (gradient of 1)
for elements inside it. This keeps the optimizer from wasting updates on
saturated weights and stabilizes training.

### Why this works

The latent weights act as a continuous accumulator of small updates. Even
though every update is quantized before it affects the output, the
accumulation of many sub-step nudges eventually flips weights across
quantization boundaries in the direction that reduces loss. This is what lets a
model with a fixed low-bit grid keep improving.

## 6. Freezing Non-Quantized Modules

Not every parameter should be trained. The guiding principle: **train exactly
the parameters that quantization perturbs, and freeze the rest.**

Typical split:

- **Trainable:** the latent weights of the quantized linear layers — the ones
  actually experiencing rounding error and thus the ones that can learn to
  compensate for it.
- **Frozen:** parameters that stay in high precision at inference — commonly
  embeddings, normalization scales/biases, biases on linear layers, the LM
  head if it is not quantized, and any other module served at full precision.

The rationale:

- **Focus.** The quality loss originates in the quantized layers. Concentrating
  the optimizer there targets the actual damage instead of drifting parameters
  that were already correct.
- **Stability.** Fewer trainable parameters and a student that already matches
  the teacher closely means small, well-behaved updates — appropriate for a
  recovery fine-tune rather than a from-scratch train.
- **Efficiency.** Freezing modules removes their optimizer state (e.g. the
  moment buffers of an Adam-style optimizer), which is a large memory saving,
  and skips their gradient computation.

The teacher, separately, is frozen in its entirety — it is a fixed reference
and must never be updated. Only the student's quantized-layer latent weights
receive optimizer updates.

## 7. Putting It Together: The Training Loop

### Step by step

For each batch of input tokens:

1. **Teacher forward (no grad, full precision):** compute `z_T`.
2. **Student forward (fake quant on target layers):** compute `z_S` using
   `Q(W)` for the quantized linears; frozen modules run normally.
3. **Distillation loss:** `L = τ² · KL(softmax(z_T/τ) || softmax(z_S/τ))`
   (optionally plus a hard-label term).
4. **Backward with STE:** gradients flow through the quantizer as identity
   (masked in the clamp region) into the latent weights `W`.
5. **Optimizer step:** update only the trainable latent weights; frozen
   parameters and the teacher are left alone.

### Pseudocode sketch (framework-agnostic)

```python
teacher = load_full_precision_checkpoint()   # frozen, eval mode
student = load_full_precision_checkpoint()   # target linears wrapped with fake quant
freeze_non_quantized_modules(student)         # train only quantized latent weights

for batch in dataloader:
    with no_grad():
        z_T = teacher(batch)                  # full-precision reference logits

    z_S = student(batch)                      # forward uses Q(W); STE on backward

    loss = kl_divergence(
        softmax(z_T / tau),
        log_softmax(z_S / tau),
    ) * (tau ** 2)

    loss.backward()                            # gradients reach latent W via STE
    optimizer.step()                           # updates trainable latent weights only
    optimizer.zero_grad()
```

The fake-quant wrapper on each target linear is responsible for (a) computing
`Q(W)` in the forward using the deployment format, and (b) applying the
straight-through gradient (with clamp masking) in the backward.

### Schedule and initialization

- **Initialization of scales.** Before training, derive each layer's
  quantization scale (and zero-point) from its weights using the same
  procedure the deployment quantizer uses. Good initial scales mean the student
  starts close to the teacher and training only has to close a small gap.
- **Learning rate.** Use a small learning rate with warmup, as befits a
  recovery fine-tune. The latent weights only need gentle nudging to cross
  quantization boundaries; large steps destabilize the STE dynamics.
- **Duration.** Because the student starts near the teacher, quality typically
  recovers in a relatively short fine-tune rather than a full pretraining
  budget.

## 8. Practical Considerations & Pitfalls

- **Bit-exact fake quant.** The single most important correctness check: the
  fake quantizer's output must match, value for value, what the inference
  kernel produces for the same weights and format. Any mismatch — a different
  rounding mode, a different group size, a slightly different scale rule — means
  you are training against a model you will not deploy, and the loss will lie
  to you.
- **Training precision vs. simulated format.** Run the surrounding computation
  in a stable precision (e.g. bf16/fp32 accumulation) while the *values* going
  through the quantizer follow the low-bit grid. Don't let the training
  framework's own reduced precision silently perturb the simulated format.
- **Distributed training.** When weights are sharded across devices, make sure
  quantization statistics (scales, groupings) are computed consistently with
  how the model is partitioned, so no shard sees a different grid than it would
  at serving time. Keep the teacher and student partitioned identically.
- **Checkpointing.** During training, checkpoint the high-precision *latent*
  weights and the quantization parameters — that is the trainable state. The
  low-bit export is a separate, final artifact.
- **Common failure modes.**
  - Loss not tracking the teacher → check that fake quant is actually active in
    the student forward and absent from the teacher.
  - Dead / vanishing gradients → check STE is applied (not honest rounding
    gradients) and that clamp masking isn't masking almost everything (scales
    too tight).
  - Scale drift / instability → learning rate too high, or scales being updated
    in a way inconsistent with the deployment quantizer.

## 9. Evaluation & Export

- **Validate recovery.** Compare three configurations on the same evaluation
  suite: the full-precision baseline, naive PTQ of the same checkpoint, and the
  distilled quantized model. Success is the distilled model closing most of the
  gap between PTQ and the baseline.
- **Evaluate in the served format.** Measure the distilled model using the same
  quantization it will be deployed with (either via the fake quantizer or,
  better, the real inference path), so the reported numbers reflect deployment.
- **Export.** Materialize the trained latent weights into the actual low-bit
  format used for serving — the packed quantized weights plus their scales and
  zero-points. Because training matched the inference format throughout, the
  exported model's behavior should align with the loss you optimized.

## 10. Summary

The four techniques reinforce one another:

- **Self-distillation** gives a strong, perfectly matched target — the model's
  own full-precision behavior.
- **Fake quantization** ensures the forward pass experiences exactly the
  numerical error of the deployment format, making the loss
  quantization-noise-aware.
- **STE** makes that non-differentiable error trainable, letting the latent
  weights keep learning across quantization boundaries.
- **Freezing** confines learning to the parameters quantization actually
  damages, for stability and efficiency.

The result is a fine-tuning procedure that optimizes the model under the very
conditions it will be served in, recovering much of the quality that naive
post-training quantization gives away.




