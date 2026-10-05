# RACF-Mamba: architecture-only reference

This package presents the structural data flow of RACF-Mamba, extracted and
abstracted from the author-supplied core code. It explains shared reliability
conditioning without supplying concrete model configurations or weights.

**Scope: architecture reference, not a standalone neural model or an end-to-end
training and evaluation release.** Importing `framework.py` does not construct a
model. Concrete modules and numerical operators must be supplied separately.

## Central mechanism

A shared proposal-derived reliability signal connects adaptive observation,
view fusion, coarse-fine representation, descriptor integration, expert
aggregation, and the consistency objective. Mamba, latent graph reasoning, and
parallel experts are supporting computations within that structure.

The reference preserves these relationships:

- The full-image context stem precedes proposal generation and ROI extraction.
- The same reliability value conditions the linked stages of the shared mode.
- ROI and context stems are separate module inputs and must not share trainable
  parameters in an implementation of the described independent-stem design.
- The graph receives the coarse-fine descriptor and returns latent relational
  evidence for descriptor integration.
- Every expert receives the same integrated descriptor.
- Prediction is a weighted mixture of expert probabilities, with no additional
  softmax on the mixture.
- Consistency uses a detached reliability weight; this does not imply detaching
  reliability in other computations.

Reliability is a classification-trained control signal, not calibrated
diagnostic confidence. Latent graph nodes are not manually labeled attributes.

## Files

- `framework.py`: symbolic forward-flow and unweighted training-term interfaces.
- `ARCHITECTURE.md`: module order, operator contracts, and omitted details.
- `PUBLIC_SCOPE.md`: included and excluded material, plus validation limits.
- `PROVENANCE.md`: origin of this reduced package.
- `THIRD_PARTY_NOTICES.md`: upstream attribution and dependency boundaries.
- `LICENSE_STATUS.md`: license status inherited from the supplied package.
- `SHA256SUMS`: checksums of the delivered public files.

## Parameters and implementation details

No concrete input resolution, channel width, depth, state size, node count,
expert count, crop coefficient, numerical tolerance, loss coefficient,
temperature, initialization value, optimizer setting, or learned weight is
provided. There are no configuration files or numerical model examples.

The operator provider is intentionally absent and has no default implementation.
This package does not report accuracy, runtime, parameter counts, gradient
validity, or agreement with experimental results. It cannot be used on its own
to run the reported experiments.

## 中文说明

本包仅保留从作者提供代码中整理的核心结构与共享可靠性连接关系。
没有提供具体结构参数、训练超参数、初始化数值、权重和完整算子实现。
它不是可直接训练或评估的完整模型。公开时请保留这一范围说明，
不要将结构参考描述为完整实验代码发布。
