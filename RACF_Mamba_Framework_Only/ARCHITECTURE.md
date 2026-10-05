# Structural flow and operator contracts

## Forward path

```text
image -> context stem -> proposal features
proposal features -> proposal head -> activation -> proposal geometry
proposal features + proposal descriptor -> reliability head -> shared r

proposal box + expansion(r) -> expanded/clipped box -> sampled ROI
ROI -> ROI stem -> ROI projection -> ROI features
proposal features -> context projection -> context features
ROI descriptor + context descriptor + r -> view gate -> alpha
alpha + ROI/context features -> fused observation

fused observation -> fine Mamba stream -> fine descriptor
coarsened fused observation -> coarse Mamba stream -> coarse descriptor
coarse descriptor + fine descriptor + r -> scale gate -> beta
beta + fine/coarse descriptors -> coarse-fine descriptor

coarse-fine descriptor -> latent graph -> graph descriptor
coarse-fine descriptor + graph descriptor + r -> integration -> z
z -> parallel experts -> expert probabilities
z + r -> expert gate -> expert weights
expert weights + expert probabilities -> final probability mixture
```

## External modules

`Modules` contains callables, not layer constructors. Stem outputs and projected
views must be spatially compatible. Fine and coarse streams must return
compatible descriptors. The graph internally supplies learned node projections,
relational attention, and readout. Its topology and numerical settings are not
specified in this package. Experts must share an output label space but may
have independently learned mappings. Their quantity is not specified here.

An external implementation is responsible for module registration, parameter
independence, placement, differentiation, and numerical stability.

## External operator provider

The `ops` argument exposes the following interface. No implementation or default
settings for these operations are distributed.

| Interface | Structural contract |
| --- | --- |
| `sigmoid(value)` | Convert the proposal activation or scalar gate input. |
| `proposal_geometry(activation)` | Return an image-dependent box and proposal descriptor. |
| `pool(features)` | Form a descriptor without exposing a grid size. |
| `concat(*values)` | Concatenate compatible descriptors. |
| `crop_expansion(r)` | Determine expansion from shared reliability. |
| `expand_and_clip(box, expansion)` | Expand the proposal and respect image boundaries. |
| `sample_roi(image, box)` | Sample the ROI; actual coordinates and interpolation remain external. |
| `blend_features(alpha, local, context)` | Weight local evidence by alpha and context by its complement. |
| `coarse_view(features)` | Construct the coarse spatial representation. |
| `blend_descriptors(beta, fine, coarse)` | Weight fine evidence by beta and coarse by its complement. |
| `softmax_classes(logits)` | Normalize each expert over the shared label space. |
| `softmax_experts(logits)` | Normalize mixture weights over the supplied experts. |
| `weighted_probability_sum(weights, probabilities)` | Form the final mixture without another softmax. |
| `stop_gradient(value)` | Detach the consistency weight only. |
| `classification_loss(probabilities, labels)` | Evaluate the classification objective. |
| `roi_context_consistency(local, context, weight)` | Apply detached reliability to cross-view consistency. |
| `prototype_regularization(z, labels)` | Provide prototype-based training regularization. |
| `expert_diversity(probabilities)` | Provide the expert-diversity objective. |

Training-term aggregation and prototype state management are omitted. A real
backend must implement stable probability and loss computations; the symbolic
flow is not a prescription to discard log-space numerical safeguards.

## Control variants

Only the shared-reliability topology is included. Numerical fixed-control
settings, stage-independent control implementations, model ablations, and
their evaluation are outside this package. It makes no validation claim for
those variants.

## Difference from the supplied core code

Concrete operators were replaced by named interfaces. The shared forward order,
fine/coarse blend orientation, graph dependency, common expert descriptor, and
probability-mixture semantics were retained. The original stable log-space
mixture is represented here by a mathematical probability-mixture interface.
The interface does not validate numerical or gradient equivalence.
