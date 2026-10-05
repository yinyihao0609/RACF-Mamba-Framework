"""Architecture-only flow extracted from the author-supplied RACF-Mamba code.

Concrete neural operators, dimensions, configurations, and weights are omitted.
The caller must supply all modules and operations; no model is constructed here.
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Modules:
    context_stem: Callable
    roi_stem: Callable
    proposal_head: Callable
    reliability_head: Callable
    context_projection: Callable
    roi_projection: Callable
    view_gate: Callable
    fine_stream: Callable
    coarse_stream: Callable
    scale_gate: Callable
    latent_graph: Callable
    descriptor: Callable
    experts: Sequence[Callable]
    expert_gate: Callable


@dataclass(frozen=True)
class Outputs:
    probabilities: Any
    expert_probabilities: Any
    expert_weights: Any
    descriptor: Any
    roi_descriptor: Any
    context_descriptor: Any
    reliability: Any


def forward(image, modules: Modules, ops) -> Outputs:
    """Show shared-reliability routing; `ops` is an external operator provider."""
    if not modules.experts:
        raise ValueError("Expert modules must be supplied by the caller.")
    proposal_features = modules.context_stem(image)
    activation = ops.sigmoid(modules.proposal_head(proposal_features))
    proposal_box, proposal_descriptor = ops.proposal_geometry(activation)
    reliability_input = ops.concat(
        ops.pool(proposal_features), proposal_descriptor
    )
    r = ops.sigmoid(modules.reliability_head(reliability_input))

    expansion = ops.crop_expansion(r)
    crop_box = ops.expand_and_clip(proposal_box, expansion)
    roi_image = ops.sample_roi(image, crop_box)

    context_features = modules.context_projection(proposal_features)
    roi_features = modules.roi_projection(modules.roi_stem(roi_image))
    f_context = ops.pool(context_features)
    f_roi = ops.pool(roi_features)
    alpha = ops.sigmoid(modules.view_gate(ops.concat(f_roi, f_context, r)))
    fused = ops.blend_features(alpha, roi_features, context_features)

    z_fine = modules.fine_stream(fused)
    z_coarse = modules.coarse_stream(ops.coarse_view(fused))
    beta = ops.sigmoid(modules.scale_gate(ops.concat(z_coarse, z_fine, r)))
    z_cf = ops.blend_descriptors(beta, z_fine, z_coarse)

    z_graph = modules.latent_graph(z_cf)
    z = modules.descriptor(ops.concat(z_cf, z_graph, r))
    expert_probabilities = tuple(
        ops.softmax_classes(head(z)) for head in modules.experts
    )
    expert_weights = ops.softmax_experts(
        modules.expert_gate(ops.concat(z, r))
    )
    probabilities = ops.weighted_probability_sum(
        expert_weights, expert_probabilities
    )

    return Outputs(
        probabilities=probabilities,
        expert_probabilities=expert_probabilities,
        expert_weights=expert_weights,
        descriptor=z,
        roi_descriptor=f_roi,
        context_descriptor=f_context,
        reliability=r,
    )


def training_terms(outputs: Outputs, labels, ops):
    """Return unweighted objective interfaces; loss configuration is not shipped."""
    classification = ops.classification_loss(outputs.probabilities, labels)
    consistency = ops.roi_context_consistency(
        outputs.roi_descriptor,
        outputs.context_descriptor,
        ops.stop_gradient(outputs.reliability),
    )
    prototype = ops.prototype_regularization(outputs.descriptor, labels)
    diversity = ops.expert_diversity(outputs.expert_probabilities)
    return {
        "classification": classification,
        "consistency": consistency,
        "prototype": prototype,
        "diversity": diversity,
    }
