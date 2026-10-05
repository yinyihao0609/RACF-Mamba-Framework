# Public scope

## Included

- Shared-reliability module ordering and data dependencies.
- Semantic interfaces for adaptive observation, fusion, representation, and
  prediction.
- Training-objective names and the detached consistency-weight relationship.
- Source attribution, omitted-material disclosure, and file checksums.

## Excluded

- Model construction, concrete operators, dimensions, depths, and counts.
- Configuration files, default values, numerical examples, and preset variants.
- Learned weights, checkpoint files, optimizer state, and pretrained downloads.
- Training schedules, losses' coefficients, initialization and prototype state.
- Input preprocessing, augmentation, data partitions, and corruption generation.
- Training loops, batch evaluation, experiment logs, metrics, and timing reports.
- Original tests and their numerical validation summaries.
- Dataset images, manuscript files, and private author-facing review records.

The original source archive is not bundled into this reduced package.

## Verification boundary

The reduced Python reference was syntax-checked and exercised using symbolic
test doubles to check call order and shared-signal connections. These are
structural tests, not tensor inference, gradient checks, model training,
numerical equivalence, or validation of reported experimental performance.
Validation reports from the supplied core implementation do not apply to this
reduced package and have not been included.

## Release status

This is a local package prepared for author review. No remote repository or
public release is asserted. An eventual repository and manuscript statement
should describe this as an architecture-only reference and specify the actual
files made available. See `LICENSE_STATUS.md` before choosing release terms.
