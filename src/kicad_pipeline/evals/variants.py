"""Variant generation — auto-generate package-size variants of golden cases.

Wraps ``orchestrator.variants.fork_requirements_for_variant()`` to create
eval cases with different package strategies (0402, 0603, 1206).
"""
from __future__ import annotations

from kicad_pipeline.evals.models import EvalCase, SoftTarget
from kicad_pipeline.orchestrator.models import PackageStrategy

# ---------------------------------------------------------------------------
# Predefined variant strategies
# ---------------------------------------------------------------------------

VARIANT_STRATEGIES: tuple[PackageStrategy, ...] = (
    PackageStrategy(
        name="0402", resistor_package="0402",
        capacitor_package="0402", led_package="0603",
    ),
    PackageStrategy(
        name="0603", resistor_package="0603",
        capacitor_package="0603", led_package="0603",
    ),
    PackageStrategy(
        name="1206", resistor_package="1206",
        capacitor_package="1206", led_package="0805",
    ),
)


def _make_variant_build_fn(
    base_build_fn: object,
    strategy: PackageStrategy,
) -> object:
    """Create a closure that builds requirements then applies variant strategy."""
    from kicad_pipeline.orchestrator.variants import fork_requirements_for_variant

    def _variant_requirements():  # type: ignore[no-untyped-def]
        reqs = base_build_fn()  # type: ignore[operator]
        return fork_requirements_for_variant(reqs, strategy)

    return _variant_requirements


def _relax_targets(
    targets: tuple[SoftTarget, ...],
    floor_reduction: float = 0.05,
) -> tuple[SoftTarget, ...]:
    """Relax soft targets for variants (smaller packages may score differently)."""
    return tuple(
        SoftTarget(
            dimension=t.dimension,
            min_value=max(0.0, t.min_value - floor_reduction),
            regression_threshold=t.regression_threshold,
        )
        for t in targets
    )


def generate_variant_cases(
    base_case: EvalCase,
    strategies: tuple[PackageStrategy, ...] = VARIANT_STRATEGIES,
) -> tuple[EvalCase, ...]:
    """Generate variant EvalCases from a base golden case."""
    variants: list[EvalCase] = []

    for strategy in strategies:
        variant_id = f"{base_case.case_id}__{strategy.name}"
        variants.append(EvalCase(
            case_id=variant_id,
            board_name=f"{base_case.board_name} ({strategy.name})",
            description=f"{base_case.description} [variant: {strategy.name} passives]",
            build_fn=_make_variant_build_fn(base_case.build_fn, strategy),  # type: ignore[arg-type]
            board_width_mm=base_case.board_width_mm,
            board_height_mm=base_case.board_height_mm,
            hard_gates=base_case.hard_gates,
            soft_targets=_relax_targets(base_case.soft_targets),
            tags=(*base_case.tags, "variant", strategy.name),
        ))

    return tuple(variants)


def all_cases_with_variants(
    golden_cases: tuple[EvalCase, ...],
) -> tuple[EvalCase, ...]:
    """Return golden cases plus their package-size variants."""
    all_cases: list[EvalCase] = list(golden_cases)
    for case in golden_cases:
        all_cases.extend(generate_variant_cases(case))
    return tuple(all_cases)
