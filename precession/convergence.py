
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

from .periastron_precession import PrecessionConfig, PrecessionResult, calculate_precession
from .conservative_kernels import ConservativeMedium
from .orbit_derivatives import BinaryOrbit


@dataclass(frozen=True)
class ConvergenceRecord:
    kind: str
    setting: str
    delta_varpi_static: float | None
    delta_varpi_osc: float | None
    delta_varpi_total: float | None
    static_ir_status: str


def _record(kind: str, setting: str, result: PrecessionResult) -> ConvergenceRecord:
    return ConvergenceRecord(
        kind=kind,
        setting=setting,
        delta_varpi_static=result.delta_varpi_static,
        delta_varpi_osc=result.delta_varpi_osc,
        delta_varpi_total=result.delta_varpi_total,
        static_ir_status=result.static_ir_status,
    )


def convergence_study(
    orbit: BinaryOrbit,
    medium: ConservativeMedium,
    config: PrecessionConfig,
    *,
    harmonic_block: int = 2,
) -> tuple[PrecessionResult, list[ConvergenceRecord]]:

    base = calculate_precession(orbit, medium, config)
    records = [_record("base", "base", base)]
    variants = [("anomaly", f"n_ell={2 * config.n_ell}", replace(config, n_ell=2 * config.n_ell))]
    if config.engine == "legacy_kspace_validation":
        base_k_max = float(base.metadata["k_max"])
        variants.extend([
            ("angular", f"n_mu={2 * config.n_mu},n_phi={2 * config.n_phi}", replace(config, n_mu=2 * config.n_mu, n_phi=2 * config.n_phi)),
            ("radial", f"radial_order={2 * config.radial_order}", replace(config, radial_order=2 * config.radial_order)),
            ("uv", f"k_max={2 * base_k_max:g}", replace(config, k_max=2 * base_k_max)),
        ])
    if config.engine == "legacy_kspace_validation" and config.n_max + harmonic_block <= config.n_ell // 2:
        variants.append(("harmonic", f"n_max={config.n_max + harmonic_block}", replace(config, n_max=config.n_max + harmonic_block)))
    differences: dict[str, float | None] = {}
    for kind, setting, variant in variants:
        result = calculate_precession(orbit, medium, variant)
        records.append(_record(kind, setting, result))
        if base.delta_varpi_total is None or result.delta_varpi_total is None:
            differences[kind] = None
        else:
            differences[kind] = abs(result.delta_varpi_total - base.delta_varpi_total)

    finite = [value for value in differences.values() if value is not None]
    radial = differences.get("radial", base.error_radial)
    anomaly = max(differences.get("anomaly", 0.0), base.error_anomaly)
    angular = differences.get("angular", 0.0)
    uv = differences.get("uv", 0.0)
    harmonic_refinement = differences.get("harmonic", 0.0)
    harmonic = (
        None
        if base.error_harmonic is None or harmonic_refinement is None
        else max(base.error_harmonic, harmonic_refinement)
    )
    error_total = None
    if base.delta_varpi_total is not None and harmonic is not None:
        error_total = math.sqrt(sum(value * value for value in finite) + harmonic**2)
    metadata = dict(base.metadata)
    metadata["convergence_differences"] = differences
    result = replace(
        base,
        error_radial=float(radial or 0.0),
        error_anomaly=float(anomaly or 0.0),
        error_angular=None if angular is None else float(angular),
        error_uv=None if uv is None else float(uv),
        error_harmonic=None if harmonic is None else float(harmonic),
        error_total=error_total,
        metadata=metadata,
    )
    return result, records


def source_size_scan(
    orbit: BinaryOrbit,
    medium: ConservativeMedium,
    config: PrecessionConfig,
    source_sizes: tuple[float, ...] = (1.0e-1, 5.0e-2, 2.0e-2, 1.0e-2, 5.0e-3),
) -> list[ConvergenceRecord]:

    if config.engine != "legacy_kspace_validation":
        raise ValueError("source_size_scan requires engine='legacy_kspace_validation'")

    rows: list[ConvergenceRecord] = []
    for source_size in source_sizes:
        result = calculate_precession(orbit, medium, replace(config, source_size=source_size))
        rows.append(_record("source_size", f"r_s/a={source_size / orbit.a:g}", result))
    return rows


def write_convergence_records(path: Path, records: list[ConvergenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ConvergenceRecord.__dataclass_fields__))
        writer.writeheader()
        for row in records:
            writer.writerow(row.__dict__)
