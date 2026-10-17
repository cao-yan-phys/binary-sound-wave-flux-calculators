
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from .conservative_kernels import (
    ClassicalFluid,
    ConservativeMedium,
    QuantumFluid,
    static_ir_is_prescription_dependent,
)
from .offshell_source import Backend, CudaQuadraticSource, OffshellSource, cuda_available, source_quadratic_parts
from .orbit_derivatives import BinaryOrbit
from .principal_value import PVIntegralResult, principal_value_integral


Sector = Literal["total", "static", "osc"]
Window = Literal["none", "gaussian", "lorentzian"]
Engine = Literal["analytic", "legacy_kspace_validation"]


@dataclass(frozen=True)
class PrecessionConfig:

    n_max: int = 8
    n_ell: int = 256
    n_mu: int = 8
    n_phi: int = 16
    radial_order: int = 16
    k_min: float = 0.0
    k_max: float | None = None
    sector: Sector = "total"
    source_size: float = 0.0
    source_window: Window = "gaussian"
    tail_window: int = 2
    tail_rtol: float = 1.0e-5
    tail_atol: float = 1.0e-10
    consecutive_windows: int = 2
    anomaly_rtol: float = 1.0e-3
    anomaly_atol: float = 1.0e-5
    anomaly_max_refinements: int = 3
    anomaly_max_n_ell: int = 2048
    strict_convergence: bool = True
    e_switch: float = 1.0e-3
    perturbative_max_shift: float = 0.1
    backend: Backend = "auto"
    engine: Engine = "analytic"

    @classmethod
    def fast(cls) -> "PrecessionConfig":

        return cls(
            n_max=8,
            n_ell=128,
            n_mu=4,
            n_phi=8,
            radial_order=8,
            tail_rtol=1.0e-3,
        )

    @classmethod
    def standard(cls) -> "PrecessionConfig":

        return cls()

    @classmethod
    def validation(cls) -> "PrecessionConfig":

        return cls(
            n_max=20,
            n_ell=512,
            n_mu=12,
            n_phi=24,
            radial_order=32,
            tail_rtol=1.0e-7,
        )

    def __post_init__(self) -> None:
        if self.n_max < 1:
            raise ValueError("n_max must be at least 1")
        if self.n_ell < 32 or self.n_ell & (self.n_ell - 1):
            raise ValueError("n_ell must be a power of two and at least 32")
        if self.n_max > self.n_ell // 2:
            raise ValueError("n_max must not exceed n_ell/2")
        if self.n_mu < 2 or self.n_phi < 4:
            raise ValueError("n_mu must be at least 2 and n_phi at least 4")
        if self.radial_order < 8:
            raise ValueError("radial_order must be at least 8")
        if self.k_min < 0.0:
            raise ValueError("k_min must be non-negative")
        if self.k_max is not None and self.k_max <= self.k_min:
            raise ValueError("k_max must be larger than k_min")
        if self.sector not in ("total", "static", "osc"):
            raise ValueError("sector must be 'total', 'static', or 'osc'")
        if self.source_size < 0.0:
            raise ValueError("source_size must be non-negative")
        if self.source_window not in ("none", "gaussian", "lorentzian"):
            raise ValueError("unsupported source_window")
        if self.tail_window < 1 or self.consecutive_windows < 1 or self.tail_rtol <= 0.0 or self.e_switch <= 0.0:
            raise ValueError("tail and circular-limit controls must be positive")
        if self.tail_atol < 0.0 or self.anomaly_atol < 0.0 or self.anomaly_rtol <= 0.0:
            raise ValueError("invalid absolute or anomaly tolerance")
        if self.anomaly_max_refinements < 0:
            raise ValueError("anomaly_max_refinements must be non-negative")
        if self.anomaly_max_n_ell < 32 or self.anomaly_max_n_ell & (self.anomaly_max_n_ell - 1):
            raise ValueError("anomaly_max_n_ell must be a power of two and at least 32")
        if self.backend not in ("auto", "cuda", "cpu"):
            raise ValueError("backend must be 'auto', 'cuda', or 'cpu'")
        if self.engine == "legacy_kspace_validation" and self.backend == "cuda" and not cuda_available():
            raise RuntimeError("CUDA backend requested, but numba.cuda is unavailable")
        if self.engine not in ("analytic", "legacy_kspace_validation"):
            raise ValueError("engine must be 'analytic' or 'legacy_kspace_validation'")


@dataclass(frozen=True)
class HarmonicContribution:
    n: int
    frequency_weight: int
    sector: str
    delta_varpi: float | None
    self_1: float | None
    self_2: float | None
    cross: float | None
    radial_error: float | None
    k_pole: float
    kappa_evanescent: float | None
    radial_subtraction: bool


@dataclass(frozen=True)
class PrecessionResult:

    delta_varpi_static: float | None
    delta_varpi_osc: float | None
    delta_varpi_total: float | None
    delta_varpi_static_self_1: float | None
    delta_varpi_static_self_2: float | None
    delta_varpi_static_cross: float | None
    delta_varpi_osc_self_1: float | None
    delta_varpi_osc_self_2: float | None
    delta_varpi_osc_cross: float | None
    delta_varpi_total_self_1: float | None
    delta_varpi_total_self_2: float | None
    delta_varpi_total_cross: float | None
    dH_static_de: float | None
    dH_osc_de: float | None
    dH_total_de: float | None
    harmonics: tuple[HarmonicContribution, ...]
    error_radial: float
    error_harmonic: float | None
    error_anomaly: float
    error_angular: float | None
    error_uv: float | None
    error_static_ir: float | None
    error_total: float | None
    harmonic_converged: bool
    static_ir_status: str
    static_sector_included: bool
    quantity: str
    literal_periapsis_defined: bool
    perturbative_valid: bool | None
    model: str
    include_self_gravity: bool
    response_prescription: str
    metadata: dict[str, object]
    anomaly_converged: bool = False
    n_ell_used: int | None = None

    @property
    def delta_varpi_total_normalized(self) -> float | None:
        scale = self.metadata.get("response_eta")
        return None if self.delta_varpi_total is None or not scale else self.delta_varpi_total / scale

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["harmonics"] = [asdict(row) for row in self.harmonics]
        data["delta_varpi_total_normalized"] = self.delta_varpi_total_normalized
        return data

    def write(self, output_dir: Path) -> None:

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "precession_result.json").write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        with (output_dir / "harmonics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(self.harmonics[0]).keys()) if self.harmonics else ["n"])
            writer.writeheader()
            for row in self.harmonics:
                writer.writerow(asdict(row))


def _angular_nodes(n_mu: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    mu, weight_mu = np.polynomial.legendre.leggauss(n_mu)
    phi = 2.0 * math.pi * (np.arange(n_phi, dtype=np.float64) + 0.5) / n_phi
    directions: list[np.ndarray] = []
    weights: list[float] = []
    for mu_value, mu_weight in zip(mu, weight_mu, strict=True):
        sin_theta = math.sqrt(max(0.0, 1.0 - float(mu_value) ** 2))
        for phi_value in phi:
            directions.append(
                np.array((sin_theta * math.cos(phi_value), sin_theta * math.sin(phi_value), float(mu_value)))
            )
            weights.append(float(mu_weight) * 2.0 * math.pi / n_phi)
    return np.asarray(directions, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _window(k: float, config: PrecessionConfig) -> float:
    if config.source_size == 0.0 or config.source_window == "none":
        return 1.0
    q = k * config.source_size
    if config.source_window == "gaussian":
        return math.exp(-0.5 * q * q)
    return 1.0 / (1.0 + q * q)


def _default_k_max(orbit: BinaryOrbit, medium: ConservativeMedium, config: PrecessionConfig) -> float:
    roots = [medium.roots(n * orbit.physical_omega)[0] for n in range(config.n_max + 1)]
    return max(20.0 / orbit.a, 3.0 * max(roots))


def _precession_prefactor(orbit: BinaryOrbit) -> float:
    return -2.0 * math.pi * orbit.total_mass * math.sqrt(1.0 - orbit.e * orbit.e) / (
        orbit.nu * orbit.a * orbit.a * orbit.physical_omega**2 * orbit.e
    )


def _angular_quadratic_function(
    source: OffshellSource,
    n: int,
    directions: np.ndarray,
    weights: np.ndarray,
    config: PrecessionConfig,
    cuda_source: CudaQuadraticSource | None = None,
):
    cache: dict[float, np.ndarray] = {}

    def q_function(k: float) -> np.ndarray:
        key = float(k)
        if key in cache:
            return cache[key]
        if cuda_source is not None:
            result = cuda_source.quadratic_parts(n, key) * _window(key, config) ** 2
            cache[key] = result
            return result
        accumulator = np.zeros(4, dtype=np.float64)
        amplitude_window = _window(key, config)
        for direction, weight in zip(directions, weights, strict=True):
            harmonic = source.on_shell_harmonic(n, key, direction)
            accumulator += weight * source_quadratic_parts(harmonic)
        result = accumulator * amplitude_window * amplitude_window
        cache[key] = result
        return result

    if cuda_source is not None:
        def prefetch(k_values: np.ndarray) -> None:
            missing: list[float] = []
            for value in np.asarray(k_values, dtype=np.float64):
                key = float(value)
                if key not in cache:
                    missing.append(key)
            if not missing:
                return
            parts = cuda_source.quadratic_parts_many(n, np.asarray(missing, dtype=np.float64))
            for key, part in zip(missing, parts, strict=True):
                cache[key] = part * _window(key, config) ** 2

        setattr(q_function, "prefetch", prefetch)

    return q_function


def _resolved_backend(config: PrecessionConfig) -> str:
    if config.backend == "auto":
        return "cuda" if cuda_available() else "cpu"
    return config.backend


def _sum_parts(rows: list[HarmonicContribution]) -> np.ndarray:
    if not rows:
        return np.zeros(4, dtype=np.float64)
    return np.sum(
        np.array([[row.delta_varpi, row.self_1, row.self_2, row.cross] for row in rows], dtype=np.float64),
        axis=0,
    )


def _calculate_non_circular(
    orbit: BinaryOrbit,
    medium: ConservativeMedium,
    config: PrecessionConfig,
) -> PrecessionResult:
    if orbit.e <= 0.0:
        raise ValueError("internal non-circular path requires e > 0")
    source = OffshellSource(orbit, n_ell=config.n_ell)
    directions, weights = _angular_nodes(config.n_mu, config.n_phi)
    cuda_source = source.cuda_quadratic_source(directions, weights) if _resolved_backend(config) == "cuda" else None
    k_max = _default_k_max(orbit, medium, config) if config.k_max is None else config.k_max
    assert k_max is not None
    prefactor = _precession_prefactor(orbit)
    static_ir = static_ir_is_prescription_dependent(medium)

    unresolved_static_rows: list[HarmonicContribution] = []
    static_rows: list[HarmonicContribution] = []
    osc_rows: list[HarmonicContribution] = []
    dH_static: float | None = None
    dH_osc: float | None = None
    radial_error_sq = 0.0
    convergence_passes = 0
    terminated_by_tail = False
    n_max_evaluated = 0

    if static_ir and config.sector in ("total", "static"):
        k_pole, kappa = medium.roots(0.0)
        unresolved_static_rows.append(
            HarmonicContribution(
                n=0,
                frequency_weight=1,
                sector="static",
                delta_varpi=None,
                self_1=None,
                self_2=None,
                cross=None,
                radial_error=None,
                k_pole=k_pole,
                kappa_evanescent=kappa,
                radial_subtraction=False,
            )
        )
    n_values = range(0, config.n_max + 1) if config.sector == "total" else (range(0, 1) if config.sector == "static" else range(1, config.n_max + 1))
    for n in n_values:
        if n == 0 and static_ir:
            continue
        n_max_evaluated = n
        omega = n * orbit.physical_omega
        q_function = _angular_quadratic_function(source, n, directions, weights, config, cuda_source)
        pv_result: PVIntegralResult = principal_value_integral(
            medium,
            omega,
            q_function,
            k_min=config.k_min,
            k_max=k_max,
            radial_order=config.radial_order,
        )
        frequency_weight = 1 if n == 0 else 2
        delta_parts = prefactor * frequency_weight * pv_result.value
        error_parts = abs(prefactor) * frequency_weight * pv_result.error
        radial_error_sq += float(np.sum(error_parts * error_parts))
        row = HarmonicContribution(
            n=n,
            frequency_weight=frequency_weight,
            sector="static" if n == 0 else "osc",
            delta_varpi=float(delta_parts[0]),
            self_1=float(delta_parts[1]),
            self_2=float(delta_parts[2]),
            cross=float(delta_parts[3]),
            radial_error=float(error_parts[0]),
            k_pole=pv_result.pole,
            kappa_evanescent=pv_result.kappa,
            radial_subtraction=pv_result.used_subtraction,
        )
        if n == 0:
            static_rows.append(row)
            dH_static = orbit.total_mass**2 * float(pv_result.value[0])
        else:
            osc_rows.append(row)
            dH_osc = (0.0 if dH_osc is None else dH_osc) + 2.0 * orbit.total_mass**2 * float(pv_result.value[0])
            if len(osc_rows) >= 2 * config.tail_window:
                recent = np.asarray(
                    [[item.delta_varpi, item.self_1, item.self_2, item.cross] for item in osc_rows[-config.tail_window :]],
                    dtype=np.float64,
                )
                previous = np.asarray(
                    [
                        [item.delta_varpi, item.self_1, item.self_2, item.cross]
                        for item in osc_rows[-2 * config.tail_window : -config.tail_window]
                    ],
                    dtype=np.float64,
                )
                current_osc = _sum_parts(osc_rows)
                scale = np.maximum(np.abs(current_osc), np.finfo(float).tiny)
                recent_absolute = np.sum(np.abs(recent), axis=0)
                signed_change = np.abs(np.sum(recent, axis=0) - np.sum(previous, axis=0))
                converged_window = bool(
                    np.all(recent_absolute / scale <= config.tail_rtol)
                    and np.all(signed_change / scale <= config.tail_rtol)
                )
                if converged_window:
                    convergence_passes += 1
                    if convergence_passes >= config.consecutive_windows:
                        terminated_by_tail = True
                        break
                else:
                    convergence_passes = 0

    static_parts = _sum_parts(static_rows)
    osc_parts = _sum_parts(osc_rows)
    static_included = config.sector in ("total", "static") and not static_ir
    static_values = static_parts if static_included else None
    osc_values = osc_parts if config.sector in ("total", "osc") else None
    total_values = None
    if config.sector == "total" and not static_ir:
        total_values = static_parts + osc_parts
    elif config.sector == "static" and not static_ir:
        total_values = static_parts
    elif config.sector == "osc":
        total_values = osc_parts

    harmonic_tail_metadata: dict[str, object]
    if len(osc_rows) >= 2 * config.tail_window:
        recent = np.asarray(
            [[row.delta_varpi, row.self_1, row.self_2, row.cross] for row in osc_rows[-config.tail_window :]],
            dtype=np.float64,
        )
        previous = np.asarray(
            [
                [row.delta_varpi, row.self_1, row.self_2, row.cross]
                for row in osc_rows[-2 * config.tail_window : -config.tail_window]
            ],
            dtype=np.float64,
        )
        scale = np.maximum(np.abs(osc_parts), np.finfo(float).tiny)
        recent_absolute = np.sum(np.abs(recent), axis=0)
        signed_change = np.abs(np.sum(recent, axis=0) - np.sum(previous, axis=0))
        relative_absolute = recent_absolute / scale
        relative_signed_change = signed_change / scale
        harmonic_converged = terminated_by_tail
        harmonic_error = float(max(recent_absolute[0], signed_change[0]))
        harmonic_tail_metadata = {
            "block_size": config.tail_window,
            "recent_absolute_total_self1_self2_cross": recent_absolute.tolist(),
            "signed_block_change_total_self1_self2_cross": signed_change.tolist(),
            "relative_recent_absolute_total_self1_self2_cross": relative_absolute.tolist(),
            "relative_signed_block_change_total_self1_self2_cross": relative_signed_change.tolist(),
            "consecutive_windows_required": config.consecutive_windows,
        }
    else:
        harmonic_converged = False
        harmonic_error = None
        harmonic_tail_metadata = {
            "block_size": config.tail_window,
            "reason": "fewer than two complete harmonic blocks",
        }
    if not harmonic_converged and config.strict_convergence:
        raise RuntimeError(
            "precession harmonic sum did not converge before the n_max safety cap; "
            "increase n_max or set strict_convergence=False to inspect the truncated diagnostic result"
        )
    error_total: float | None
    if total_values is None or harmonic_error is None:
        error_total = None
    else:
        error_total = math.sqrt(radial_error_sq + harmonic_error * harmonic_error)
    result_value = None if total_values is None else float(total_values[0])
    perturbative_valid = None if result_value is None else abs(result_value) <= config.perturbative_max_shift

    dH_total = None
    if config.sector == "total" and not static_ir:
        dH_total = (dH_static or 0.0) + (dH_osc or 0.0)
    elif config.sector == "static":
        dH_total = dH_static
    elif config.sector == "osc":
        dH_total = dH_osc
    all_rows = tuple(unresolved_static_rows + static_rows + osc_rows)
    return PrecessionResult(
        delta_varpi_static=None if static_values is None else float(static_values[0]),
        delta_varpi_osc=None if osc_values is None else float(osc_values[0]),
        delta_varpi_total=result_value,
        delta_varpi_static_self_1=None if static_values is None else float(static_values[1]),
        delta_varpi_static_self_2=None if static_values is None else float(static_values[2]),
        delta_varpi_static_cross=None if static_values is None else float(static_values[3]),
        delta_varpi_osc_self_1=None if osc_values is None else float(osc_values[1]),
        delta_varpi_osc_self_2=None if osc_values is None else float(osc_values[2]),
        delta_varpi_osc_cross=None if osc_values is None else float(osc_values[3]),
        delta_varpi_total_self_1=None if total_values is None else float(total_values[1]),
        delta_varpi_total_self_2=None if total_values is None else float(total_values[2]),
        delta_varpi_total_cross=None if total_values is None else float(total_values[3]),
        dH_static_de=dH_static,
        dH_osc_de=dH_osc,
        dH_total_de=dH_total,
        harmonics=all_rows,
        error_radial=math.sqrt(radial_error_sq),
        error_harmonic=harmonic_error,
        error_anomaly=0.0,
        error_angular=None,
        error_uv=None,
        error_static_ir=None if not static_ir else math.inf,
        error_total=error_total,
        harmonic_converged=harmonic_converged,
        static_ir_status="PRESCRIPTION_DEPENDENT" if static_ir else "FINITE_UNDER_SELECTED_PRESCRIPTION",
        static_sector_included=static_included,
        quantity="periastron_precession",
        literal_periapsis_defined=True,
        perturbative_valid=perturbative_valid,
        model="classical" if isinstance(medium, ClassicalFluid) else "quantum",
        include_self_gravity=medium.include_self_gravity,
        response_prescription=medium.response_prescription,
        metadata={
            "units": "G=c=hbar=1",
            "engine": "legacy_kspace_validation",
            "omega_tilde": orbit.physical_omega,
            "M": orbit.total_mass,
            "nu": orbit.nu,
            "a": orbit.a,
            "e": orbit.e,
            "n_max_safety": config.n_max,
            "n_max_evaluated": n_max_evaluated,
            "harmonic_termination": "tail" if terminated_by_tail else "safety_cap",
            "n_ell": config.n_ell,
            "n_mu": config.n_mu,
            "n_phi": config.n_phi,
            "backend": _resolved_backend(config),
            "radial_order": config.radial_order,
            "k_min": config.k_min,
            "k_max": k_max,
            "highest_retained_propagating_root": medium.roots(n_max_evaluated * orbit.physical_omega)[0],
            "highest_retained_pole_within_k_max": medium.roots(n_max_evaluated * orbit.physical_omega)[0] < k_max,
            "source_size": config.source_size,
            "source_window": config.source_window,
            "frequency_weights": {"n=0": 1, "n>=1": 2},
            "harmonic_tail": harmonic_tail_metadata,
            "response_interpretation": (
                "time-symmetric conservative PV prescription in a Jeans-unstable homogeneous background"
                if medium.include_self_gravity
                else "conservative/Hermitian part of the stable retarded response"
            ),
            "self_cross_closure": {
                "static": None if static_values is None else float(static_values[0] - sum(static_values[1:])),
                "osc": None if osc_values is None else float(osc_values[0] - sum(osc_values[1:])),
                "total": None if total_values is None else float(total_values[0] - sum(total_values[1:])),
            },
            "static_sector_note": (
                "pure quantum-pressure static sector requires a finite-background or IR-matching prescription"
                if static_ir
                else "included using the selected time-symmetric PV prescription"
            ),
            "max_k_over_mphi": (
                None if not isinstance(medium, QuantumFluid) else k_max / medium.m_phi
            ),
            "max_omega_over_mphi": (
                None
                if not isinstance(medium, QuantumFluid)
                else n_max_evaluated * orbit.physical_omega / medium.m_phi
            ),
        },
    )


def _fit_circular_limit(
    orbit: BinaryOrbit,
    medium: ConservativeMedium,
    config: PrecessionConfig,
) -> PrecessionResult:
    sample_e = np.array((config.e_switch, 1.5 * config.e_switch, 2.0 * config.e_switch), dtype=np.float64)
    samples = [_calculate_non_circular(replace(orbit, e=float(e)), medium, config) for e in sample_e]

    def fit_field(name: str) -> tuple[float | None, float]:
        values = [getattr(sample, name) for sample in samples]
        if any(value is None for value in values):
            return None, math.inf
        coefficients_1 = np.polynomial.polynomial.polyfit(sample_e**2, np.asarray(values, dtype=np.float64), 1)
        coefficients_2 = np.polynomial.polynomial.polyfit(sample_e**2, np.asarray(values, dtype=np.float64), 2)
        return float(coefficients_2[0]), abs(float(coefficients_2[0] - coefficients_1[0]))

    fields = [
        "delta_varpi_static", "delta_varpi_osc", "delta_varpi_total",
        "delta_varpi_static_self_1", "delta_varpi_static_self_2", "delta_varpi_static_cross",
        "delta_varpi_osc_self_1", "delta_varpi_osc_self_2", "delta_varpi_osc_cross",
        "delta_varpi_total_self_1", "delta_varpi_total_self_2", "delta_varpi_total_cross",
        "dH_static_de", "dH_osc_de", "dH_total_de",
    ]
    fitted = {field: fit_field(field) for field in fields}
    reference = samples[1]
    anomaly_error = max(error for _, error in fitted.values() if math.isfinite(error))
    metadata = dict(reference.metadata)
    metadata["circular_limit_fit_e"] = sample_e.tolist()
    metadata["literal_periapsis_defined"] = False
    return PrecessionResult(
        **{field: fitted[field][0] for field in fields},
        harmonics=reference.harmonics,
        error_radial=reference.error_radial,
        error_harmonic=reference.error_harmonic,
        error_anomaly=anomaly_error,
        error_angular=reference.error_angular,
        error_uv=reference.error_uv,
        error_static_ir=reference.error_static_ir,
        error_total=(
            None
            if fitted["delta_varpi_total"][0] is None or reference.error_harmonic is None
            else math.sqrt(reference.error_radial**2 + reference.error_harmonic**2 + anomaly_error**2)
        ),
        harmonic_converged=all(sample.harmonic_converged for sample in samples),
        static_ir_status=reference.static_ir_status,
        static_sector_included=reference.static_sector_included,
        quantity="circular_limit_precession",
        literal_periapsis_defined=False,
        perturbative_valid=(None if fitted["delta_varpi_total"][0] is None else abs(fitted["delta_varpi_total"][0]) <= config.perturbative_max_shift),
        model=reference.model,
        include_self_gravity=reference.include_self_gravity,
        response_prescription=reference.response_prescription,
        metadata=metadata,
    )


def calculate_precession(
    orbit: BinaryOrbit,
    medium: ConservativeMedium,
    config: PrecessionConfig | None = None,
) -> PrecessionResult:

    config = PrecessionConfig() if config is None else config
    if config.engine == "analytic":
        from .analytic_precession import calculate_precession_analytic

        return calculate_precession_analytic(orbit, medium, config)
    if orbit.e < config.e_switch:
        return _fit_circular_limit(orbit, medium, config)
    return _calculate_non_circular(orbit, medium, config)


def classical_precession(
    orbit: BinaryOrbit,
    fluid: ClassicalFluid,
    config: PrecessionConfig | None = None,
) -> PrecessionResult:
    return calculate_precession(orbit, fluid, config)


def quantum_precession(
    orbit: BinaryOrbit,
    fluid: QuantumFluid,
    config: PrecessionConfig | None = None,
) -> PrecessionResult:
    return calculate_precession(orbit, fluid, config)
