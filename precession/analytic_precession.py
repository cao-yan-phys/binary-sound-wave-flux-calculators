
from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .classical_closed_sum import classical_pair_average
from .conservative_kernels import ClassicalFluid, ConservativeMedium, QuantumFluid, static_ir_is_prescription_dependent
from .offshell_source import cuda_available
from .pair_orbit import PairOrbitGeometry, build_pair_orbit_geometry
from .realspace_cuda import CudaClassicalSelfGravityCorrection, CudaQuantumPairAverage
from .realspace_kernels import (
    classical_gprime_n,
    quantum_gprime_n,
    quantum_static_no_sg_negative_cs2_gprime,
    quantum_static_no_sg_finite_cs_gprime,
)

if TYPE_CHECKING:
    from .periastron_precession import HarmonicContribution, PrecessionConfig, PrecessionResult
    from .orbit_derivatives import BinaryOrbit


def _precession_prefactor(orbit: "BinaryOrbit") -> float:
    return -math.pi * orbit.a * math.sqrt(1.0 - orbit.e * orbit.e) / (orbit.nu * orbit.e)


def _parts_from_radial_function(
    geometry: PairOrbitGeometry,
    radial_derivative,
    phase: np.ndarray | float = 1.0,
) -> np.ndarray:
    return geometry.derivative_parts(radial_derivative, phase)


def _closed_classical_parts(
    geometry: PairOrbitGeometry,
    orbit: "BinaryOrbit",
    fluid: ClassicalFluid,
) -> np.ndarray:
    return classical_pair_average(orbit, rho_bar=fluid.rho_bar, c_s=fluid.c_s, n_ell=len(geometry.grid.ell))[0]


def _harmonic_tail(rows: list["HarmonicContribution"], config: "PrecessionConfig", response_scale: float = 1.0) -> tuple[bool, float | None, dict[str, object]]:
    osc_rows = [row for row in rows if row.n > 0 and row.delta_varpi is not None]
    if not osc_rows:
        return True, 0.0, {"mode": "not_applicable"}
    if len(osc_rows) < 2 * config.tail_window:
        return False, None, {"reason": "fewer than two complete harmonic blocks", "block_size": config.tail_window}
    recent = np.asarray(
        [[row.delta_varpi, row.self_1, row.self_2, row.cross] for row in osc_rows[-config.tail_window :]],
        dtype=np.float64,
    )
    previous = np.asarray(
        [[row.delta_varpi, row.self_1, row.self_2, row.cross] for row in osc_rows[-2 * config.tail_window : -config.tail_window]],
        dtype=np.float64,
    )
    accumulated = np.sum(np.abs(np.asarray(
        [[row.delta_varpi, row.self_1, row.self_2, row.cross] for row in osc_rows], dtype=np.float64
    )), axis=0)
    static = next((row for row in rows if row.n == 0 and row.delta_varpi is not None), None)
    static_scale = np.zeros(4) if static is None else np.abs([static.delta_varpi, static.self_1, static.self_2, static.cross])
    scale = np.maximum(np.maximum(accumulated, static_scale), np.finfo(float).tiny)
    recent_absolute = np.sum(np.abs(recent), axis=0)
    signed_change = np.abs(np.sum(recent, axis=0) - np.sum(previous, axis=0))
    converged = bool(
        np.all(recent_absolute <= config.tail_atol * response_scale + config.tail_rtol * scale)
        and np.all(signed_change <= config.tail_atol * response_scale + config.tail_rtol * scale)
    )
    return converged, float(max(recent_absolute[0], signed_change[0])), {
        "block_size": config.tail_window,
        "recent_absolute_total_self1_self2_cross": recent_absolute.tolist(),
        "signed_block_change_total_self1_self2_cross": signed_change.tolist(),
        "relative_recent_absolute_total_self1_self2_cross": (recent_absolute / scale).tolist(),
        "relative_signed_block_change_total_self1_self2_cross": (signed_change / scale).tolist(),
        "consecutive_windows_required": config.consecutive_windows,
    }


def _build_result(
    orbit: "BinaryOrbit",
    medium: ConservativeMedium,
    config: "PrecessionConfig",
    *,
    static_parts: np.ndarray | None,
    osc_parts: np.ndarray | None,
    rows: list["HarmonicContribution"],
    harmonic_converged: bool,
    harmonic_error: float | None,
    harmonic_metadata: dict[str, object],
    analytic_mode: str,
    static_ir: bool,
    realspace_backend: str = "numpy_realspace",
) -> "PrecessionResult":
    from .periastron_precession import PrecessionResult

    static_included = static_parts is not None
    if config.sector == "static":
        total_parts = static_parts
    elif config.sector == "osc":
        total_parts = osc_parts
    elif static_parts is None:
        total_parts = None
    else:
        total_parts = static_parts + (np.zeros(4, dtype=np.float64) if osc_parts is None else osc_parts)

    def scalar(parts: np.ndarray | None, index: int) -> float | None:
        return None if parts is None else float(parts[index])

    gauss_prefactor = -2.0 * math.pi * math.sqrt(1.0 - orbit.e * orbit.e) / (
        orbit.mu * orbit.a * orbit.a * orbit.physical_omega**2 * orbit.e
    )

    def d_h(parts: np.ndarray | None) -> float | None:
        return None if parts is None else float(parts[0] / gauss_prefactor)

    d_h_static = d_h(static_parts)
    d_h_osc = d_h(osc_parts)
    d_h_total = d_h(total_parts)
    if config.strict_convergence and config.sector != "static" and not harmonic_converged:
        raise RuntimeError("analytic harmonic sum did not converge before the n_max safety cap; increase n_max")
    error_total = harmonic_error
    result_value = scalar(total_parts, 0)
    return PrecessionResult(
        delta_varpi_static=scalar(static_parts, 0),
        delta_varpi_osc=scalar(osc_parts, 0),
        delta_varpi_total=result_value,
        delta_varpi_static_self_1=scalar(static_parts, 1),
        delta_varpi_static_self_2=scalar(static_parts, 2),
        delta_varpi_static_cross=scalar(static_parts, 3),
        delta_varpi_osc_self_1=scalar(osc_parts, 1),
        delta_varpi_osc_self_2=scalar(osc_parts, 2),
        delta_varpi_osc_cross=scalar(osc_parts, 3),
        delta_varpi_total_self_1=scalar(total_parts, 1),
        delta_varpi_total_self_2=scalar(total_parts, 2),
        delta_varpi_total_cross=scalar(total_parts, 3),
        dH_static_de=d_h_static,
        dH_osc_de=d_h_osc,
        dH_total_de=d_h_total,
        harmonics=tuple(rows),
        error_radial=0.0,
        error_harmonic=harmonic_error,
        error_anomaly=math.inf,
        error_angular=0.0,
        error_uv=0.0,
        error_static_ir=math.inf if static_ir else None,
        error_total=error_total,
        harmonic_converged=harmonic_converged,
        static_ir_status="PRESCRIPTION_DEPENDENT" if static_ir else "FINITE_UNDER_SELECTED_PRESCRIPTION",
        static_sector_included=static_included,
        quantity="periastron_precession",
        literal_periapsis_defined=True,
        perturbative_valid=None if result_value is None else abs(result_value) <= config.perturbative_max_shift,
        model="classical" if isinstance(medium, ClassicalFluid) else "quantum",
        include_self_gravity=medium.include_self_gravity,
        response_prescription=medium.response_prescription,
        metadata={
            "units": "G=c=hbar=1",
            "engine": "analytic_realspace",
            "analytic_mode": analytic_mode,
            "omega_tilde": orbit.physical_omega,
            "M": orbit.total_mass,
            "nu": orbit.nu,
            "a": orbit.a,
            "e": orbit.e,
            "n_ell": config.n_ell,
            "orbit_quadrature": "uniform mean-anomaly double average",
            "n_max_safety": None if analytic_mode == "closed_all_harmonics" else config.n_max,
            "n_max_evaluated": None if analytic_mode == "closed_all_harmonics" else max((row.n for row in rows), default=0),
            "harmonic_termination": "analytic_closed_sum" if analytic_mode == "closed_all_harmonics" else ("tail" if harmonic_converged else "safety_cap"),
            "harmonic_tail": harmonic_metadata,
            "source_size": None,
            "source_window": None,
            "k_min": None,
            "k_max": None,
            "radial_order": None,
            "n_mu": None,
            "n_phi": None,
            "backend": realspace_backend,
            "frequency_weights": {"n=0": 1, "n>=1": 2},
            "self_cross_closure": {
                "static": None if static_parts is None else float(static_parts[0] - sum(static_parts[1:])),
                "osc": None if osc_parts is None else float(osc_parts[0] - sum(osc_parts[1:])),
                "total": None if total_parts is None else float(total_parts[0] - sum(total_parts[1:])),
            },
            "response_interpretation": (
                "time-symmetric conservative PV prescription in a Jeans-unstable homogeneous background"
                if medium.include_self_gravity
                else "exact point-source real-space conservative kernel"
            ),
            "static_sector_note": (
                "pure quantum-pressure static sector requires a finite-background or IR-matching prescription"
                if static_ir
                else "included analytically in the real-space kernel"
            ),
        },
    )


def _per_harmonic(
    orbit: "BinaryOrbit",
    medium: ConservativeMedium,
    config: "PrecessionConfig",
) -> "PrecessionResult":
    from .periastron_precession import HarmonicContribution

    geometry = build_pair_orbit_geometry(orbit, config.n_ell)
    prefactor = _precession_prefactor(orbit)
    static_ir = static_ir_is_prescription_dependent(medium)
    cuda_quantum: CudaQuantumPairAverage | None = None
    realspace_backend = "numpy_realspace"
    if isinstance(medium, QuantumFluid):
        if config.backend == "cuda":
            if not cuda_available():
                raise RuntimeError("CUDA backend requested, but numba.cuda is unavailable")
            cuda_quantum = CudaQuantumPairAverage(geometry)
            realspace_backend = "cuda_realspace"
        elif config.backend == "auto" and cuda_available():
            cuda_quantum = CudaQuantumPairAverage(geometry)
            realspace_backend = "cuda_realspace"
    static_parts: np.ndarray | None = None
    osc_parts: np.ndarray | None = None
    rows: list[HarmonicContribution] = []
    n_values = range(0, config.n_max + 1) if config.sector == "total" else (range(0, 1) if config.sector == "static" else range(1, config.n_max + 1))
    consecutive = 0
    terminated = False
    response_scale = 4.0 * math.pi * medium.rho_bar / orbit.physical_omega**2
    for n in n_values:
        if n == 0 and static_ir:
            rows.append(
                HarmonicContribution(
                    n=0, frequency_weight=1, sector="static", delta_varpi=None, self_1=None, self_2=None, cross=None,
                    radial_error=None, k_pole=0.0, kappa_evanescent=None, radial_subtraction=False,
                )
            )
            continue
        omega = n * orbit.physical_omega
        if cuda_quantum is not None:
            pair_parts = cuda_quantum.pair_parts(
                n,
                omega,
                medium.rho_bar,
                medium.m_phi,
                medium.c_S_squared,
                medium.include_self_gravity,
            )
        elif isinstance(medium, ClassicalFluid):
            derivative = lambda radius, omega=omega: classical_gprime_n(medium, omega, radius)
        elif n == 0 and not medium.include_self_gravity:
            derivative = (
                (lambda radius: quantum_static_no_sg_finite_cs_gprime(medium, radius))
                if medium.c_S_squared > 0.0
                else (lambda radius: quantum_static_no_sg_negative_cs2_gprime(medium, radius))
            )
        else:
            derivative = lambda radius, omega=omega: quantum_gprime_n(medium, omega, radius)
        phase: np.ndarray | float = 1.0 if n == 0 else np.cos(n * geometry.delta)
        if cuda_quantum is None:
            pair_parts = _parts_from_radial_function(geometry, derivative, phase)
        weight = 1 if n == 0 else 2
        delta_parts = prefactor * weight * pair_parts
        pole, kappa = medium.roots(omega)
        row = HarmonicContribution(
            n=n,
            frequency_weight=weight,
            sector="static" if n == 0 else "osc",
            delta_varpi=float(delta_parts[0]),
            self_1=float(delta_parts[1]),
            self_2=float(delta_parts[2]),
            cross=float(delta_parts[3]),
            radial_error=0.0,
            k_pole=pole,
            kappa_evanescent=kappa,
            radial_subtraction=False,
        )
        rows.append(row)
        if n == 0:
            static_parts = delta_parts
        else:
            osc_parts = delta_parts if osc_parts is None else osc_parts + delta_parts
            converged, _, _ = _harmonic_tail(rows, config, response_scale)
            if converged:
                consecutive += 1
                if consecutive >= config.consecutive_windows:
                    terminated = True
                    break
            else:
                consecutive = 0
    _, harmonic_error, tail_metadata = _harmonic_tail(rows, config, response_scale)
    harmonic_converged = config.sector == "static" or terminated
    return _build_result(
        orbit,
        medium,
        config,
        static_parts=static_parts,
        osc_parts=osc_parts,
        rows=rows,
        harmonic_converged=harmonic_converged,
        harmonic_error=harmonic_error,
        harmonic_metadata=tail_metadata,
        analytic_mode="realspace_per_harmonic",
        static_ir=static_ir,
        realspace_backend=realspace_backend,
    )


def _classical_closed_sum(
    orbit: "BinaryOrbit",
    fluid: ClassicalFluid,
    config: "PrecessionConfig",
) -> "PrecessionResult":
    from .periastron_precession import HarmonicContribution

    prefactor = _precession_prefactor(orbit)
    total_pair_parts, static_pair_parts = classical_pair_average(
        orbit, rho_bar=fluid.rho_bar, c_s=fluid.c_s, n_ell=config.n_ell,
    )
    static_parts = prefactor * static_pair_parts if config.sector in ("total", "static") else None
    total_parts = prefactor * total_pair_parts
    if config.sector == "osc":
        osc_parts = total_parts - prefactor * static_pair_parts
    elif config.sector == "static":
        osc_parts = None
    else:
        osc_parts = total_parts - static_parts
    static_row = HarmonicContribution(
        n=0,
        frequency_weight=1,
        sector="static",
        delta_varpi=float(prefactor * static_pair_parts[0]),
        self_1=float(prefactor * static_pair_parts[1]),
        self_2=float(prefactor * static_pair_parts[2]),
        cross=float(prefactor * static_pair_parts[3]),
        radial_error=0.0,
        k_pole=0.0,
        kappa_evanescent=None,
        radial_subtraction=False,
    )
    return _build_result(
        orbit,
        fluid,
        config,
        static_parts=static_parts,
        osc_parts=osc_parts,
        rows=[static_row],
        harmonic_converged=True,
        harmonic_error=0.0,
        harmonic_metadata={"mode": "closed_all_harmonics"},
        analytic_mode="closed_all_harmonics",
        static_ir=False,
    )


def _classical_self_gravity_closed_correction(
    orbit: "BinaryOrbit",
    fluid: ClassicalFluid,
    config: "PrecessionConfig",
) -> "PrecessionResult":

    from .periastron_precession import HarmonicContribution

    geometry = build_pair_orbit_geometry(orbit, config.n_ell)
    prefactor = _precession_prefactor(orbit)
    acoustic = ClassicalFluid(
        rho_bar=fluid.rho_bar,
        c_s=fluid.c_s,
        include_self_gravity=False,
        response_prescription=fluid.response_prescription,
    )
    cuda_correction: CudaClassicalSelfGravityCorrection | None = None
    realspace_backend = "numpy_realspace"
    if config.backend == "cuda":
        if not cuda_available():
            raise RuntimeError("CUDA backend requested, but numba.cuda is unavailable")
        cuda_correction = CudaClassicalSelfGravityCorrection(geometry)
        realspace_backend = "cuda_realspace"
    elif config.backend == "auto" and cuda_available():
        cuda_correction = CudaClassicalSelfGravityCorrection(geometry)
        realspace_backend = "cuda_realspace"
    acoustic_static_pair_parts = _parts_from_radial_function(
        geometry,
        lambda radius: classical_gprime_n(acoustic, 0.0, radius),
    )
    rows: list[HarmonicContribution] = []
    static_parts: np.ndarray | None = None
    if config.sector in ("total", "static"):
        if cuda_correction is None:
            static_pair_parts = _parts_from_radial_function(
                geometry,
                lambda radius: classical_gprime_n(fluid, 0.0, radius),
            )
        else:
            static_pair_parts = acoustic_static_pair_parts + cuda_correction.correction_parts(
                0,
                0.0,
                fluid.rho_bar,
                fluid.c_s,
            )
        static_parts = prefactor * static_pair_parts
        pole, kappa = fluid.roots(0.0)
        rows.append(
            HarmonicContribution(
                n=0,
                frequency_weight=1,
                sector="static",
                delta_varpi=float(static_parts[0]),
                self_1=float(static_parts[1]),
                self_2=float(static_parts[2]),
                cross=float(static_parts[3]),
                radial_error=0.0,
                k_pole=pole,
                kappa_evanescent=kappa,
                radial_subtraction=False,
            )
        )
    if config.sector == "static":
        return _build_result(
            orbit,
            fluid,
            config,
            static_parts=static_parts,
            osc_parts=None,
            rows=rows,
            harmonic_converged=True,
            harmonic_error=0.0,
            harmonic_metadata={"mode": "static_exact"},
            analytic_mode="closed_acoustic_plus_self_gravity_correction",
            static_ir=False,
            realspace_backend=realspace_backend,
        )

    acoustic_total_pair_parts = _closed_classical_parts(geometry, orbit, acoustic)
    osc_parts = prefactor * (acoustic_total_pair_parts - acoustic_static_pair_parts)
    correction_parts = np.zeros(4, dtype=np.float64)
    consecutive = 0
    terminated = False
    response_scale = 4.0 * math.pi * fluid.rho_bar / orbit.physical_omega**2
    for n in range(1, config.n_max + 1):
        omega = n * orbit.physical_omega
        if cuda_correction is None:
            derivative = lambda radius, omega=omega: (
                classical_gprime_n(fluid, omega, radius) - classical_gprime_n(acoustic, omega, radius)
            )
            pair_parts = _parts_from_radial_function(geometry, derivative, np.cos(n * geometry.delta))
        else:
            pair_parts = cuda_correction.correction_parts(n, omega, fluid.rho_bar, fluid.c_s)
        delta_parts = 2.0 * prefactor * pair_parts
        correction_parts += delta_parts
        pole, kappa = fluid.roots(omega)
        rows.append(
            HarmonicContribution(
                n=n,
                frequency_weight=2,
                sector="osc",
                delta_varpi=float(delta_parts[0]),
                self_1=float(delta_parts[1]),
                self_2=float(delta_parts[2]),
                cross=float(delta_parts[3]),
                radial_error=0.0,
                k_pole=pole,
                kappa_evanescent=kappa,
                radial_subtraction=False,
            )
        )
        converged, _, _ = _harmonic_tail(rows, config, response_scale)
        if converged:
            consecutive += 1
            if consecutive >= config.consecutive_windows:
                terminated = True
                break
        else:
            consecutive = 0
    _, harmonic_error, harmonic_metadata = _harmonic_tail(rows, config, response_scale)
    harmonic_converged = terminated
    harmonic_metadata["summand"] = "self_gravity_minus_acoustic_reference"
    return _build_result(
        orbit,
        fluid,
        config,
        static_parts=static_parts,
        osc_parts=osc_parts + correction_parts,
        rows=rows,
        harmonic_converged=harmonic_converged,
        harmonic_error=harmonic_error,
        harmonic_metadata=harmonic_metadata,
        analytic_mode="closed_acoustic_plus_self_gravity_correction",
        static_ir=False,
        realspace_backend=realspace_backend,
    )


def _fit_circular_limit_analytic(
    orbit: "BinaryOrbit",
    medium: ConservativeMedium,
    config: "PrecessionConfig",
) -> "PrecessionResult":
    from .periastron_precession import PrecessionResult

    sample_e = np.array((config.e_switch, 1.5 * config.e_switch, 2.0 * config.e_switch), dtype=np.float64)
    samples = [_calculate_analytic(replace(orbit, e=float(e)), medium, config) for e in sample_e]
    fields = (
        "delta_varpi_static", "delta_varpi_osc", "delta_varpi_total",
        "delta_varpi_static_self_1", "delta_varpi_static_self_2", "delta_varpi_static_cross",
        "delta_varpi_osc_self_1", "delta_varpi_osc_self_2", "delta_varpi_osc_cross",
        "delta_varpi_total_self_1", "delta_varpi_total_self_2", "delta_varpi_total_cross",
        "dH_static_de", "dH_osc_de", "dH_total_de",
    )
    fitted: dict[str, tuple[float | None, float]] = {}
    for field in fields:
        values = [getattr(sample, field) for sample in samples]
        if any(value is None for value in values):
            fitted[field] = (None, math.inf)
            continue
        linear = np.polynomial.polynomial.polyfit(sample_e * sample_e, np.asarray(values, dtype=np.float64), 1)
        quadratic = np.polynomial.polynomial.polyfit(sample_e * sample_e, np.asarray(values, dtype=np.float64), 2)
        fitted[field] = (float(quadratic[0]), abs(float(quadratic[0] - linear[0])))
    reference = samples[1]
    metadata = dict(reference.metadata)
    metadata["circular_limit_fit_e"] = sample_e.tolist()
    metadata["literal_periapsis_defined"] = False
    anomaly_error = max(error for _, error in fitted.values() if math.isfinite(error))
    return PrecessionResult(
        **{field: fitted[field][0] for field in fields},
        harmonics=reference.harmonics,
        error_radial=0.0,
        error_harmonic=reference.error_harmonic,
        error_anomaly=anomaly_error,
        error_angular=0.0,
        error_uv=0.0,
        error_static_ir=reference.error_static_ir,
        error_total=None if fitted["delta_varpi_total"][0] is None else math.sqrt((reference.error_harmonic or 0.0) ** 2 + anomaly_error**2),
        harmonic_converged=all(sample.harmonic_converged for sample in samples),
        static_ir_status=reference.static_ir_status,
        static_sector_included=reference.static_sector_included,
        quantity="circular_limit_precession",
        literal_periapsis_defined=False,
        perturbative_valid=None if fitted["delta_varpi_total"][0] is None else abs(fitted["delta_varpi_total"][0]) <= config.perturbative_max_shift,
        model=reference.model,
        include_self_gravity=reference.include_self_gravity,
        response_prescription=reference.response_prescription,
        metadata=metadata,
    )


def _calculate_analytic(orbit: "BinaryOrbit", medium: ConservativeMedium, config: "PrecessionConfig") -> "PrecessionResult":
    if isinstance(medium, ClassicalFluid):
        if medium.include_self_gravity:
            return _classical_self_gravity_closed_correction(orbit, medium, config)
        return _classical_closed_sum(orbit, medium, config)
    return _per_harmonic(orbit, medium, config)


def _calculate_with_circular_limit(
    orbit: "BinaryOrbit",
    medium: ConservativeMedium,
    config: "PrecessionConfig",
) -> "PrecessionResult":
    if orbit.e < config.e_switch:
        return _fit_circular_limit_analytic(orbit, medium, config)
    return _calculate_analytic(orbit, medium, config)


def calculate_precession_analytic(
    orbit: "BinaryOrbit",
    medium: ConservativeMedium,
    config: "PrecessionConfig",
) -> "PrecessionResult":

    if config.source_size != 0.0:
        raise ValueError("analytic_realspace is point-source only; use engine='legacy_kspace_validation' for a finite source window")
    response_scale = 4.0 * math.pi * medium.rho_bar / orbit.physical_omega**2
    inner_config = replace(config, strict_convergence=False)
    result = _calculate_with_circular_limit(orbit, medium, inner_config)
    history = []
    error = math.inf
    converged = False

    def values(sample):
        return np.asarray([
            value for name, value in vars(sample).items()
            if name.startswith("delta_varpi_") and value is not None
        ], dtype=np.float64)

    history.append({"n_ell": config.n_ell, "delta_varpi_total": result.delta_varpi_total, "error": None})
    for refinement in range(1, config.anomaly_max_refinements + 1):
        next_n = config.n_ell * 2**refinement
        if next_n > config.anomaly_max_n_ell:
            break
        refined_config = replace(inner_config, n_ell=next_n)
        refined = _calculate_with_circular_limit(orbit, medium, refined_config)
        old, new = values(result), values(refined)
        differences = np.abs(new - old)
        error = float(np.max(differences, initial=0.0))
        converged = bool(np.all(differences <= config.anomaly_atol * response_scale + config.anomaly_rtol * np.abs(new)))
        history.append({"n_ell": refined_config.n_ell, "delta_varpi_total": refined.delta_varpi_total, "error": error})
        result = refined
        if converged:
            break
    metadata = dict(result.metadata)
    metadata["anomaly_refinement"] = history
    metadata["anomaly_termination"] = "tolerance" if converged else "safety_cap"
    metadata["response_eta"] = response_scale
    metadata["orbital_perturbation_only"] = result.perturbative_valid
    if isinstance(medium, ClassicalFluid):
        mach = orbit.a * orbit.physical_omega / medium.c_s
        factor = math.sqrt((1.0 + orbit.e) / (1.0 - orbit.e))
        metadata["maximum_body_mach"] = [orbit.f2 * mach * factor, orbit.f1 * mach * factor]
        metadata["orbital_scale_source_strength"] = [orbit.f1 * mach**2, orbit.f2 * mach**2]
    if isinstance(medium, ClassicalFluid):
        metadata["orbit_quadrature"] = "wavefront-split Kepler quadrature plus mean-anomaly harmonic correction"
    error = max(error, result.error_anomaly if math.isfinite(result.error_anomaly) else 0.0)
    result = replace(
        result,
        anomaly_converged=converged,
        n_ell_used=int(metadata["n_ell"]),
        error_anomaly=error,
        error_total=None if result.delta_varpi_total is None or result.error_harmonic is None else math.hypot(error, result.error_harmonic),
        metadata=metadata,
    )
    if config.strict_convergence and not result.harmonic_converged:
        raise RuntimeError("analytic harmonic sum did not converge before the n_max safety cap")
    if config.strict_convergence and not converged:
        raise RuntimeError("orbital quadrature did not converge before the anomaly refinement safety cap")
    return result
