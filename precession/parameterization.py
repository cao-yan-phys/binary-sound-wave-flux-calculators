
from __future__ import annotations

import math
from dataclasses import replace

from .periastron_precession import PrecessionConfig, PrecessionResult, classical_precession, quantum_precession
from .conservative_kernels import ClassicalFluid, QuantumFluid
from .orbit_derivatives import BinaryOrbit


def _mass_fractions_from_nu(nu: float) -> tuple[float, float]:
    if not (0.0 < nu <= 0.25):
        raise ValueError("nu must satisfy 0 < nu <= 1/4")
    delta = math.sqrt(max(0.0, 1.0 - 4.0 * nu))
    return 0.5 * (1.0 + delta), 0.5 * (1.0 - delta)


def _orbit_from_flux_parameters(nu: float, e: float) -> BinaryOrbit:
    q1, q2 = _mass_fractions_from_nu(nu)
    return BinaryOrbit(m1=q1, m2=q2, a=1.0, e=e)


def _annotate(result: PrecessionResult, parameters: dict[str, float | bool | str]) -> PrecessionResult:
    metadata = dict(result.metadata)
    metadata["flux_parameterization"] = parameters
    metadata["response_eta"] = parameters["n0"]**2
    metadata["propagation_n0"] = parameters["n0"] if parameters["include_self_gravity"] else 0.0
    return replace(result, metadata=metadata)


def classical_precession_flux_parameters(
    *,
    nu: float,
    e: float,
    n0: float,
    A: float,
    include_self_gravity: bool = True,
    config: PrecessionConfig | None = None,
) -> PrecessionResult:

    if not math.isfinite(n0) or n0 < 0.0:
        raise ValueError("n0 must be non-negative")
    if not math.isfinite(A) or A <= 0.0:
        raise ValueError("A = a*Omega for a classical fluid must be positive")
    orbit = _orbit_from_flux_parameters(nu, e)
    fluid = ClassicalFluid(
        rho_bar=n0 * n0 / (4.0 * math.pi),
        c_s=1.0 / A,
        include_self_gravity=include_self_gravity,
    )
    result = classical_precession(orbit, fluid, config)
    return _annotate(
        result,
        {
            "model": "classical",
            "nu": nu,
            "e": e,
            "n0": n0,
            "A": A,
            "A_definition": "A = a*Omega = a*tildeOmega/c_s",
            "include_self_gravity": include_self_gravity,
        },
    )


def quantum_precession_flux_parameters(
    *,
    nu: float,
    e: float,
    n0: float,
    A: float,
    cS2_over_Omega: float = 0.0,
    include_self_gravity: bool = True,
    config: PrecessionConfig | None = None,
) -> PrecessionResult:

    if not math.isfinite(n0) or n0 < 0.0:
        raise ValueError("n0 must be non-negative")
    if not math.isfinite(A) or A <= 0.0:
        raise ValueError("A = a*sqrt(Omega) for a quantum fluid must be positive")
    if not math.isfinite(cS2_over_Omega):
        raise ValueError("cS2_over_Omega must be finite")
    orbit = _orbit_from_flux_parameters(nu, e)
    omega_auxiliary = A * A
    fluid = QuantumFluid(
        rho_bar=n0 * n0 / (4.0 * math.pi),
        m_phi=0.5 * omega_auxiliary,
        c_S_squared=cS2_over_Omega * omega_auxiliary,
        include_self_gravity=include_self_gravity,
    )
    result = quantum_precession(orbit, fluid, config)
    return _annotate(
        result,
        {
            "model": "quantum",
            "nu": nu,
            "e": e,
            "n0": n0,
            "A": A,
            "A_definition": "A = a*sqrt(Omega)",
            "cS2_over_Omega": cS2_over_Omega,
            "include_self_gravity": include_self_gravity,
        },
    )
