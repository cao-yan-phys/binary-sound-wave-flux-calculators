
from __future__ import annotations

import argparse
from pathlib import Path

from precession import (
    PrecessionConfig,
    classical_precession_flux_parameters,
    quantum_precession_flux_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("classical", "quantum"), required=True)
    parser.add_argument("--nu", type=float, required=True, help="nu = m1*m2/(m1+m2)^2")
    parser.add_argument("--e", type=float, required=True)
    parser.add_argument("--n0", type=float, required=True, help="n0 = m/Omega")
    parser.add_argument("--A", type=float, required=True, help="a*Omega (classical fluid) or a*sqrt(Omega) (quantum fluid)")
    parser.add_argument("--cS2-over-Omega", type=float, default=0.0, help="quantum-fluid c_S^2/Omega")
    parser.add_argument("--no-self-gravity", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--engine", choices=("analytic", "legacy_kspace_validation"), default="analytic")
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--n-ell", type=int, default=256)
    parser.add_argument("--n-mu", type=int, default=8)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--radial-order", type=int, default=16)
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--source-size", type=float, default=0.0)
    parser.add_argument("--source-window", choices=("none", "gaussian", "lorentzian"), default="gaussian")
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--anomaly-rtol", type=float, default=1.0e-3)
    parser.add_argument("--anomaly-atol", type=float, default=1.0e-5)
    parser.add_argument("--anomaly-max-refinements", type=int, default=3)
    parser.add_argument("--anomaly-max-n-ell", type=int, default=2048)
    parser.add_argument("--allow-unconverged", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PrecessionConfig(
        n_max=args.n_max,
        n_ell=args.n_ell,
        n_mu=args.n_mu,
        n_phi=args.n_phi,
        radial_order=args.radial_order,
        k_max=args.k_max,
        source_size=args.source_size,
        source_window=args.source_window,
        tail_rtol=args.rtol,
        tail_atol=args.atol,
        anomaly_rtol=args.anomaly_rtol,
        anomaly_atol=args.anomaly_atol,
        anomaly_max_refinements=args.anomaly_max_refinements,
        anomaly_max_n_ell=args.anomaly_max_n_ell,
        strict_convergence=not args.allow_unconverged,
        backend=args.backend,
        engine=args.engine,
    )
    common = {
        "nu": args.nu,
        "e": args.e,
        "n0": args.n0,
        "A": args.A,
        "include_self_gravity": not args.no_self_gravity,
        "config": config,
    }
    if args.model == "classical":
        result = classical_precession_flux_parameters(**common)
    else:
        result = quantum_precession_flux_parameters(
            **common,
            cS2_over_Omega=args.cS2_over_Omega,
        )
    if args.output_dir is not None:
        result.write(args.output_dir)
    print(f"delta_varpi_static = {result.delta_varpi_static}")
    print(f"delta_varpi_osc = {result.delta_varpi_osc}")
    print(f"delta_varpi_total = {result.delta_varpi_total}")
    print(f"delta_varpi_total_normalized = {result.delta_varpi_total_normalized}")
    print(f"engine = {result.metadata['engine']}")
    print(f"backend = {result.metadata['backend']}")
    print(f"static_ir_status = {result.static_ir_status}")
    print(f"harmonic_converged = {result.harmonic_converged}")
    print(f"harmonic_termination = {result.metadata['harmonic_termination']}")
    print(f"n_max_evaluated = {result.metadata['n_max_evaluated']}")
    print(f"error_radial = {result.error_radial}")
    print(f"error_harmonic = {result.error_harmonic}")
    print(f"anomaly_converged = {result.anomaly_converged}")
    print(f"n_ell_used = {result.n_ell_used}")
    print(f"error_anomaly = {result.error_anomaly}")


if __name__ == "__main__":
    main()
