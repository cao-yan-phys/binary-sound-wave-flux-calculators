
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classical_fluid import classical_fluid_force_y, classical_fluid_power, classical_fluid_tau_z
from edg_single_perturber_coefficients import edg_single_perturber_coefficients
from quadrupole_fluxes import classical_quadrupole_flux_normalized
from single_perturber_classical_fluid import (
    single_perturber_force_y,
    single_perturber_power,
    single_perturber_tau_z,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_plots"))
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--e", type=float, default=0.2)
    parser.add_argument("--nu-proxy", type=float, default=1.0e-5)
    parser.add_argument("--mach-min", type=float, default=0.05)
    parser.add_argument("--mach-max", type=float, default=0.78)
    parser.add_argument("--num-mach", type=int, default=26)
    parser.add_argument("--n-max", type=int, default=2048)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--n-mu", type=int, default=24)
    parser.add_argument("--n-phi", type=int, default=48)
    parser.add_argument("--xi-per-n", type=int, default=4)
    parser.add_argument("--edg-lmax", type=int, default=20)
    parser.add_argument("--edg-jmax", type=int, default=20)
    parser.add_argument("--edg-points", type=int, default=8)
    parser.add_argument("--edg-n-xi", type=int, default=8192)
    return parser.parse_args()


def uv_mach_limit(e: float) -> float:
    return math.sqrt((1.0 - e) / (1.0 + e))


def compute_full_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    mach_values = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    emri_scale = args.nu_proxy**2
    total = len(mach_values) * 2
    index = 0
    for n0 in (0.0, 1.0):
        for mach in mach_values:
            index += 1
            print(f"[numerical {index}/{total}] n0={n0:g} Mach={mach:.6g}", flush=True)
            common = dict(
                nu=args.nu_proxy,
                e=args.e,
                n0=n0,
                A=float(mach),
                n_max=args.n_max,
                n_mu=args.n_mu,
                n_phi=args.n_phi,
                backend=args.backend,
                chunk_size=64,
                rtol=args.rtol,
                tail_window=32,
                consecutive_windows=2,
                strict_convergence=True,
                speed_threshold_guard=True,
                xi_per_n=args.xi_per_n,
            )
            p = classical_fluid_power(**common)
            tau = classical_fluid_tau_z(**common)
            fy = classical_fluid_force_y(**common)
            rows.append(
                {
                    "kind": "full_binary_emri",
                    "e": float(args.e),
                    "n0": float(n0),
                    "Mach": float(mach),
                    "P_hat": float(p.value / emri_scale),
                    "tau_hat": float(tau.value / emri_scale),
                    "F_y_hat": float(fy.value / emri_scale),
                    "minus_F_y_hat": float(-fy.value / emri_scale),
                    "P_converged": bool(p.converged),
                    "tau_converged": bool(tau.converged),
                    "Fy_converged": bool(fy.converged),
                    "P_n": int(p.n_values[-1]),
                    "tau_n": int(tau.n_values[-1]),
                    "Fy_n": int(fy.n_values[-1]),
                    "P_tail": float(p.tail_ratio),
                    "tau_tail": float(tau.tail_ratio),
                    "Fy_tail": float(fy.tail_ratio),
                }
            )
    return pd.DataFrame(rows)


def compute_quadrupole_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str | None]] = []
    mach_values = np.linspace(args.mach_min, args.mach_max, 220)
    emri_scale = args.nu_proxy**2
    for n0 in (0.0, 1.0):
        for mach in mach_values:
            q = classical_quadrupole_flux_normalized(
                nu=args.nu_proxy,
                e=args.e,
                n0=n0,
                A=float(mach),
                n_max=1024,
                rtol=1.0e-12,
                strict_convergence=False,
                warning_ak=0.3,
            )
            rows.append(
                {
                    "kind": "quadrupole",
                    "e": float(args.e),
                    "n0": float(n0),
                    "Mach": float(mach),
                    "P_hat": float(q.P / emri_scale),
                    "tau_hat": float(q.tau_z_tildeOmega / emri_scale),
                    "F_y_hat": np.nan,
                    "minus_F_y_hat": np.nan,
                    "P_converged": bool(q.converged),
                    "tau_converged": bool(q.converged),
                    "Fy_converged": False,
                    "P_n": int(q.n_max_evaluated),
                    "tau_n": int(q.n_max_evaluated),
                    "Fy_n": 0,
                    "P_tail": float(q.tail_ratio_P),
                    "tau_tail": float(q.tail_ratio_tau_z),
                    "Fy_tail": np.nan,
                    "quadrupole_warning": q.quadrupole_warning,
                }
            )
    return pd.DataFrame(rows)


def compute_fixed_center_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    mach_values = np.linspace(args.mach_min, args.mach_max, args.num_mach)
    for idx, mach in enumerate(mach_values, start=1):
        print(f"[fixed-center {idx}/{len(mach_values)}] Mach={mach:.6g}", flush=True)
        common = dict(
            e=args.e,
            n0=0.0,
            A=float(mach),
            n_max=args.n_max,
            n_mu=args.n_mu,
            n_phi=args.n_phi,
            backend=args.backend,
            chunk_size=64,
            rtol=args.rtol,
            tail_window=32,
            consecutive_windows=2,
            strict_convergence=True,
            xi_per_n=args.xi_per_n,
        )
        p = single_perturber_power(**common)
        tau = single_perturber_tau_z(**common)
        fy = single_perturber_force_y(**common)
        rows.append(
            {
                "kind": "fixed_center_single",
                "e": float(args.e),
                "n0": 0.0,
                "Mach": float(mach),
                "P_hat": float(p.value),
                "tau_hat": float(tau.value),
                "F_y_hat": float(fy.value),
                "minus_F_y_hat": float(-fy.value),
                "P_converged": bool(p.converged),
                "tau_converged": bool(tau.converged),
                "Fy_converged": bool(fy.converged),
                "P_n": int(p.n_values[-1]),
                "tau_n": int(tau.n_values[-1]),
                "Fy_n": int(fy.n_values[-1]),
                "P_tail": float(p.tail_ratio),
                "tau_tail": float(tau.tail_ratio),
                "Fy_tail": float(fy.tail_ratio),
            }
        )
    return pd.DataFrame(rows)


def compute_edg_rows(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    mach_values = np.linspace(args.mach_min, args.mach_max, args.edg_points)
    for idx, mach in enumerate(mach_values, start=1):
        print(f"[EDG {idx}/{len(mach_values)}] Mach={mach:.6g}", flush=True)
        coeff = edg_single_perturber_coefficients(
            A=float(mach),
            e=args.e,
            jmax=args.edg_jmax,
            lmax=args.edg_lmax,
            n_xi=args.edg_n_xi,
        )
        rows.append(
            {
                "kind": "edg_lmax20",
                "e": float(args.e),
                "n0": 0.0,
                "Mach": float(mach),
                "P_hat": float(2.0 * math.pi * coeff.P_shape),
                "tau_hat": float(2.0 * math.pi * mach * coeff.tau_z_shape),
                "F_y_hat": np.nan,
                "minus_F_y_hat": np.nan,
                "IE": float(coeff.IE),
                "IL": float(coeff.IL),
                "jmax": int(args.edg_jmax),
                "lmax": int(args.edg_lmax),
            }
        )
    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    full = df[df["kind"] == "full_binary_emri"].copy()
    quad = df[df["kind"] == "quadrupole"].copy()
    fixed = df[df["kind"] == "fixed_center_single"].copy()
    edg = df[df["kind"] == "edg_lmax20"].copy()

    colors = {0.0: "#1f77b4", 1.0: "#d62728"}
    labels = {0.0: r"$n_0=0$", 1.0: r"$n_0=1$"}
    quantities = [
        ("P_hat", r"$P/(2\bar\rho\nu^2M^2/c_s)$"),
        ("tau_hat", r"$\tau_z\tilde\Omega/(2\bar\rho\nu^2M^2/c_s)$"),
        ("minus_F_y_hat", r"$-F_y/(2\bar\rho\nu^2M^2/c_s^2)$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True)
    for ax, (col, ylabel) in zip(axes, quantities):
        for n0, group in full.groupby("n0"):
            group = group.sort_values("Mach")
            ax.plot(
                group["Mach"],
                group[col],
                color=colors[float(n0)],
                lw=2.0,
                label=labels[float(n0)],
            )
        fixed_sorted = fixed.sort_values("Mach")
        ax.plot(
            fixed_sorted["Mach"],
            fixed_sorted[col],
            color="0.45",
            lw=1.9,
            ls="-.",
            label="single-perturber",
        )
        if col in ("P_hat", "tau_hat"):
            for n0, group in quad.groupby("n0"):
                group = group.sort_values("Mach")
                ax.plot(
                    group["Mach"],
                    group[col],
                    color=colors[float(n0)],
                    lw=1.7,
                    ls="--",
                    label=labels[float(n0)] + " (quad)",
                )
            ax.plot(
                edg["Mach"],
                edg[col],
                "o",
                ms=4.5,
                color="black",
                label=r"$\mathrm{Eytan}\endash\mathrm{Desjacques}\endash\mathrm{Ginat}$",
            )
        ax.set_xlabel(r"$\mathcal{M}$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlim(args.mach_min * 0.9, args.mach_max * 1.04)
        ax.tick_params(axis="both", which="major", labelsize=11)

    axes[0].text(
        0.06,
        0.94,
        rf"$e={args.e:g}$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=14,
    )
    axes[0].legend(fontsize=7.3, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"paper_fig1_emri_fluxes.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    limit = uv_mach_limit(args.e)
    if args.mach_max >= limit:
        raise ValueError(
            f"mach_max={args.mach_max} is in the point-source UV-divergent region; "
            f"for e={args.e}, require Mach < {limit:.12g}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = compute_full_rows(args)
    quad = compute_quadrupole_rows(args)
    fixed = compute_fixed_center_rows(args)
    edg = compute_edg_rows(args)
    df = pd.concat([full, quad, fixed, edg], ignore_index=True, sort=False)

    csv_path = args.output_dir / "paper_fig1_emri_fluxes_data.csv"
    df.to_csv(csv_path, index=False)
    save_plot(df, args, args.output_dir)

    summary = {
        "e": args.e,
        "nu_proxy_for_scaled_limit": args.nu_proxy,
        "mach_range": [args.mach_min, args.mach_max],
        "uv_mach_limit": limit,
        "n0_values": [0, 1],
        "normalization": {
            "P_hat": "P/(2*rho_bar*nu^2*M^2/c_s)",
            "tau_hat": "tau_z*tilde_Omega/(2*rho_bar*nu^2*M^2/c_s)",
            "F_y_hat": "F_y/(2*rho_bar*nu^2*M^2/c_s^2)",
            "EDG": "P_hat=2*pi*P_shape, tau_hat=2*pi*A*tau_z_shape",
        },
        "convergence": {
            "full_all_converged": bool(
                full[["P_converged", "tau_converged", "Fy_converged"]].to_numpy().all()
            ),
            "max_full_n": int(full[["P_n", "tau_n", "Fy_n"]].to_numpy().max()),
            "max_full_tail": float(full[["P_tail", "tau_tail", "Fy_tail"]].to_numpy().max()),
        },
        "edg": {
            "jmax": args.edg_jmax,
            "lmax": args.edg_lmax,
            "points": args.edg_points,
            "note": "EDG points are single-perturber n0=0 values.",
        },
        "fixed_center_single": {
            "note": "Gray dash-dot curve is the fixed-center single-perturber high-order sum.",
        },
    }
    summary_path = args.output_dir / "paper_fig1_emri_fluxes_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"csv = {csv_path}")
    print(f"figure = {args.output_dir / 'paper_fig1_emri_fluxes.png'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
