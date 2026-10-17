
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from .orbit_derivatives import BinaryOrbit


TWO_PI = 2.0 * math.pi


def s2_periodic(argument: np.ndarray | float) -> np.ndarray:

    reduced = np.remainder(np.asarray(argument, dtype=np.float64), TWO_PI)
    return math.pi * math.pi / 6.0 - 0.5 * math.pi * reduced + 0.25 * reduced * reduced


def _image_sum(delta: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    lower = int(math.floor(float(np.min(delta - y)) / TWO_PI))
    upper = int(math.ceil(float(np.max(delta + y)) / TWO_PI))
    triangles = np.zeros_like(y)
    distances = np.zeros_like(y)
    for image in range(lower, upper + 1):
        distance = np.abs(delta - TWO_PI * image)
        active = distance < y
        triangles += np.where(active, y - distance, 0.0)
        distances += np.where(active, distance, 0.0)
    return triangles, distances


def classical_total_pair_kernel(
    delta: np.ndarray | float,
    radius: np.ndarray | float,
    *,
    rho_bar: float,
    omega_tilde: float,
    c_s: float,
) -> np.ndarray:

    r, difference = np.broadcast_arrays(np.asarray(radius, dtype=np.float64), np.asarray(delta, dtype=np.float64))
    y = omega_tilde * r / c_s
    triangles, _ = _image_sum(difference, y)
    prefactor = 4.0 * math.pi * math.pi * rho_bar / (omega_tilde * omega_tilde)
    kernel = np.divide(prefactor * triangles, r, out=np.zeros_like(r), where=r > 0.0)
    zero = (r == 0.0) & (np.remainder(difference, TWO_PI) == 0.0)
    if np.any(zero):
        kernel = np.where(zero, prefactor * omega_tilde / c_s, kernel)
    return kernel


def classical_total_pair_kernel_dr(
    delta: np.ndarray | float,
    radius: np.ndarray | float,
    *,
    rho_bar: float,
    omega_tilde: float,
    c_s: float,
) -> np.ndarray:

    r, difference = np.broadcast_arrays(np.asarray(radius, dtype=np.float64), np.asarray(delta, dtype=np.float64))
    y = omega_tilde * r / c_s
    _, distances = _image_sum(difference, y)
    prefactor = 4.0 * math.pi * math.pi * rho_bar / (omega_tilde * omega_tilde)
    return np.divide(prefactor * distances, r * r, out=np.zeros_like(r), where=r > 0.0)


def classical_pair_average(
    orbit: BinaryOrbit,
    *,
    rho_bar: float,
    c_s: float,
    n_ell: int,
    energy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(max(16, n_ell // 8))
    scan = np.linspace(-math.pi, math.pi, max(65, n_ell // 2 + 1))
    beta = math.sqrt(1.0 - orbit.e * orbit.e)
    fractions = (orbit.f1, orbit.f2)
    alpha = (orbit.f2, -orbit.f1)
    mach = orbit.a * orbit.physical_omega / c_s
    image_count = int(math.ceil((2.0 * mach * (1.0 + orbit.e) + math.pi + 2.0 * orbit.e) / TWO_PI))
    total = np.zeros(4)
    static = np.zeros(4)

    def state(xi):
        cosine, sine = np.cos(xi), np.sin(xi)
        jacobian = 1.0 - orbit.e * cosine
        x = orbit.a * (cosine - orbit.e)
        y = orbit.a * beta * sine
        xe = orbit.a * (-1.0 - sine * sine / jacobian)
        ye = orbit.a * (-orbit.e * sine / beta + beta * cosine * sine / jacobian)
        return x, y, xe, ye, jacobian

    for body_a in range(2):
        for body_b in range(2):
            if body_a != body_b and (abs(alpha[body_a]) < abs(alpha[body_b]) or (abs(alpha[body_a]) == abs(alpha[body_b]) and body_a > body_b)):
                continue
            closed_value = 0.0
            static_value = 0.0
            self_subsonic = body_a == body_b and abs(alpha[body_a]) * mach * math.sqrt((1.0 + orbit.e) / (1.0 - orbit.e)) < 1.0
            for xi in TWO_PI * (np.arange(n_ell) + 0.5) / n_ell:
                x, y, xe, ye, jacobian = state(xi)

                def separation(offset):
                    xp, yp, xep, yep, jp = state(xi + offset)
                    dx = alpha[body_a] * x - alpha[body_b] * xp
                    dy = alpha[body_a] * y - alpha[body_b] * yp
                    radius = np.hypot(dx, dy)
                    numerator = dx * (alpha[body_a] * xe - alpha[body_b] * xep) + dy * (alpha[body_a] * ye - alpha[body_b] * yep)
                    radius_e = np.divide(numerator, radius, out=np.zeros_like(radius), where=radius > 0.0)
                    delta = -offset + orbit.e * (np.sin(xi + offset) - math.sin(xi))
                    return radius, radius_e, delta, jp

                boundaries = [-math.pi, 0.0, math.pi]
                if not self_subsonic:
                    radius_scan, _, delta_scan, _ = separation(scan)
                    for image in range(-image_count, image_count + 1):
                        values = orbit.physical_omega * radius_scan / c_s - np.abs(delta_scan - TWO_PI * image)
                        intervals = np.flatnonzero(values[:-1] * values[1:] < 0.0)
                        for index in intervals:
                            def wavefront(offset):
                                radius, _, delta, _ = separation(offset)
                                return float(orbit.physical_omega * radius / c_s - abs(delta - TWO_PI * image))
                            boundaries.append(brentq(wavefront, scan[index], scan[index + 1], xtol=5.0e-15))
                boundaries = np.unique(boundaries)
                half = 0.5 * np.diff(boundaries)
                midpoint = 0.5 * (boundaries[1:] + boundaries[:-1])
                offsets = (midpoint[:, None] + half[:, None] * nodes).ravel()
                quadrature_weights = (half[:, None] * weights).ravel()
                radius, radius_e, delta, jp = separation(offsets)
                averaging_weights = jacobian * jp * quadrature_weights / (TWO_PI * n_ell)
                static_kernel = 2.0 * math.pi * rho_bar / (c_s * c_s)
                static_integrand = static_kernel * (radius if energy else radius_e)
                static_value += float(np.dot(averaging_weights, static_integrand))
                if not self_subsonic:
                    kernel = classical_total_pair_kernel if energy else classical_total_pair_kernel_dr
                    values = kernel(delta, radius, rho_bar=rho_bar, omega_tilde=orbit.physical_omega, c_s=c_s)
                    closed_value += float(np.dot(averaging_weights, values if energy else values * radius_e))
            weight = fractions[body_a] * fractions[body_b]
            if body_a != body_b:
                weight *= 2.0
            index = 1 if body_a == body_b == 0 else (2 if body_a == body_b == 1 else 3)
            total[index] += weight * closed_value
            static[index] += weight * static_value
    total[0] = np.sum(total[1:])
    static[0] = np.sum(static[1:])
    return total, static
