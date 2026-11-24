# Calculators

- `classical_fluid.py`: normalized classical-fluid energy, angular-momentum, and linear-momentum fluxes, $\widehat P$, $\widehat\tau_z$, and $\widehat F_y$.
- `quantum_fluid.py`: normalized quantum-fluid energy, angular-momentum, and linear-momentum fluxes, $\widehat P$, $\widehat\tau_z$, and $\widehat F_y$.
- `quadrupole_fluxes.py`: classical-fluid and $c_S=0$ quantum-fluid fluxes at quadrupole order (quadrupole + monopole radiation).
- `classical_fluid_quadrupole.py`: closed-form $n_0=0$ classical-fluid fluxes at quadrupole order.
- `single_perturber_classical_fluid.py`: classical-fluid flux calculator for a single perturber in an eccentric Keplerian orbit. The outputs are $P/(2\bar\rho m_p^2/c_s)$, $\tau_z\tilde\Omega/(2\bar\rho m_p^2/c_s)$, and $F_y/(2\bar\rho m_p^2/c_s^2)$, with $m_p$ the perturber mass.
- `edg_single_perturber_coefficients.py`: $n_0=0$ single-perturber coefficients of Eytan, Desjacques, and Ginat ([arXiv:2509.15632](https://arxiv.org/abs/2509.15632)). The dimensionless outputs `P_shape` and `tau_z_shape` are related to the fluxes by $P/(2\bar\rho m_p^2/c_s)=2\pi P_{\rm shape}$ and $\tau_z\tilde\Omega/(2\bar\rho m_p^2/c_s)=2\pi A\tau_{z,{\rm shape}}$.
