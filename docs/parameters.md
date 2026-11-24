# Parameters and Normalizations

The following symbols are used throughout:

| Quantity | Meaning |
| --- | --- |
| $\bar\rho$ | homogeneous background mass density |
| $\tilde\Omega$ | physical orbital angular frequency |
| $\Omega$ | rescaled orbital frequency |
| $n_0$ | $n_0=\sqrt{4\pi\bar\rho}/\tilde\Omega$ |
| $c_s$ | classical-fluid sound speed |
| $m_\phi$ | scalar particle mass in the quantum-fluid case |

The orbital period, total binary mass, and symmetric mass ratio are

$$
T=\frac{2\pi}{\tilde\Omega},\qquad
M=m_1+m_2,\qquad
\nu=\frac{m_1m_2}{(m_1+m_2)^2},\qquad 0<\nu\leq\frac14.
$$

The relative separation $\mathbf X=\mathbf X_1-\mathbf X_2$ is parameterized by the eccentricity $e$ and eccentric anomaly $\xi$:

$$
\frac{\mathbf X}{a}=(\cos\xi-e,\sqrt{1-e^2}\sin\xi,0),\qquad
\tilde\Omega t=\xi-e\sin\xi.
$$

In the center-of-mass frame,

$$
\mathbf X_1=\frac{m_2}{M}\mathbf X,\qquad
\mathbf X_2=-\frac{m_1}{M}\mathbf X.
$$

## Classical Fluid

For the classical inviscid Newtonian barotropic fluid, the code uses the orbital Mach number

$$
\begin{aligned}
\mathcal M &= a\Omega=\frac{a\tilde\Omega}{c_s},\qquad
A=a\Omega=\frac{a\tilde\Omega}{c_s},\qquad
\Omega=\frac{\tilde\Omega}{c_s},\\
m^2 &= \frac{4\pi\bar\rho}{c_s^2},\qquad
n_0=\frac{m}{\Omega}=\frac{\sqrt{4\pi\bar\rho}}{\tilde\Omega},\qquad
\bar\rho=\frac{(n_0\tilde\Omega)^2}{4\pi}=\frac{c_s^2m^2}{4\pi}.
\end{aligned}
$$

For $n>0$, the radiating wavenumber is

$$
k_n=\Omega\sqrt{n^2+n_0^2}.
$$

The outputs are

$$
\widehat P=\frac{P}{2\bar\rho M^2/c_s},\qquad
\widehat\tau_z=\frac{\tau_z\tilde\Omega}{2\bar\rho M^2/c_s},\qquad
\widehat F_y=\frac{F_y}{2\bar\rho M^2/c_s^2}.
$$
Here $P$, $\tau_z$, and $F_y$ denote the energy flux, the $z$-component of the angular-momentum flux, and the $y$-component of the linear-momentum flux, respectively.

## Quantum Fluid

For the Schrödinger-Poisson quantum fluid, the code uses

$$
\begin{aligned}
\mathcal M_Q &= a\sqrt{\Omega}=a\sqrt{2m_\phi\tilde\Omega},\qquad
A=a\sqrt{\Omega},\qquad
\Omega=2m_\phi\tilde\Omega,\\
m^2 &= 16\pi m_\phi^2\bar\rho,\qquad
n_0=\frac{m}{\Omega}=\frac{\sqrt{4\pi\bar\rho}}{\tilde\Omega},\qquad
\bar\rho=\frac{(n_0\tilde\Omega)^2}{4\pi}=\frac{m^2}{16\pi m_\phi^2}.
\end{aligned}
$$

For $n>0$, the radiating wavenumber is

$$
k_n=\sqrt{\Omega}\left[\frac{\sqrt{(c_S^2/\Omega)^2+4(n^2+n_0^2)}-c_S^2/\Omega}{2}\right]^{1/2}.
$$

A finite quantum-fluid sound-speed term is specified by `cS2_over_Omega`, with $c_S^2/\Omega\in\mathbb R$. The physical quantum-fluid sound speed is $c_S/(2m_\phi)$. By default, $c_S=0$.

The outputs are

$$
\widehat P=\frac{P}{2\bar\rho M^2m_\phi/\sqrt{\Omega}},\qquad
\widehat\tau_z=\frac{\tau_z\tilde\Omega}{2\bar\rho M^2m_\phi/\sqrt{\Omega}},\qquad
\widehat F_y=\frac{F_y\sqrt{\Omega}/m_\phi}{2\bar\rho M^2m_\phi/\sqrt{\Omega}}.
$$

## Uniform-Sphere Model

The optional uniform-sphere model uses `R1_over_a=R_1/a` and `R2_over_a=R_2/a`; both default to zero in the point-particle model. For body $I$, the source factor is multiplied by

$$
W(k_nR_I)=3\frac{\sin(k_nR_I)-k_nR_I\cos(k_nR_I)}{(k_nR_I)^3},
$$
where $R_I$ is the radius of the sphere with uniform mass density.
