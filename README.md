# pism

[![Python package](https://github.com/mikegrudic/pism/actions/workflows/test.yml/badge.svg)](https://github.com/mikegrudic/pism/actions/workflows/test.yml)
[![Readthedocs Status][docs-badge]][docs-link]

[docs-link]:           https://pism-code.readthedocs.io
[docs-badge]:          https://readthedocs.org/projects/pism-code/badge

`pism` is a flexible, object-oriented implementation for systems of ISM microphysics and chemistry equations, with numerical solvers implemented in JAX, and interfaces for embedding the equations and their Jacobians into other codes.

## Do we really need yet another ISM code?

`pism` might be interesting because it combines two powerful concepts:
1. **Object-oriented implementation of microphysics and chemistry via the `Process` class**, which implements methods for representing physical processes, composing them into a network in a fully-symbolic `sympy` representation. OOP is nice here because if you want to add a new process to `pism`, you typically only have to do it in one file. Rate expressions never have to be repeated in-code. Most processes one would want to implement follow very common patterns (e.g. 2-body processes), so class inheritance is also used to minimize new lines of code. 
Once you've constructed your system, `pism` can give you the symbolic equations to manipulate and analyze as you please. If you want to solve the equations numerically, `Process` has methods for substituting known values into numerical solvers. It can also automatically generate compilable implementations of the RHS of the system to embed in your choice of simulation code and plug into your choice of solver.
2. **Fast, differentiable implementation of nonlinear algebraic and differential-algebraic equation solvers with JAX**, implemented in its functional programming paradigm (e.g. `pism.numerics.newton_rootsolve`). These can achieve excellent numerical throughput running natively on GPUs - in fact, crunching iterates in-place is essentially the best-case application of numerics on GPUs. Differentiability enables sensitivity analysis with respect to all parameters in a single pass, instead of constructing a grid of `N` parameter variations for `N` parameters. This makes it easier in principle to directly answer questions like "How sensitive is this temperature to the abundance of C or the ionization energy of H?", etc.

## Roadmap

`pism` is in an early prototyping phase right now. Here are some things I would eventually like to add:
* Flexible implementation of a reduced network suitable for RHD simulations in GIZMO and potentially other codes.
* Dust and radiation physics: add the dust energy equation and evolution of photon number densities to the network.
* Thermochemical equilibrium and time-dependent solvers: only chemical equilibrium at fixed `T` is implemented right now.
* Interfaces to convert from other existing chemistry network formats to the `Process` representation.
* Solver robustness upgrades: thermochemical networks can be quite challenging numerically, due to how steeply terms switch on with increasing `T`. In can be hard to get a solution without good initial guesses.
* If possible, glue interface allowing an existing compiled code to call the JAX solvers on-the-fly.

pls halp.

## Installation

Clone the repo and run `pip install .` from the directory.

# Quickstart: Collisional Ionization Equilibrium

Example of using `pism` to solve for collisional ionization equilibrium (CIE) for a hydrogen-helium mixture and plot the ionization states as a function of temperature.


```python
%matplotlib inline
%config InlineBackend.figure_format='retina'
import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
```

## Simple processes
A simple process is defined by a single reaction, with a specified rate.

Let's inspect the structure of a single process, the gas-phase recombination of H+: `H+ + e- -> H + hν` 


```python
from pism.processes import CollisionalIonization, GasPhaseRecombination

process = GasPhaseRecombination("H+")
print(f"Name: {process.name}")
print(f"Heating rate coefficient: {process.heat_rate_coefficient}")
print(f"Heating rate per cm^-3: {process.heat}"),
print(f"Rate coefficient: {process.rate_coefficient}")
print(f"Recombination rate per cm^-3: {process.rate}")
print(f"RHS of e- number density equation: {process.network["e-"]}")
```

    Name: Gas-phase recombination of H+
    Heating rate coefficient: -1.46719838641439e-26*sqrt(T)/((0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)
    Heating rate per cm^-3: -1.46719838641439e-26*sqrt(T)*n_H+*n_e-/((0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)
    Rate coefficient: 1.41621465870114e-10/(sqrt(T)*(0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)
    Recombination rate per cm^-3: 1.41621465870114e-10*n_H+*n_e-/(sqrt(T)*(0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)
    RHS of e- number density equation: -1.41621465870114e-10*n_H+*n_e-/(sqrt(T)*(0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)


Note that all symbolic representations assume CGS units as is standard in ISM physics.

## Composing processes
Now let's define our full network as a sum of simple processes


```python
processes = [CollisionalIonization(s) for s in ("H", "He", "He+")] + [GasPhaseRecombination(i) for i in ("H+", "He+", "He++")]
system = sum(processes)

system.subprocesses
```




    [Collisional Ionization of H,
     Collisional Ionization of He,
     Collisional Ionization of He+,
     Gas-phase recombination of H+,
     Gas-phase recombination of He+,
     Gas-phase recombination of He++]



Summed processes keep track of all subprocesses, e.g. the total net heating rate is:


```python
system.heat
```




$\displaystyle - \frac{1.55 \cdot 10^{-26} n_{He+} n_{e-}}{T^{0.3647}} - \frac{1.2746917300104 \cdot 10^{-21} \sqrt{T} n_{H} n_{e-} e^{- \frac{157809.1}{T}}}{\frac{\sqrt{10} \sqrt{T}}{1000} + 1} - \frac{1.46719838641439 \cdot 10^{-26} \sqrt{T} n_{H+} n_{e-}}{\left(0.00119216696847702 \sqrt{T} + 1.0\right)^{1.748} \left(0.563615123664978 \sqrt{T} + 1.0\right)^{0.252}} - \frac{9.37661057635428 \cdot 10^{-22} \sqrt{T} n_{He} n_{e-} e^{- \frac{285335.4}{T}}}{\frac{\sqrt{10} \sqrt{T}}{1000} + 1} - \frac{4.9524176975855 \cdot 10^{-22} \sqrt{T} n_{He+} n_{e-} e^{- \frac{631515}{T}}}{\frac{\sqrt{10} \sqrt{T}}{1000} + 1} - \frac{5.86879354565754 \cdot 10^{-26} \sqrt{T} n_{He++} n_{e-}}{\left(0.00119216696847702 \sqrt{T} + 1.0\right)^{1.748} \left(0.563615123664978 \sqrt{T} + 1.0\right)^{0.252}}$



Summing processes also sums all chemical and gas/dust cooling/heating rates. 


```python
system.print_network_equations()
```

    dn_H+/dt = 5.85e-11*sqrt(T)*n_H*n_e-*exp(-157809.1/T)/(sqrt(10)*sqrt(T)/1000 + 1) - 1.41621465870114e-10*n_H+*n_e-/(sqrt(T)*(0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)
    dn_He+/dt = 2.38e-11*sqrt(T)*n_He*n_e-*exp(-285335.4/T)/(sqrt(10)*sqrt(T)/1000 + 1) - 5.68e-12*sqrt(T)*n_He+*n_e-*exp(-631515/T)/(sqrt(10)*sqrt(T)/1000 + 1) - n_He+*n_e-*(0.0019*(1 + 0.3*exp(-94000.0/T))*exp(-470000.0/T)/T**1.5 + 1.93241606228058e-10/(sqrt(T)*(0.000164934781188511*sqrt(T) + 1.0)**1.7892*(4.84160744811772*sqrt(T) + 1.0)**0.2108)) + 5.66485863480458e-10*n_He++*n_e-/(sqrt(T)*(0.00059608348423851*sqrt(T) + 1.0)**1.748*(0.281807561832489*sqrt(T) + 1.0)**0.252)
    dn_He++/dt = 5.68e-12*sqrt(T)*n_He+*n_e-*exp(-631515/T)/(sqrt(10)*sqrt(T)/1000 + 1) - 5.66485863480458e-10*n_He++*n_e-/(sqrt(T)*(0.00059608348423851*sqrt(T) + 1.0)**1.748*(0.281807561832489*sqrt(T) + 1.0)**0.252)
    dn_H/dt = -5.85e-11*sqrt(T)*n_H*n_e-*exp(-157809.1/T)/(sqrt(10)*sqrt(T)/1000 + 1) + 1.41621465870114e-10*n_H+*n_e-/(sqrt(T)*(0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252)
    dn_e-/dt = 5.85e-11*sqrt(T)*n_H*n_e-*exp(-157809.1/T)/(sqrt(10)*sqrt(T)/1000 + 1) + 2.38e-11*sqrt(T)*n_He*n_e-*exp(-285335.4/T)/(sqrt(10)*sqrt(T)/1000 + 1) + 5.68e-12*sqrt(T)*n_He+*n_e-*exp(-631515/T)/(sqrt(10)*sqrt(T)/1000 + 1) - n_He+*n_e-*(0.0019*(1 + 0.3*exp(-94000.0/T))*exp(-470000.0/T)/T**1.5 + 1.93241606228058e-10/(sqrt(T)*(0.000164934781188511*sqrt(T) + 1.0)**1.7892*(4.84160744811772*sqrt(T) + 1.0)**0.2108)) - 1.41621465870114e-10*n_H+*n_e-/(sqrt(T)*(0.00119216696847702*sqrt(T) + 1.0)**1.748*(0.563615123664978*sqrt(T) + 1.0)**0.252) - 5.66485863480458e-10*n_He++*n_e-/(sqrt(T)*(0.00059608348423851*sqrt(T) + 1.0)**1.748*(0.281807561832489*sqrt(T) + 1.0)**0.252)
    dn_He/dt = -2.38e-11*sqrt(T)*n_He*n_e-*exp(-285335.4/T)/(sqrt(10)*sqrt(T)/1000 + 1) + n_He+*n_e-*(0.0019*(1 + 0.3*exp(-94000.0/T))*exp(-470000.0/T)/T**1.5 + 1.93241606228058e-10/(sqrt(T)*(0.000164934781188511*sqrt(T) + 1.0)**1.7892*(4.84160744811772*sqrt(T) + 1.0)**0.2108))


## Solving ionization equilibrium

We would like to solve for ionization equilibrium given a temperature $T$, overall H number density $n_{\rm H,tot}$, and helium mass fraction $Y$.  We define a dictionary of those input quantities and also one for the initial guesses of the number densities of the species in the reduced network.


```python
Tgrid = np.logspace(3,6,10**6)
ngrid = np.ones_like(Tgrid) 
Ygrid = 0.24*np.ones_like(Tgrid)

knowns = {"T": Tgrid, "n_Htot": ngrid, "Y": Ygrid}

```

Note that by default, the solver only directly solves for $n_{\rm H}$, $n_{\rm He}$ and $n_{\rm He+}$ because $n_{\rm H+}$, $n_{\rm He++}$, and $n_{\rm e-}$ are eliminated by conservation equations. So we only need initial guesses for those 3 quantities. By default the solver takes abundances $x_i = n_i / n_{\rm H,tot}$ as inputs and outputs.


```python
guesses = {
    "H": 0.5*np.ones_like(Tgrid),
    "He": 1e-5*np.ones_like(Tgrid),
    "He+": 1e-5*np.ones_like(Tgrid)
}
sol = system.chemical_equilibrium(knowns, guesses)
print(sol)
```

    {'He+': Array([ 5.9581362e-16, -2.7435927e-16,  6.8873975e-16, ...,
            6.5619079e-06,  6.5618306e-06,  6.5617523e-06], dtype=float32), 'H': Array([9.9999994e-01, 9.9999994e-01, 9.9999994e-01, ..., 6.0612069e-07,
           6.0611501e-07, 6.0610910e-07], dtype=float32), 'He': Array([7.8947358e-02, 7.8947358e-02, 7.8947358e-02, ..., 2.3453652e-09,
           2.3453146e-09, 2.3452633e-09], dtype=float32), 'H+': Array([5.9604645e-08, 5.9604645e-08, 5.9604645e-08, ..., 9.9999940e-01,
           9.9999940e-01, 9.9999940e-01], dtype=float32), 'e-': Array([7.4505806e-08, 7.4505806e-08, 7.4505806e-08, ..., 1.1578876e+00,
           1.1578876e+00, 1.1578876e+00], dtype=float32), 'He++': Array([7.4505802e-09, 7.4505806e-09, 7.4505797e-09, ..., 7.8940801e-02,
           7.8940801e-02, 7.8940801e-02], dtype=float32)}



```python
for i, xi in sorted(sol.items()):
    plt.loglog(Tgrid, xi, label=i)
plt.legend(labelspacing=0)
plt.ylabel("$x_i$")
plt.xlabel("T (K)")
plt.ylim(1e-4,3)
```




    (0.0001, 3)




    
![png](CIE_files/CIE_16_1.png)
    

