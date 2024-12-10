# Device-independent quantum cryptography and nonlocal distillation

## Overview

This repository contains the LaTeX source and code for my final year (honours) project in Computer Engineering, at the National University of Singapore, titled "Device-independent quantum cryptography and nonlocal distillation". We focus on a particular type of *quantum cryptography*, namely *quantum key distribution* (QKD), which exploits the laws of quantum physics to perform the task of establishing a shared secret between two parties who are connected by an untrusted channel. Typically, this takes the form of a protocol with multiple rounds, where one party (Alice) randomly chooses a *state* and sends it to the other (Bob), who randomly chooses a *measurement* and measures the state with it, or where Alice and Bob receive two parts of an *entangled* state and randomly choose measurements to perform on it.

*Device-independent QKD* (DI-QKD) refers to a class of QKD protocols where the quantum operations (state generation and measurement) performed by the devices are *untrusted* - security is proven based solely on the probability distribution of the measurement outcomes (also referred to as a *behaviour*), and can be estimated even if the states and measurements deviate from the intended ones. The advantage of this is that we do not need to verify the implementation of the states and measurements, which can be difficult to do in practice.

The key observation is that some probability distributions are *nonlocal* - they cannot have been generated using some pre-agreed strategy, and must genuinely have come from quantum measurement results, whose outcome cannot be predicted. This provides the security in DI-QKD.

The work done in this project aims to generalise the observation that, for a certain DI-QKD protocol, the results of two rounds can be combined to increase the overall key rate, even increasing it from zero to nonzero. This process of combination is referred to as *nonlocal distillation*, as it makes a weakly nonlocal behaviour more nonlocal, and the specific processes that are used are referred to *wirings*. With this background, you, dear reader, are now equipped to understand the abstract of this work:

> Quantum key distribution (QKD) allows secret keys with the highest possible level of security to be agreed on over an insecure channel, something impossible in classical cryptography. Device-independent quantum key distribution (DIQKD) relies only on classical statistical data to guarantee security, instead of requiring difficult-to-verify hardware guarantees as in device-dependent QKD. We initiate a foundational study of distillation in DIQKD using wirings: classical processing that combines multiple instances of DIQKD devices into one. We generalise existing definitions in the literature for DIQKD, identify various neglected issues, study the structure of the space of wirings, and establish key results on the potential for distillation to enhance quantum cryptography within this broader framework: namely, a necessary condition for wirings to increase the secret key capacity of a DIQKD device, and the fact that extremal wirings maximise the secret key capacity. We also develop a computational representation for wirings, and examine computational methods for finding, given a specific DIQKD device, upper bounds on its secret key capacity and the wiring which maximises it.

**NOTE:** Some issues were found in the report after its submission, and revisions were required, hence the TODOs. It has been a while, but I hope to iron out all the bugs EventuallyTM. Do take the areas marked with TODOs with a pinch of salt, as further work there may uncover more problems.

## Organisation

The `wiring` and `presentation` folders contain the LaTeX source code for the report and presentation, respectively. The `code` folder contains:
- `lrs`, containing files representing some of the polytopes studied Section 5.4,
- `unitary.py` and `wiring.py`, which set up Sympy sessions for exploring parametrisations of states/measurements and wirings respectively,
- `wiring.ipynb`, which explores some of the key rate calculations under wirings, and
- `DIQKDWirings.jl`, which contains everything else.

While having the structure of a package, `DIQKDWirings.jl` is essentially just a collection of scripts used to calculate and plot various functions. A high-level description of the main scripts is as follows:
|Script          |Description                                                  |
|---             |---
|`nonlocality.jl`|Defines nonlocal Settings, and Behaviours and Correlators therein|
|`keyrates.jl`   |Calculates the key rates for a behaviour when used in more complex protocols|
|`optims.py`     |Lower bounds the key rates of various behaviours with semidefinite programming, using the [Brown, Fawzi and Fawzi](https://github.com/peterjbrown519/DI-rates) approach|
|`makie.jl`      |Plots maximal correlation and key rates using the Makie backend|
|`plots.jl`      |Plots maximal correlation and key rates in a much wider variety of scenarios, mostly using the Plotly backend|
|`wiring.jl`     |Computations for the wiring function and wiring matrix representations (Section 5.2-5.3)|
|`polytope.jl`   |Computations for polytopes of behaviours and wirings (Section 5.4)|
|`maxcorr.jl`    |Calculates and explores maximal correlation (Section 7.1)|
|`quantum.jl`    |Parametrisation of quantum states and measurements (to enable optimisation)|
|`upper.jl`      |Defines classes and functions to be used for upper bounding the key rate of a behaviour by finding a concrete attack against it (Section 8). `upoly.jl`, `udirect.jl` and `umatlab.jl` implement different approaches to finding these attacks.|
