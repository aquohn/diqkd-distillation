import Pkg
Pkg.activate(@__DIR__)

import CondaPkg

CondaPkg.add_channel("aquohn")
CondaPkg.add_channel("mosek")
CondaPkg.add("numpy")
CondaPkg.add("chaospy")
CondaPkg.add("sympy")
CondaPkg.add("ncpol2sdpa")
CondaPkg.add("scs")
CondaPkg.add("cvxpy")
CondaPkg.status()
pirntln("NOTE: If the installed ncpol2sdpa is the conda-forge \
        version, activate the environment from the command line \
        and run
        ```
        conda remove ncpol2sdpa
        conda install -c aquohn --override-channels ncpol2sdpa
        ```")

