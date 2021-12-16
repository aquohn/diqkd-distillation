import Pkg
Pkg.activate(@__DIR__)

import Conda

envdir = String(@__DIR__) * "/conda_env"
Conda.add_channel("aquohn", envdir)
Conda.add("numpy", envdir)
Conda.add("chaospy", envdir)
Conda.add("sympy", envdir)
Conda.add("ncpol2sdpa", envdir; channel="aquohn")
Conda.list(envdir)
pirntln("NOTE: If the installed ncpol2sdpa is the conda-forge \
        version, activate the environment from the command line \
        and run
        ```
        conda remove ncpol2sdpa
        conda install -c aquohn --override-channels ncpol2sdpa
        ```")

