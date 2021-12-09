import numpy as np
from math import pi
import ncpol2sdpa as ncp
from glob import glob
from joblib import Parallel, delayed, parallel_backend
import qre
from qre import BFFProblem
import datetime

VERBOSE = 2
qre.VERBOSE = VERBOSE
NPA_LEVEL = 2
NUM_SUBWORKERS = 4  # Number of cores each worker has access to

SOLVEF = lambda sdp: sdp.solve()
try:
    import mosek

    SOLVEF = lambda sdp: sdp.solve(
        "mosek", solverparameters={"num_threads": int(NUM_SUBWORKERS)}
    )
except ModuleNotFoundError:
    try:
        import scs, cvxpy

        SOLVEF = lambda sdp: sdp.solve("scs")
    except ModuleNotFoundError:
        pass


def test_behav_bound(test_p=None, solvef=SOLVEF):
    starttime = datetime.datetime.now()
    print(f"Start: {starttime}")

    # Defining the test sys
    test_q = 0.01

    if test_p is None:
        test_p = np.array(
            [
                [
                    [
                        [0.49912513, 0.00554339, 0.49693897],
                        [0.00487971, 0.00547885, 0.49462103],
                    ],
                    [[0.00087834, 0.49446008, 0.0030645], [0.49514176, 0.49454262, 0.00540044]],
                ],
                [
                    [
                        [0.00510044, 0.49491103, 0.00304437],
                        [0.49934586, 0.49497557, 0.00536231],
                    ],
                    [[0.49489609, 0.0050855, 0.49695216], [0.00063268, 0.00500297, 0.49461623]],
                ],
            ]
        )

    prob = BFFProblem(solvef=solvef)
    ops = ncp.flatten([prob.A, prob.B, prob.Z])  # Base monomials involved in problem
    obj = prob.objective(1, test_q)  # Placeholder objective function

    sdp = ncp.SdpRelaxation(ops, verbose=VERBOSE - 1, normalized=True, parallel=0)
    momeqs = prob.moment_eqs + prob.behav_eqs(test_p)
    sdp.get_relaxation(
        level=NPA_LEVEL,
        equalities=prob.op_eqs[:],
        inequalities=prob.op_ineqs[:],
        momentequalities=momeqs,
        momentinequalities=prob.moment_ineqs[:],
        objective=obj,
        substitutions=prob.substitutions,
        extramonomials=prob.extra_monos,
    )
    setuptime = datetime.datetime.now()
    print(f"Setup Done At: {setuptime}, Delta: {setuptime - starttime}")
    ent = prob.compute_entropy(sdp, test_q)

    print(f"Entropy: {ent}")
    endtime = datetime.datetime.now()
    print(f"End: {endtime}, Delta: {endtime - setuptime}")
