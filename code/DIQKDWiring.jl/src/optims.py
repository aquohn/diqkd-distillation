import numpy as np
import ncpol2sdpa as ncp
from itertools import product as itprod
import bidict
import qre
from qre import BFFProblem
import datetime
from math import prod

VERBOSE = 2
qre.VERBOSE = VERBOSE
NPA_LEVEL = 2
NUM_SUBWORKERS = 4  # Number of cores each worker has access to
TOL = 1e-5
TEST_P = np.array(
    [
        [  # a = 0
            [  # b = 0
                [0.49912513, 0.00554339, 0.49693897],  # x = 0
                [0.00487971, 0.00547885, 0.49462103],  # x = 1
            ],
            [  # b = 1
                [0.00087834, 0.49446008, 0.0030645],  # x = 0
                [0.49514176, 0.49454262, 0.00540044],  # x = 1
            ],
        ],
        [  # a = 1
            [  # b = 0
                [0.00510044, 0.49491103, 0.00304437],  # x = 0
                [0.49934586, 0.49497557, 0.00536231],  # x = 1
            ],
            [  # b = 1
                [0.49489609, 0.0050855, 0.49695216],  # x = 0
                [0.00063268, 0.00500297, 0.49461623],  # x = 1
            ],
        ],
    ]
)


def SOLVEF(sdp):
    sdp.solve()


try:
    import mosek

    def SOLVEF(sdp):
        sdp.solve("mosek", solverparameters={"num_threads": int(NUM_SUBWORKERS)})

except ModuleNotFoundError:
    try:
        import scs, cvxpy

        def SOLVEF(sdp):
            sdp.solve("scs")

    except ModuleNotFoundError:
        pass


def bound_entr_using_behav(p=None, q=0, solvef=SOLVEF):
    starttime = datetime.datetime.now()
    print(f"Start: {starttime}")

    if p is None:
        print("No behaviour provided; using a fixed test behaviour.")
        p = TEST_P

    prob = BFFProblem(solvef=solvef)
    ops = ncp.flatten([prob.A, prob.B, prob.Z])  # Base monomials involved in problem
    obj = prob.objective(1, q)  # Placeholder objective function

    sdp = ncp.SdpRelaxation(ops, verbose=VERBOSE - 1, normalized=True, parallel=0)
    momeqs = prob.moment_eqs + prob.behav_eqs(p)
    sdp.get_relaxation(
        level=NPA_LEVEL,
        equalities=prob.op_eqs[:],
        inequalities=prob.op_ineqs[:],
        momentequalities=momeqs[:],
        momentinequalities=prob.moment_ineqs[:],
        objective=obj,
        substitutions=prob.substitutions,
        extramonomials=prob.extra_monos,
    )
    setuptime = datetime.datetime.now()
    print(f"Setup Done At: {setuptime}, Delta: {setuptime - starttime}")
    ent = prob.compute_entropy(sdp, q)

    print(f"Entropy: {ent}")
    endtime = datetime.datetime.now()
    print(f"End: {endtime}, Delta: {endtime - setuptime}")


class Wiring(object):
    def __init__(self, pabxy, c):
        # Check probability
        if len(pabxy.shape) != 4:
            raise ValueError("pabxy has the wrong dimensions!")
        self.pabxy = pabxy[:]
        normsum = np.sum(pabxy, (0, 1))
        if not np.all([abs(norm - 1) < TOL for norm in normsum]):
            raise ValueError("pabxy is not normalised!")

        # Compute shape
        self.c = int(c)
        oA, oB, iA, iB = self.pabxy.shape
        # chi(a, b, x^c, y^c|a^c, b^c, x, y)
        self.shape = (oA, oB) + (iA,) * c + (iB,) * c + (oA,) * c + (oB,) * c + (iA, iB)

        # Generate chi variables
        self.n_params = prod(self.shape)
        self.params = ncp.generate_variables(
            "chi", self.n_params, hermitian=True, commutative=True
        )
        idx_ranges = [range(i) for i in self.shape]
        param_by_idx_dict = {
            idx: self.params[paramsidx]
            for paramsidx, idx in enumerate(itprod(*idx_ranges))
        }
        self.param_by_idx = bidict.bidict(param_by_idx_dict)
        rawidx_ranges = idx_ranges[2:-2]
        # use param_by_idx.inverse to get idx tuple from param operator

        # Generate effective pabxy as sympy expression
        self.ppabxy = np.empty_like(pabxy, object)
        for (a, b, x, y) in itprod(*idx_ranges):
            self.ppabxy[a, b, x, y] = 0
            for vals in itprod(*rawidx_ranges):
                vallists = [vals[(i * c): ((i + 1) * c)] for i in range(4)]
                xvals, yvals, avals, bvals = vallists
                vallists_for_p = [avals, bvals, xvals, yvals]
                p = prod([self.pabxy[(*args,)] for args in zip(*vallists_for_p)])
                param = self.param_by_idx[(a, b) + sum(vallists, ()) + (x, y)]
                self.ppabxy[a, b, x, y] += p * param

        # Generate normalisation constraints for chi
        self.eqconstr = [
            sum(
                [
                    self.param_by_idx[idx + cond_idx]
                    for idx in itprod(*idx_ranges[: 2 + 2 * c])
                ]
            )
            - 1
            for cond_idx in itprod(*idx_ranges[2 + 2 * c:])
        ]

        # TODO add LOSR constraints


def bound_entr_with_wirings(c=2, p=None, q=0, solvef=SOLVEF):
    starttime = datetime.datetime.now()
    print(f"Start: {starttime}")

    if p is None:
        print("No behaviour provided; using a fixed test behaviour.")
        p = TEST_P

    prob = BFFProblem(solvef=solvef)
    ops = ncp.flatten([prob.A, prob.B, prob.Z])  # Base monomials involved in problem
    obj = prob.objective(1, q)  # Placeholder objective function
    chi = WiringChar(p, c)

    # TODO see if chi.params need to be parameters or variables
    sdp = ncp.SdpRelaxation(
        ops, parameters=chi.params, verbose=VERBOSE - 1, normalized=True, parallel=0
    )
    momeqs = prob.moment_eqs + prob.behav_eqs(chi.ppabxy) + chi.eqconstr
    sdp.get_relaxation(
        level=NPA_LEVEL,
        equalities=prob.op_eqs[:],
        inequalities=prob.op_ineqs[:],
        momentequalities=momeqs[:],
        momentinequalities=prob.moment_ineqs[:],
        objective=obj,
        substitutions=prob.substitutions[:],
        extramonomials=prob.extra_monos[:],
    )
    setuptime = datetime.datetime.now()
    print(f"Setup Done At: {setuptime}, Delta: {setuptime - starttime}")
    ent = prob.compute_entropy(sdp, q)

    print(f"Entropy: {ent}")
    endtime = datetime.datetime.now()
    print(f"End: {endtime}, Delta: {endtime - setuptime}")