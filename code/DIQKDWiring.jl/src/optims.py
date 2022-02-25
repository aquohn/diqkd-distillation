import numpy as np
import ncpol2sdpa as ncp
from itertools import product as itprod
import bidict
import qre
from qre import BFFProblem
import datetime
from math import prod
from sympy import symbols

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

REVIVAL_P = np.array(
    [
        [
            [
                [0.95939617, 0.95628008, 0.96954558],
                [0.9321242, 0.89989456, 0.92203276]],
            [
                [0.00442781, 0.00518686, 0.00194679],
                [0.03169978, 0.06157238, 0.04945962],
            ],
        ],
        [
            [
                [0.01209621, 0.01521229, 0.00194679],
                [0.00335293, 0.03558257, 0.01344437],
            ],
            [
                [0.02407981, 0.02332076, 0.02656083],
                [0.03282309, 0.00295049, 0.01506325],
            ],
        ],
    ]
)


SINGLET_P = np.array(
    [
        [
            [
                [0.5, 0.426776695296637, 0.426776695296637],
                [0.25, 0.426776695296637, 0.0732233047033631],
            ],
            [
                [0, 0.0732233047033631, 0.0732233047033631],
                [0.25, 0.0732233047033631, 0.426776695296637],
            ],
        ],
        [
            [
                [0, 0.0732233047033631, 0.0732233047033631],
                [0.25, 0.0732233047033631, 0.426776695296637],
            ],
            [
                [0.5, 0.426776695296637, 0.426776695296637],
                [0.25, 0.426776695296637, 0.0732233047033631],
            ],
        ],
    ]
)

WERNER_P = np.array(
    [
        [
            [[0.37839291, 0.37839291, 0.431575], [0.37839291, 0.12160709, 0.25]],
            [[0.12160709, 0.12160709, 0.068425], [0.12160709, 0.37839291, 0.25]],
        ],
        [
            [[0.12160709, 0.12160709, 0.068425], [0.12160709, 0.37839291, 0.25]],
            [[0.37839291, 0.37839291, 0.431575], [0.37839291, 0.12160709, 0.25]],
        ],
    ]
)


def SOLVEF(sdp):
    sdp.solve()


try:
    import mosek

    def MOSEK_SOLVEF(sdp):
        sdp.solve("mosek", solverparameters={"num_threads": int(NUM_SUBWORKERS), "intpnt_co_tol_dfeas": 1.0e-6, "intpnt_co_tol_pfeas": 1.0e-6})

except ModuleNotFoundError:
    MOSEK_SOLVEF = None

try:
    import scs, cvxpy

    def SCS_SOLVEF(sdp):
        sdp.solve("scs")

except ModuleNotFoundError:
    SCS_SOLVEF = None


def compute_entropy(prob, sdp, q=0):
    starttime = datetime.datetime.now()
    print(f"Start: {starttime}")
    ent = prob.compute_entropy(sdp, q)
    print(f"Entropy: {ent}")
    endtime = datetime.datetime.now()
    print(f"End: {endtime}, Delta: {endtime - starttime}")
    return ent


def behav_problem(p=None, **kwargs):
    verbose = kwargs.get("verbose", VERBOSE)
    starttime = datetime.datetime.now()
    if verbose > 1:
        print(f"Start: {starttime}")

    if p is None:
        if verbose > 0:
            print("No behaviour provided; using a fixed test behaviour.")
        p = TEST_P
    prob = BFFProblem(**kwargs)
    obj = kwargs.get("objective", None)
    extra_monos = []
    if obj is None:
        if verbose > 0:
            print("No objective provided, using H(A|E,x*=0)")
        t, q = symbols(r"t q", nonnegative=True)
        extra_monos += prob.extract_monomials_from_obj(prob.HAgEx_objective(t, q))
        obj = 0
    else:
        extra_monos += prob.extract_monomials_from_obj(obj)
    extra_monos_f = kwargs.get("extra_monos_f", lambda prob: prob.generate_ABE_monomials())
    extra_monos += extra_monos_f(prob)
    npa = kwargs.get("npa", NPA_LEVEL)
    parallel = kwargs.get("parallel", 0)
    moment_eqs = prob.analyse_behav(p)

    sdp = ncp.SdpRelaxation(
        list(prob.op_set), verbose=verbose, normalized=True, parallel=parallel
    )
    sdp.get_relaxation(
        level=npa,
        equalities=[],
        inequalities=[],
        momentequalities=moment_eqs,
        momentinequalities=[],
        objective=obj,
        substitutions=prob.substitutions,
        extramonomials=extra_monos,
    )
    sdp.process_constraints(momentequalities=moment_eqs)
    setuptime = datetime.datetime.now()
    if verbose > 1:
        print(f"Setup Done At: {setuptime}, Delta: {setuptime - starttime}")
    return prob, sdp


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
                vallists = [vals[(i * c) : ((i + 1) * c)] for i in range(4)]
                xvals, yvals, avals, bvals = vallists
                vallists_for_p = [avals, bvals, xvals, yvals]
                p = prod(self.pabxy[(*args,)] for args in zip(*vallists_for_p))
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
            for cond_idx in itprod(*idx_ranges[2 + 2 * c :])
        ]

        # TODO add LOSR variables


def behav_wirs_problem(c=2, p=None, solvef=SOLVEF):
    starttime = datetime.datetime.now()
    print(f"Start: {starttime}")

    if p is None:
        print("No behaviour provided; using a fixed test behaviour.")
        p = TEST_P

    prob = BFFProblem(solvef=solvef)
    ops = ncp.flatten([prob.A, prob.B, prob.Z])  # Base monomials involved in problem
    obj = prob.objective(1, 0)  # Placeholder objective function
    chi = Wiring(p, c)

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
    return prob, sdp


# SCS, M+6, singlet, KEEP_M: Entropy: 0.9937548088160353, Delta: 0:30:22.533141
# SCS, M+6, singlet, !KEEP_M: Entropy: 0.9937548088160353, Delta: 0:28:11.155171
