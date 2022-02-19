"""
Script to compute converging lower bounds on the DIQKD rates of a protocol using
two devices that are constrained by some 2 input 2 output distribution.
More specifically, computes a sequence of lower bounds on the problem

            inf H(A|X=0,E) - H(A|X=0, Y=2, B)

where the infimum is over all quantum devices with some expected behaviour. See
the accompanying paper for more details (Figure 4)

Code also analyzes the scenario where we have inefficient detectors and implements
a subroutine to optimize the randomness gained from a family of two-qubit systems.
Also uses the noisy-preprocessing technique in order to boost rates. Bob is given
third input for key generation and doesn't bin his no-click when generating key.
"""

import numpy as np
from itertools import product as itprod
from math import log2, log, pi, cos, sin, prod
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import chaospy

DEFAULT_HALF_M = 6  # Number of nodes / 2 in gaussian quadrature
VERBOSE = 0  # If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-4, 1e-4  # Multiplicative/Additive epsilon in iterative optimization


def generate_quadrature(m):
    """
    Generates the Gaussian quadrature nodes t and weights w. Due to the way the
    package works it generates 2*M nodes and weights. Maybe consider finding a
    better package if want to compute for odd values of M.

         m    --    number of nodes in quadrature / 2
    """
    t, w = chaospy.quad_gauss_radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w


def cond_ent(joint, marg):
    """
    Returns H(A|B) = H(AB) - H(B)

    Inputs:
        joint    --     joint distribution on AB
        marg     --     marginal distribution on B
    """

    hab, hb = 0.0, 0.0

    for prob in joint:
        if 0.0 < prob < 1.0:
            hab += -prob * log2(prob)

    for prob in marg:
        if 0.0 < prob < 1.0:
            hb += -prob * log2(prob)

    return hab - hb


# TODO generalise beyond x* = 0 (zero-based indexing)
class BFFProblem(object):
    def __init__(self, **kwargs):
        # number of outputs for each input of Alice's / Bob's devices
        self.A_config = kwargs.get("A_config", [2, 2])
        self.B_config = kwargs.get("B_config", [2, 2, 2])
        self.update_half_m(kwargs.get("half_m", DEFAULT_HALF_M))

        # Operators in problem (only o-1 for o outputs, because the last
        # operator is enforced by normalisation). Here, A and B are measurement
        # operators
        self.A = [Ai for Ai in ncp.generate_measurements(self.A_config, "A")]
        self.B = [Bj for Bj in ncp.generate_measurements(self.B_config, "B")]
        self.Z = ncp.generate_operators("Z", max(self.A_config), hermitian=0)
        self.op_set = set(ncp.flatten([self.A, self.B, self.Z]))

        self.solvef = kwargs.get("solvef", lambda sdp: sdp.solve())
        self.verbose = kwargs.get("verbose", VERBOSE)
        self.safe = kwargs.get("safe", True)  # accept non-optimal solutions
        self.substitutions = self.get_subs()  # substitutions used in ncpol2sdpa

    def update_half_m(self, half_m):
        self.m = 2 * half_m
        self.T, self.W = generate_quadrature(
            half_m
        )  # Nodes, weights of quadrature

    def HAgEx_objective(self, ti, q):
        """
        Returns the objective function for the faster computations.
            Key generation on X=0
            Only two outcomes for Alice

            ti     --    i-th node
            q      --    bit flip probability
        """
        obj = 0.0
        F = [self.A[0][0], 1 - self.A[0][0]]  # POVM for self.Alices key gen measurement
        for a in range(self.A_config[0]):
            b = (a + 1) % 2  # (a + 1 mod 2)
            M = (1 - q) * F[a] + q * F[b]  # Noisy preprocessing povm element
            obj += M * (
                self.Z[a] + Dagger(self.Z[a]) + (1 - ti) * Dagger(self.Z[a]) * self.Z[a]
            ) + ti * self.Z[a] * Dagger(self.Z[a])

        return obj

    def compute_entropy(self, sdp, q=0):
        """
        Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

            sdp   --   sdp relaxation object
            q     --   probability of bitflip
        """
        ck = 0.0  # kth coefficient
        ent = 0.0  # lower bound on H(A|X=0,E)

        # We can also decide whether to perform the final optimization in the sequence
        # or bound it trivially. Best to keep it unless running into numerical problems
        # with it. Added a nontrivial bound when removing the final term
        # (WARNING: proof is not yet in the associated paper).

        bound_on_last = 2 * q * (1 - q) * self.W[-1] / log(2)
        m = len(self.T)
        for k in range(m):
            ck = self.W[k] / (self.T[k] * log(2))

            # Get the k-th objective function
            new_objective = self.HAgEx_objective(self.T[k], q)

            sdp.set_objective(new_objective)
            self.solvef(sdp)

            if self.verbose > 0:
                print("Status for i = ", k, ":", sdp.status)
            if not self.safe:  # ignore status and just add to entropy
                ent += ck * (1 + sdp.dual)
                continue

            if sdp.status == "optimal":
                # 1 contributes to the constant term
                ent += ck * (1 + sdp.dual)
            elif k == m:
                ent += bound_on_last
                if self.verbose > 0:
                    print("Could not solve last sdp, bounding its value")
            else:
                # If we didn't solve the sdp well enough then just bound the entropy
                # trivially
                ent = 0
                break

        return ent

    def analyse_behav(self, pabxy):
        """
        Generates the moment equality constraints for the distribution specified
        by the observed p(ab|xy), and searches through them to find all those that
        are relevant to the operators in objective_ops. Returns the relevant constraints
        and operators.

            pabxy          --     4D numpy array such that p[a, b, x, y] = p(ab|xy)
            objective_ops  --     list of operators in the objective
        """

        constraints = []
        oA, oB, iA, iB = pabxy.shape
        idx_ranges = [range(idx) for idx in [oA - 1, oB - 1, iA, iB]]
        for (a, b, x, y) in itprod(*idx_ranges):
            constraints.append(self.A[x][a] * self.B[y][b] - pabxy[a, b, x, y])
        # for marginals, select only input 0
        for (a, x) in itprod(range(oA - 1), range(iA)):
            constraints.append(
                self.A[x][a] - sum([pabxy[a, b, x, 0] for b in range(oB)])
            )
        for (b, y) in itprod(range(oB - 1), range(iB)):
            constraints.append(
                self.B[y][b] - sum([pabxy[a, b, 0, y] for a in range(oA)])
            )
        return constraints

    def optimise_q(self, sdp, sys, eta, q):
        """
        Optimizes the choice of q.

            sdp    --    sdp relaxation object
            sys    --    parameters of system that are optimized
            eta --     detection efficiency
            q     --     bitflip probability

        This function can probably be improved to make the search a bit more efficient and fine grained.
        """
        q_eps = 0.005  # Can be tuned
        q_eps_min = 0.001

        opt_q = q
        rate = self.compute_rate(sdp, sys, eta, q)  # Computes rate for given q
        starting_rate = rate

        # We check if we improve going left
        if q - q_eps < 0:
            LEFT = 0
        else:
            new_rate = self.compute_rate(sdp, sys, eta, opt_q - q_eps)
            if new_rate > rate:
                opt_q = opt_q - q_eps
                rate = new_rate
                if self.verbose > 0:
                    print(
                        "Optimizing q (eta,q) =",
                        (eta, opt_q),
                        " ... ",
                        starting_rate,
                        "->",
                        rate,
                    )
                LEFT = 1
            else:
                LEFT = 0

        def next_q(q0, step_size):
            q1 = q0 + ((-1) ** LEFT) * step_size
            if q1 >= 0 and q1 <= 0.5:
                return q1
            elif step_size / 2 >= q_eps_min:
                return next_q(q0, step_size / 2)
            else:
                return -1

        STILL_OPTIMIZING = 1

        while STILL_OPTIMIZING:
            # define the next q
            new_q = next_q(opt_q, q_eps)
            if new_q < 0:
                break

            # compute the rate
            new_rate = self.compute_rate(sdp, sys, eta, new_q)

            if new_rate > rate:
                opt_q = new_q
                rate = new_rate
                if self.verbose > 0:
                    print(
                        "Optimizing q (eta,q) =",
                        (eta, opt_q),
                        " ... ",
                        starting_rate,
                        "->",
                        rate,
                    )
            else:
                # If we didn't improve try shortening the distance
                q_eps = q_eps / 2
                if q_eps < q_eps_min:
                    STILL_OPTIMIZING = 0

        return rate, opt_q

    def optimise_rate(self, sdp, sys, eta, q):
        """
        Iterates between optimizing sys and optimizing q in order to optimize overall rate.
        """

        STILL_OPTIMIZING = 1

        best_rate = self.compute_rate(sdp, sys, eta, q)
        best_sys = sys[:]
        best_q = q

        while STILL_OPTIMIZING:
            _, new_sys = self.optimise_sys(sdp, best_sys[:], eta, best_q)
            new_rate, new_q = self.optimise_q(sdp, new_sys[:], eta, best_q)

            if (new_rate < best_rate + best_rate * EPS_M) or (
                new_rate < best_rate + EPS_A
            ):
                STILL_OPTIMIZING = 0

            if new_rate > best_rate:
                best_rate = new_rate
                best_sys = new_sys[:]
                best_q = new_q

        return best_rate, best_sys, best_q

    def get_subs(self):
        """
        Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
        commutation relations.
        """

        subs = {}
        # Get Alice and Bob's projective measurement constraints
        subs.update(ncp.projective_measurement_constraints(self.A, self.B))

        # Finally we note that Alice and Bob's operators should All commute with Eve's ops
        for a in ncp.flatten([self.A, self.B]):
            for z in self.Z:
                subs.update({z * a: a * z, Dagger(z) * a: a * Dagger(z)})

        return subs

    def get_extra_monomials(self, objective):
        """
        Additional monos: Alice-Bob-Eve products and monos in objective function
        """

        monos = []

        # Add ABZ
        ZZ = self.Z + [Dagger(z) for z in self.Z]
        Aflat = ncp.flatten(self.A)
        Bflat = ncp.flatten(self.B)
        monos += [a * b * z for (a, b, z) in itprod(Aflat, Bflat, ZZ)]

        # Add monos appearing in objective function
        add_args = objective.expand().args
        monos += [prod([a in self.op_set for a in aargs.args]) for aargs in add_args]

        return monos

    def get_extra_z_monomials(self, zs=2):
        """
        Returns additional monomials to add to sdp relaxation.
        """
        monos = []
        ZS = [
            z
            for z in ncp.nc_utils.get_monomials(self.Z, zs)
            if ncp.nc_utils.ncdegree(z) >= 2
        ]
        AB = ncp.flatten([self.A, self.B])
        for a in AB:
            for z in ZS:
                monos += [a * z]
        return monos[:]


# TODO add scalar variables, either as diagonal entries in the variable matrix,
# or as identity operators
