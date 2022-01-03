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
from math import log2, log, pi, cos, sin
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import chaospy

# import mosek


M = 6  # Number of nodes / 2 in gaussian quadrature
KEEP_M = 0  # Optimizing mth objective function?
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


def sdp_dual_vec(SDP):
    """
    Extracts the dual vector from the solved sdp by ncpol2sdpa

        SDP -- sdp relaxation object

    Would need to be modified if the number of moment constraints or their
    nature (equalities vs inequalities) changes.
    """
    # TODO generalise
    raw_vec = SDP.y_mat[-16:]
    vec = [0 for _ in range(8)]
    for k in range(8):
        vec[k] = raw_vec[2 * k][0][0] - raw_vec[2 * k + 1][0][0]
    return np.array(vec[:])


class BFFProblem(object):
    def __init__(self, **kwargs):
        self.T, self.W = generate_quadrature(M)  # Nodes, weights of quadrature

        # number of outputs for each inputs of Alice / Bobs devices
        # (Dont need to include 3rd input for Bob here as we only constrain the statistics
        # for the other inputs).
        self.A_config = kwargs.get("A_config", [2, 2])
        self.B_config = kwargs.get("B_config", [2, 2, 2])
        self.M = kwargs.get("M", 6)
        self.solvef = kwargs.get("solvef", lambda sdp: sdp.solve())

        # Operators in problem (only o-1 for o outputs, because the last
        # operator is enforced by normalisation)
        self.A = [Ai for Ai in ncp.generate_measurements(self.A_config, "A")]
        self.B = [Bj for Bj in ncp.generate_measurements(self.B_config, "B")]
        self.Z = ncp.generate_operators("Z", 2, hermitian=0)

        self.substitutions = self.get_subs()  # substitutions used in ncpol2sdpa
        self.moment_ineqs = []  # moment inequalities
        self.moment_eqs = []  # moment equalities
        self.op_eqs = []  # operator equalities
        self.op_ineqs = []  # operator inequalities
        self.extra_monos = self.get_extra_monomials()  # extra monomials

    def objective(self, ti, q):
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

    def compute_entropy(self, SDP, q):
        """
        Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

            SDP   --   sdp relaxation object
            q     --   probability of bitflip
        """
        SDP.process_constraints(
            equalities=self.op_eqs,
            inequalities=self.op_ineqs,
            momentequalities=self.moment_eqs[:],
            momentinequalities=self.moment_ineqs,
        )
        ck = 0.0  # kth coefficient
        ent = 0.0  # lower bound on H(A|X=0,E)

        # We can also decide whether to perform the final optimization in the sequence
        # or bound it trivially. Best to keep it unless running into numerical problems
        # with it. Added a nontrivial bound when removing the final term
        # (WARNING: proof is not yet in the associated paper).
        if KEEP_M:
            num_opt = len(self.T)
        else:
            num_opt = len(self.T) - 1
            ent = 2 * q * (1 - q) * self.W[-1] / log(2)

        for k in range(num_opt):
            ck = self.W[k] / (self.T[k] * log(2))

            # Get the k-th objective function
            new_objective = self.objective(self.T[k], q)

            SDP.set_objective(new_objective)
            self.solvef(SDP)

            if SDP.status == "optimal":
                # 1 contributes to the constant term
                ent += ck * (1 + SDP.dual)
            else:
                # If we didn't solve the SDP well enough then just bound the entropy
                # trivially
                ent = 0
                if VERBOSE > 0:
                    print("Bad solve: ", k, SDP.status)
                break

        return ent

    def compute_dual_vector(self, SDP, q):
        """
        Extracts the vector from the dual problem(s) that builds into the affine function
        of the constraints that lower bounds H(A|X=0,E)

            SDP    --     sdp relaxation object
            q      --     probability of bitflip
        """

        dual_vec = np.zeros(8)  # dual vector
        ck = 0.0  # kth coefficient
        ent = 0.0  # lower bound on H(A|X=0,E)

        if KEEP_M:
            num_opt = len(self.T)
        else:
            num_opt = len(self.T) - 1
            ent = 2 * q * (1 - q) * self.W[-1] / log(2)

        # Compute entropy and build dual vector from each sdp solved
        for k in range(num_opt):
            ck = self.W[k] / (self.T[k] * log(2))

            # Get the k-th objective function
            new_objective = self.objective(self.T[k], q)

            # Set the objective and solve
            SDP.set_objective(new_objective)
            self.solvef(SDP)

            # Check solution status
            if SDP.status == "optimal":
                ent += ck * (1 + SDP.dual)
                # Extract the dual vector from the solved sdp
                d = sdp_dual_vec(SDP)
                # Add the dual vector to the total dual vector
                dual_vec = dual_vec + ck * d
            else:
                ent = 0
                dual_vec = np.zeros(8)
                break

        return dual_vec, ent

    def behav_eqs(self, pabxy):
        """
        Returns the moment equality constraints for the distribution specified by the
        observed p(ab|xy). Note that we ignore constraints for a, b = 1; as in the
        Collins-Gisin representation, these are unecessary.

            pabxy  --     4D numpy array such that p[a, b, x, y] = p(ab|xy)
        """

        constraints = []
        oA, oB, iA, iB = pabxy.shape
        for (a, b, x, y) in itprod(range(oA - 1), range(oB - 1), range(iA), range(iB)):
            constraints.append(self.A[x][a] * self.B[y][b] - pabxy[a, b, x, y])
        for (a, x) in itprod(range(oA - 1), range(iA)):
            constraints.append(
                self.A[x][a]
                - sum([pabxy[a, b, x, y] for (b, y) in itprod(range(oB), range(iB))])
            )
        for (b, y) in itprod(range(oB - 1), range(iB)):
            constraints.append(
                self.B[y][b]
                - sum([pabxy[a, b, x, y] for (a, x) in itprod(range(oA), range(iA))])
            )
        return constraints

    def optimise_q(self, SDP, sys, eta, q):
        """
        Optimizes the choice of q.

            SDP    --    sdp relaxation object
            sys    --    parameters of system that are optimized
            eta --     detection efficiency
            q     --     bitflip probability

        This function can probably be improved to make the search a bit more efficient and fine grained.
        """
        q_eps = 0.005  # Can be tuned
        q_eps_min = 0.001

        opt_q = q
        rate = self.compute_rate(SDP, sys, eta, q)  # Computes rate for given q
        starting_rate = rate

        # We check if we improve going left
        if q - q_eps < 0:
            LEFT = 0
        else:
            new_rate = self.compute_rate(SDP, sys, eta, opt_q - q_eps)
            if new_rate > rate:
                opt_q = opt_q - q_eps
                rate = new_rate
                if VERBOSE > 0:
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
            new_rate = self.compute_rate(SDP, sys, eta, new_q)

            if new_rate > rate:
                opt_q = new_q
                rate = new_rate
                if VERBOSE > 0:
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

    def optimise_rate(self, SDP, sys, eta, q):
        """
        Iterates between optimizing sys and optimizing q in order to optimize overall rate.
        """

        STILL_OPTIMIZING = 1

        best_rate = self.compute_rate(SDP, sys, eta, q)
        best_sys = sys[:]
        best_q = q

        while STILL_OPTIMIZING:
            _, new_sys = self.optimise_sys(SDP, best_sys[:], eta, best_q)
            new_rate, new_q = self.optimise_q(SDP, new_sys[:], eta, best_q)

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

    def get_extra_monomials(self):
        """
        Returns additional monomials to add to sdp relaxation.
        """

        monos = []

        # Add ABZ
        ZZ = self.Z + [Dagger(z) for z in self.Z]
        Aflat = ncp.flatten(self.A)
        Bflat = ncp.flatten(self.B)
        for a in Aflat:
            for b in Bflat:
                for z in ZZ:
                    monos += [a * b * z]

        # Add monos appearing in objective function
        for z in self.Z:
            monos += [self.A[0][0] * Dagger(z) * z]

        return monos[:]
