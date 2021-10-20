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
from math import log2, log, pi, cos, sin
import ncpol2sdpa as ncp
import qutip as qtp
from scipy.optimize import minimize
from sympy.physics.quantum.dagger import Dagger

# import mosek
import chaospy


LEVEL = 2  # NPA relaxation level
M = 6  # Number of nodes / 2 in gaussian quadrature
KEEP_M = 0  # Optimizing mth objective function?
VERBOSE = 0  # If > 1 then ncpol2sdpa will also be verbose
EPS_M, EPS_A = 1e-4, 1e-4  # Multiplicative/Additive epsilon in iterative optimization
NUM_SUBWORKERS = 4  # Number of cores each worker has access to


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


def HAgB(sys, eta, q):
    """
    Computes the error correction term in the key rate for a given system,
    a fixed detection efficiency and noisy preprocessing. Computes the relevant
    components of the distribution and then evaluates the conditional entropy.

        sys    --    parameters of system
        eta    --    detection efficiency
        q      --    bitflip probability
    """

    # Computes H(A|B) required for rate
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [theta, a0, a1, b0, b1, b2] = sys[:]
    rho = (cos(theta) * qtp.ket("00") + sin(theta) * qtp.ket("11")).proj()

    # Noiseless measurements
    a00 = 0.5 * (id + cos(a0) * sz + sin(a0) * sx)
    b20 = 0.5 * (id + cos(b2) * sz + sin(b2) * sx)

    # Alice bins to 0 transforms povm
    A00 = eta * a00 + (1 - eta) * id
    # Final povm transformation from the bitflip
    A00 = (1 - q) * A00 + q * (id - A00)
    A01 = id - A00

    # Bob has inefficient measurement but doesn't bin
    B20 = eta * b20
    B21 = eta * (id - b20)
    B22 = (1 - eta) * id

    # joint distribution
    q00 = (rho * qtp.tensor(A00, B20)).tr().real
    q01 = (rho * qtp.tensor(A00, B21)).tr().real
    q02 = (rho * qtp.tensor(A00, B22)).tr().real
    q10 = (rho * qtp.tensor(A01, B20)).tr().real
    q11 = (rho * qtp.tensor(A01, B21)).tr().real
    q12 = (rho * qtp.tensor(A01, B22)).tr().real

    qb0 = (rho * qtp.tensor(id, B20)).tr().real
    qb1 = (rho * qtp.tensor(id, B21)).tr().real
    qb2 = (rho * qtp.tensor(id, B22)).tr().real

    qjoint = [q00, q01, q02, q10, q11, q12]
    qmarg = [qb0, qb1, qb2]

    return cond_ent(qjoint, qmarg)


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


def sys2vec(sys, eta=1.0):
    """
    Returns a vector of probabilities determined from the system in the same order as specified
    in the function score_constraints()

        sys    --     system parameters
        eta    --     detection efficiency
    """
    # Get the system from the parameters
    [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
    [theta, a0, a1, b0, b1, b2] = sys[:]
    rho = (cos(theta) * qtp.ket("00") + sin(theta) * qtp.ket("11")).proj()

    # Define the first projectors for each of the measurements of Alice and Bob
    a00 = 0.5 * (id + cos(a0) * sz + sin(a0) * sx)
    a01 = id - a00
    a10 = 0.5 * (id + cos(a1) * sz + sin(a1) * sx)
    a11 = id - a10
    b00 = 0.5 * (id + cos(b0) * sz + sin(b0) * sx)
    b01 = id - b00
    b10 = 0.5 * (id + cos(b1) * sz + sin(b1) * sx)
    b11 = id - b10

    A_meas = [[a00, a01], [a10, a11]]
    B_meas = [[b00, b01], [b10, b11]]

    vec = []

    # Get p(00|xy)
    for x in range(2):
        for y in range(2):
            vec += [
                (
                    eta ** 2 * (rho * qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real
                    + +eta
                    * (1 - eta)
                    * (
                        (rho * qtp.tensor(A_meas[x][0], id)).tr().real
                        + (rho * qtp.tensor(id, B_meas[y][0])).tr().real
                    )
                    + +(1 - eta) * (1 - eta)
                )
            ]

    # And now the marginals
    vec += [eta * (rho * qtp.tensor(A_meas[0][0], id)).tr().real + (1 - eta)]
    vec += [eta * (rho * qtp.tensor(id, B_meas[0][0])).tr().real + (1 - eta)]
    vec += [eta * (rho * qtp.tensor(A_meas[1][0], id)).tr().real + (1 - eta)]
    vec += [eta * (rho * qtp.tensor(id, B_meas[1][0])).tr().real + (1 - eta)]

    return vec


class BFFProblem:
    def __init__(self, **kwargs):
        self.T, self.W = generate_quadrature(M)  # Nodes, weights of quadrature

        # number of outputs for each inputs of Alice / Bobs devices
        # (Dont need to include 3rd input for Bob here as we only constrain the statistics
        # for the other inputs).
        self.A_config = kwargs.get("A_config", [2, 2])
        self.B_config = kwargs.get("B_config", [2, 2])
        self.M = kwargs.get("M", 6)
        self.solvef = kwargs.get("solvef", lambda sdp: sdp.solve(
            "mosek", solverparameters={"num_threads": int(NUM_SUBWORKERS)}))

        # Operators in problem
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
                if VERBOSE:
                    print("Bad solve: ", k, SDP.status)
                break

        return ent

    def compute_rate(self, SDP, sys, eta, q):
        """
        Computes a lower bound on the rate H(A|X=0,E) - H(A|X=0,Y=2,B) using the fast
        method

            SDP       --     sdp relaxation object
            sys       --     parameters of the system
            eta       --     detection efficiency
            q         --     bitflip probability
        """
        score_cons = self.score_constraints(sys[:], eta)
        SDP.process_constraints(
            equalities=self.op_eqs,
            inequalities=self.op_ineqs,
            momentequalities=self.moment_eqs[:] + score_cons[:],
            momentinequalities=self.moment_ineqs,
        )
        ent = self.compute_entropy(SDP, q)
        err = HAgB(sys, eta, q)
        return ent - err

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

    def score_constraints(self, sys, eta=1.0):
        """
        Returns the moment equality constraints for the distribution specified by the
        system sys and the detection efficiency eta. We only look at constraints coming
        from the inputs 0/1. Potential to improve by adding input 2 also?

            sys    --     system parameters
            eta    --     detection efficiency
        """

        # Extract the system
        [id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
        [theta, a0, a1, b0, b1, b2] = sys[:]
        rho = (cos(theta) * qtp.ket("00") + sin(theta) * qtp.ket("11")).proj()

        # Define the first projectors for each of the measurements of Alice and Bob
        a00 = 0.5 * (id + cos(a0) * sz + sin(a0) * sx)
        a01 = id - a00
        a10 = 0.5 * (id + cos(a1) * sz + sin(a1) * sx)
        a11 = id - a10
        b00 = 0.5 * (id + cos(b0) * sz + sin(b0) * sx)
        b01 = id - b00
        b10 = 0.5 * (id + cos(b1) * sz + sin(b1) * sx)
        b11 = id - b10

        A_meas = [[a00, a01], [a10, a11]]
        B_meas = [[b00, b01], [b10, b11]]

        constraints = []

        # Add constraints for p(00|xy)
        for x in range(2):
            for y in range(2):
                constraints += [
                    self.A[x][0] * self.B[y][0]
                    - (
                        eta ** 2
                        * (rho * qtp.tensor(A_meas[x][0], B_meas[y][0])).tr().real
                        + +eta
                        * (1 - eta)
                        * (
                            (rho * qtp.tensor(A_meas[x][0], id)).tr().real
                            + (rho * qtp.tensor(id, B_meas[y][0])).tr().real
                        )
                        + +(1 - eta) * (1 - eta)
                    )
                ]

        # Now add marginal constraints p(0|x) and p(0|y)
        constraints += [
            self.A[0][0]
            - eta * (rho * qtp.tensor(A_meas[0][0], id)).tr().real
            - (1 - eta)
        ]
        constraints += [
            self.B[0][0]
            - eta * (rho * qtp.tensor(id, B_meas[0][0])).tr().real
            - (1 - eta)
        ]
        constraints += [
            self.A[1][0]
            - eta * (rho * qtp.tensor(A_meas[1][0], id)).tr().real
            - (1 - eta)
        ]
        constraints += [
            self.B[1][0]
            - eta * (rho * qtp.tensor(id, B_meas[1][0])).tr().real
            - (1 - eta)
        ]

        return constraints[:]

    def optimise_sys(self, SDP, sys, eta, q):
        """
        Optimizes the rate using the iterative method via the dual vectors.

            SDP    --    sdp relaxation object
            sys    --    parameters of system that are optimized
            eta    --    detection efficiency
            q      --    bitflip probability
        """

        NEEDS_IMPROVING = True  # Flag to check if needs optimizing still
        FIRST_PASS = True  # Checks if first time through loop
        improved_sys = sys[:]  # Improved choice of system
        best_sys = sys[:]  # Best system found
        dual_vec = np.zeros(8)  # Dual vector same length as num constraints

        # Loop until we converge on something
        while NEEDS_IMPROVING:
            # On the first loop we just solve and extract the dual vector
            if not FIRST_PASS:
                # Here we optimize the dual vector
                # The distribution associated with the improved system
                pstar = sys2vec(improved_sys[:], eta)

                # function to optimize parameters over
                def f0(x):
                    # x is sys that we are optimizing
                    p = sys2vec(x, eta)
                    return -np.dot(p, dual_vec) + HAgB(x, eta, q)

                # Bounds on the parameters of sys
                bounds = [[0, pi / 2]] + [[-pi, pi] for _ in range(len(sys) - 1)]
                # Optmize qubit system (maximizing due to negation in f0)
                res = minimize(f0, improved_sys[:], bounds=bounds)
                improved_sys = res.x.tolist()[:]  # Extract optimizer

            # Apply the new system to the sdp
            score_cons = self.score_constraints(improved_sys[:], eta)
            SDP.process_constraints(
                equalities=self.op_eqs,
                inequalities=self.op_ineqs,
                momentequalities=self.moment_eqs[:] + score_cons[:],
                momentinequalities=self.moment_ineqs,
            )

            # Compute new dual vector and the rate
            dual_vec, new_ent = self.compute_dual_vector(SDP, q)
            new_rate = new_ent - HAgB(improved_sys[:], eta, q)

            if FIRST_PASS:
                # If first run through then this is the initial entropy
                starting_rate = new_rate
                best_rate = new_rate
                FIRST_PASS = False
            else:
                if (
                    new_rate < best_rate + best_rate * EPS_M
                    or new_rate < best_rate + EPS_A
                ):
                    NEEDS_IMPROVING = False

            if new_rate > best_rate:
                if VERBOSE > 0:
                    print(
                        "Optimizing sys (eta, q) =",
                        (eta, q),
                        " ... ",
                        starting_rate,
                        "->",
                        new_rate,
                    )
                best_rate = new_rate
                best_sys = improved_sys[:]

        return best_rate, best_sys[:]

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
