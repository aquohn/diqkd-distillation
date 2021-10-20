import numpy as np
from math import pi
import ncpol2sdpa as ncp
from glob import glob
from joblib import Parallel, delayed, parallel_backend
from qre import (
    BFFProblem,
    VERBOSE,
    LEVEL,
    M
)

# Some parameters for optimizing systems for a range of eta
NUM_SAMPLES = 2  # Number of random samples
RAND_PROB = 0.2  # Probability we choose uniformly our random system
THETA_VAR = pi / 128  # Variance for choosing random state angle
ANG_VAR = pi / 24  # Variance for choosing random measurement angle
NUM_WORKERS = 1  # Number of workers to split parallelization over

# Defining the test sys
test_sys = [pi / 4, 0, pi / 2, pi / 4, -pi / 4, 0]
test_eta = 0.99
test_q = 0.01

prob = BFFProblem()
ops = ncp.flatten([prob.A, prob.B, prob.Z])  # Base monomials involved in problem
obj = prob.objective(1, test_q)  # Placeholder objective function

sdp = ncp.SdpRelaxation(ops, verbose=VERBOSE - 1, normalized=True, parallel=0)
sdp.get_relaxation(
    level=LEVEL,
    equalities=prob.op_eqs[:],
    inequalities=prob.op_ineqs[:],
    momentequalities=prob.moment_eqs[:] + prob.score_constraints(test_sys, test_eta),
    momentinequalities=prob.moment_ineqs[:],
    objective=obj,
    substitutions=prob.substitutions,
    extramonomials=prob.extra_monos,
)


"""
We now have an sdp relaxation of our problem for the test system introduced above.

We can test it and try to optimize our parameters
"""
# ent = compute_entropy(sdp,test_q)
# err = HAgB(test_sys, test_eta, test_q)
# print(ent, err, ent-err)
# exit()
#
# new_rate, new_sys, new_q = optimise_rate(sdp, test_sys, test_eta, test_q)
# print(new_rate)
# print(new_sys)
# print(new_q)
# exit()


"""
Generating the rate plots for a range of detection efficiencies
"""
eta_range = (
    np.linspace(0.8, 0.85, 20)[:-1].tolist()
    + np.linspace(0.85, 0.95, 20)[:-1].tolist()
    + np.linspace(0.95, 1.0, 20).tolist()
)


# Define a function that we can distribute to a parallel processing job
def task(ETA):
    print("Starting eta ", ETA)
    fn = "./data/qkd_2322_" + str(2 * M) + "M_" + str(int(100000 * ETA)) + ".csv"

    # We want good sys choices to propogate through the optimization for different
    # eta so we load the data for nearby eta and we shall also optimize their best system
    known_systems = []
    for filename in list(glob("./data/qkd_2322_" + str(2 * M) + "M_*.csv")):
        data = np.loadtxt(filename, delimiter=",").tolist()
        if len(data) > 0:
            known_systems += [data]
    if len(known_systems) > 0:
        # order w.r.t. eta -- orders in ascending order
        known_systems.sort(key=lambda x: x[0])
        # Grabbing the systems either side of the system currently being optimized.
        try:
            idx = [i for i, x in enumerate(known_systems) if x[0] == ETA][0]
            if idx > 0:
                previous_sys = known_systems[idx - 1][:]
            else:
                previous_sys = [None]
            if idx < len(known_systems) - 1:
                next_sys = known_systems[idx + 1][:]
            else:
                next_sys = [None]
        except:
            previous_sys = [None]
            next_sys = [None]

    # If we have optimized before then open the data and collect best sys and hmin
    try:
        data = np.loadtxt(fn, delimiter=",").tolist()
        BEST_RATE = data[1]
        BEST_Q = data[2]
        BEST_SYS = data[3:]
        NEEDS_PRESOLVE = False
    except:
        # If we've never optimized before then try some ansatz (can be modified)
        BEST_RATE = -10
        BEST_Q = 0.0
        NEEDS_PRESOLVE = True
        BEST_SYS = [
            0.798381403026085,
            2.95356587736751,
            -1.79796447473952,
            2.05617041931652,
            0.653912312507124,
            2.95155725819389,
        ]

    # Attempt 0 -- FIRST ATTEMPT AT SUPPLIED DEFAULT SYSTEM
    # if ETA < 0.92:
    #     # Forcing it to solve a system we already know does well at low eta.
    #     NEEDS_PRESOLVE = True
    if NEEDS_PRESOLVE:
        try:
            print("Needs presolving, no file found...")
            curr_sys = [
                0.150415183475178,
                -0.0269996391717213,
                0.96107271555041,
                -0.565579360974102,
                0.115396460799296,
                -0.00955679420634085,
            ]
            BEST_Q = 0.0749375
            opt_rate, opt_sys, opt_q = prob.optimise_rate(sdp, curr_sys[:], ETA, BEST_Q)

            if opt_rate > BEST_RATE:
                print("New best rate for eta", ETA, ": ", BEST_RATE, " -> ", opt_rate)
                BEST_RATE = opt_rate
                BEST_SYS = opt_sys[:]
                BEST_Q = opt_q
        except:
            pass

    # ATTEMPT 1 -- TRY BEST SYSTEM FROM PREVIOUS POINT
    try:
        print("trying previous point's current best system...")
        curr_q = previous_sys[2]
        curr_sys = previous_sys[3:]
        opt_rate, opt_sys, opt_q = prob.optimise_rate(sdp, curr_sys[:], ETA, curr_q)

        if opt_rate > BEST_RATE:
            print("New best rate for eta", ETA, ": ", BEST_RATE, " -> ", opt_rate)
            BEST_RATE = opt_rate
            BEST_SYS = opt_sys[:]
            BEST_Q = opt_q
    except:
        pass

    # ATTEMPT 2 -- TRY BEST SYSTEM FROM NEXT POINT
    try:
        print("trying next point's current best system...")
        curr_q = next_sys[2]
        curr_sys = next_sys[3:]
        opt_rate, opt_sys, opt_q = prob.optimise_rate(sdp, curr_sys[:], ETA, curr_q)

        if opt_rate > BEST_RATE:
            print("New best rate for eta", ETA, ": ", BEST_RATE, " -> ", opt_rate)
            BEST_RATE = opt_rate
            BEST_SYS = opt_sys[:]
            BEST_Q = opt_q
    except:
        pass

    # ATTEMPT 3 -- TRY RANDOM POINTS
    try:
        print("trying random systems")
        for _ in range(NUM_SAMPLES):
            if np.random.random() < RAND_PROB:
                curr_sys = [
                    np.random.normal(BEST_SYS[0], THETA_VAR)
                ] + np.random.uniform(-pi, pi, len(BEST_SYS) - 1).tolist()
            else:
                curr_sys = np.random.normal(
                    BEST_SYS, [THETA_VAR] + [ANG_VAR for _ in range(len(BEST_SYS) - 1)]
                )

            opt_rate, opt_sys, opt_q = prob.optimise_rate(sdp, curr_sys[:], ETA, BEST_Q)

            if opt_rate > BEST_RATE:
                print("New best rate for eta", ETA, ": ", BEST_RATE, " -> ", opt_rate)
                BEST_RATE = opt_rate
                BEST_SYS = opt_sys[:]
                BEST_Q = opt_q
    except:
        pass

    np.savetxt(
        fn, [ETA, BEST_RATE, BEST_Q] + np.array(BEST_SYS).tolist(), delimiter=","
    )
    print("Finished eta ", ETA, " with rate ", BEST_RATE)
    return 0


# Run the optimization.
with parallel_backend("loky"):
    results = Parallel(n_jobs=NUM_WORKERS, verbose=0)(
        delayed(task)(eta) for eta in reversed(eta_range)
    )
