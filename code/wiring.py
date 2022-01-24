from sympy import init_session
from sympy.logic.boolalg import Xor

init_session()

# Expt model

nc, eta = symbols(r"n_c \eta", nonnegative=True)
Atld = symbols(r"\langle\tilde{A}\rangle")
Btld = symbols(r"\langle\tilde{B}\rangle")
ABtld = symbols(r"\langle\tilde{A}\tilde{B}\rangle")


def h(x):
    return -x * log(x, 2) - (1 - x) * log(1 - x, 2)


def phi(x):
    return h(0.5 + 0.5 * x)


iA, oA, iB, oB = 2, 2, 3, 2
Eax = -nc - (1 - nc) * ((1 - eta) - eta * Atld)
Eby = -nc - (1 - nc) * ((1 - eta) - eta * Btld)
Eabxy = nc + (1 - nc) * (
    eta ** 2 * ABtld - eta * (1 - eta) * (Atld + Btld) + (1 - eta) ** 2
)
dEabxydnc = diff(Eabxy, nc).simplify()
dEabxydeta = diff(Eabxy, eta).simplify()

Q, S = symbols(r"Q S", real=True)
HAE = (1 - phi(sqrt(S ** 2 / 4 - 1))).simplify()
dHAEdS = diff(HAE, S).simplify()

# Asym CHSH

alp, q, s = symbols(r"\alpha q s")
R1 = sqrt((1 - 2 * q) ** 2 + 4 * q * (1 - q) * (s ** 2 / 4 - alp ** 2))
R2 = sqrt(s ** 2 / 4 - alp ** 2)
g = 1 + phi(R1) - phi(R2)
dg = diff(g, s).simplify()

# Maximal correlation
ps = [symbols(r"p_{%u\,0:2}" % i, real=True) for i in range(2)]
PAB = Matrix([[ps[0][0], ps[0][1]], [ps[1][0], ps[1][1]]])
PA = Matrix([[ps[0][0] + ps[0][1], 0], [0, ps[1][0] + ps[1][1]]])
PB = Matrix([[ps[0][0] + ps[1][0], 0], [0, ps[0][1] + ps[1][1]]])
Ptld = sqrt(PA) * PAB * sqrt(PB)


def reg_to_pauli(M):
    R = [[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0], [0, 0.5j, 0.5j, 0], [0.5, 0, 0, -0.5]]
    return Matrix(R) * Matrix([M[0, 0], M[0, 1], M[1, 0], M[1, 1]])


def pauli_to_sv(v):
    p = sum([abs(z) ** 2 for z in v])
    m = v[1] ** 2 - sum([z ** 2 for z in v[1:4]])
    pm = sqrt(abs(p) ** 2 - abs(m) ** 2)
    return sqrt(p + pm), sqrt(p - pm)


lambdap, lambdam = pauli_to_sv(reg_to_pauli(Ptld))

# Twiriling
def twirling_actions():
    a, b, x, y, alp, bet, gam = symbols(r'a b x y \alpha \beta \gamma')
    xp = x ^ alp
    ap = a ^ (bet & x) ^ (alp & bet) ^ gam
    yp = y ^ bet
    bp = b ^ (alp & y) ^ gam

    for tup in itprod(range(2), range(2), range(2)):
        print(f"Alpha = {tup[0]}, Beta = {tup[1]}, Gamma = {tup[2]}")
        print(latex((ap.subs({alp: tup[0], bet: tup[1], gam: tup[2]})))
        print(latex((bp.subs({alp: tup[0], bet: tup[1], gam: tup[2]})))
        print(latex((xp.subs({alp: tup[0], bet: tup[1], gam: tup[2]})))
        print(latex((yp.subs({alp: tup[0], bet: tup[1], gam: tup[2]})))
        print()
