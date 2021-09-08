from sympy import init_session

init_session()

nc, eta = symbols(r"n_c \eta", nonnegative=True)
Atlds = symbols(r"\langle\tilde{A}_0:3\rangle")
Btlds = symbols(r"\langle\tilde{B}_0:4\rangle")
ABtlds = [
    symbols(r"\langle\tilde{A}_" + str(x) + r"\tilde{B}_0:4\rangle")
    for x in range(0, 3)
]


def h(x):
    return -x * log(x, 2) - (1 - x) * log(1 - x, 2)


def phi(x):
    return h(0.5 + 0.5 * x)

'''
iA, oA, iB, oB = 2, 2, 3, 2
Eax = [-nc - (1 - nc) * ((1 - eta) - eta * Atlds[x]) for x in range(iA + 1)]
Eby = [-nc - (1 - nc) * ((1 - eta) - eta * Btlds[y]) for y in range(iB + 1)]
Eabxy = [
    [
        nc + (1 - nc) * (eta ** 2 * ABtlds[x][y]
            - eta * (1 - eta) * (Atlds[x] + Btlds[y])
            + (1 - eta) ** 2)
        for y in range(iB + 1)
    ]
    for x in range(iA + 1)
]

Q = (1 - Eabxy[1][3]) / 2  # QBER H(A|B)
S = Eabxy[1][1] + Eabxy[1][2] + Eabxy[2][1] - Eabxy[2][2]
HAE = (1 - phi(sqrt(S**2 / 4 - 1))).simplify()

dQdeta = diff(Q, eta).simplify()
dQdnc = diff(Q, nc).simplify()
dHAEdeta = diff(HAE, eta).simplify()
dHAEdnc = diff(HAE, nc).simplify()
'''

alp, q, s = symbols(r'\alpha q s')
R1 = sqrt((1-2*q)**2 + 4*q*(1-q)*(s**2/4 - alp**2))
R2 = sqrt(s**2/4 - alp**2)
g = 1 + phi(R1) - phi(R2)
dg = diff(g, s).simplify()
