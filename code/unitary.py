from sympy import Matrix, prod, exp, cos, sin, symbols, I, init_printing
from sympy.physics.quantum import TensorProduct as kron
from sympy.physics.quantum import Dagger
from itertools import product as itprod

init_printing()


def gen_diagmat(L):
    def diag(i, j):
        if i == j:
            return L[i]
        else:
            return 0

    return diag


def gen_diagproj(x):
    def diagproj(i, j):
        if i == x and j == x:
            return 1
        else:
            return 0

    return diagproj


def gen_antisym(x, y):
    def antisym(i, j):
        if x == i and y == j:
            return -I
        elif x == j and y == i:
            return I
        else:
            return 0

    return antisym


def gen_projmat(x, lam):
    def proj(i, j):
        if i == x and j == x:
            return exp(I * lam)
        elif i == j:
            return 1
        else:
            return 0

    return proj


def gen_rotmat(x, y, lamxy, lamyx):
    def rot(i, j):
        if i == x and j == y:
            return sin(lamxy)
        elif i == y and j == x:
            return -exp(I * lamyx) * sin(lamxy)
        elif i == x and j == x:
            return cos(lamxy)
        elif i == y and j == y:
            return exp(I * lamyx) * cos(lamxy)
        elif i == j:
            return 1
        else:
            return 0

    return rot


def projmat(x, lam, d):
    return Matrix(d, d, gen_projmat(x, lam))


def rotmat(x, y, lamxy, lamyx, d):
    return Matrix(d, d, gen_rotmat(x, y, lamxy, lamyx))


def antisymmat(x, y, lamxy, lamyx, d):
    return exp(I * lamyx * Matrix(d, d, gen_diagproj(y))) * exp(I * lamxy * Matrix(d, d, gen_antisym(x, y)))


def unitary(L):
    return prod(unitarylist(L))


def unitarylist(L):
    d = min(L.shape)
    phases = [exp(I * L[i, i]) for i in range(d)]
    phasemat = Matrix(d, d, gen_diagmat(phases))
    ulist = []
    for m in range(0, d - 1):
        ulist += [rotmat(m, n, L[m, n], L[n, m], d) for n in range(m + 1, d)]
    return ulist + [phasemat]


# first k columns generate the eigenprojectors of a rank-k density matrix
def densmatunitary(L, k=0):
    d = min(L.shape)
    if k == 0 or k >= d:
        k = d - 1
    ulist = []
    for m in range(0, k):
        ulist += [rotmat(m, n, L[m, n], L[n, m], d) for n in range(m + 1, d)]
    return prod(ulist)


# first k columns are a basis for each possible k-dimensional subspace
# k = d - 1 => columns are basis for whole space
def basisunitary(L, k=0):
    d = min(L.shape)
    if k == 0 or k >= d:
        k = d - 1
    ulist = []
    for m in range(0, k):
        ulist += [rotmat(m, n, L[m, n], L[n, m], d) for n in range(k, d)]
    return prod(ulist)


def symbmat(d, sym):
    return Matrix(
        d,
        d,
        lambda i, j: symbols(sym + "_{" + str(i) + ";" + str(j) + "}", complex=True),
    )


def diagmat(d, sym):
    L = [symbols(sym + "_{" + str(i) + "}") for i in range(d)]
    return Matrix(d, d, gen_diagmat(L))


def symL(d):
    return symbmat(d, r"\lambda")


def paramet_povm(o, d, sym=r"\lambda"):
    L = symbmat(o * d, sym)
    U = densmatunitary(L, d)  # phases will be lost anyway
    povm = []
    for a in range(o):
        sqrtMa = U[a * d:(a + 1) * d, 0:d]
        povm.append(Dagger(sqrtMa) * sqrtMa)
    return povm, L


def paramet_densmat(k, d, Usym=r"\lambda", psym=r"p"):
    L = symbmat(d, Usym)
    U = densmatunitary(L, d)  # phases will be lost anyway
    rhop = diagmat(d, psym)
    return U * rhop * Dagger(U), rhop, L
