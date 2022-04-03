using Revise
using Combinatorics
using LinearAlgebra, SparseArrays, QuantumInformation
using SymEngine

includet("helpers.jl")
includet("nonlocality.jl")
includet("maxcorr.jl")

const vLl = 999 * 689 * 10^-6 * cos(pi/50)^4
const vNL = 0.6964
vcrit(vL) = (vL+1)/(3-vL)

kd(i,j) = (i == j) ? 1 : 0
E(M, rho) = tr(M * rho)
const sigmas = [[kd(j, 3) kd(j,1)-im*kd(j,2); kd(j,1)+im*kd(j,2) -kd(j,3)] for j in 1:3]
const sigma_eigs = eigen.(sigmas)
Emat(d, x, y) = sparse([x], [y], [1], d, d)
# Generalised Gell-Mann matrices
function tr0_herm_basis(n::Integer)
  Emats = [Emat(n, j, k) for j in 1:n, k in 1:n]
  T = typeof(n)
  idxs = Vector{Tuple{T,T}}(undef, Int(n * (n-1) / 2))
  pos = 1
  for j in 1:n
    for k in 1:j
      j != k || continue
      idxs[pos] = (j,k)
      pos += 1
    end
  end
  syms = [Emats[k,j] + Emats[j,k] for (j,k) in idxs]
  antisyms = [-im * (Emats[k,j] - Emats[j,k]) for (j,k) in idxs]
  diags = [sqrt(2/(l*(l+1))) * sum(Emats[j,j] - l*Emats[l+1,l+1] for j in 1:l)
           for l in 1:n-1]
  return vcat(syms, antisyms, diags)
end

# Impt states
const singlet_corrs = Correlators([0, 0],
                                 [0, 0, 0],
                                 [1/sqrt(2) 1/sqrt(2) 1;
                                  1/sqrt(2) -1/sqrt(2) 0])

const bound_corrs = Correlators([0, 0],
                         [0, 0, 0],
                         [1//2 1//2 1//2;
                          1//2 -1//2 1//2])

const PR_corrs = Correlators([0, 0],
                      [0, 0, 0],
                      [1 1 1;
                       1 -1 1])

const mix_corrs = Correlators([0, 0],
                       [0, 0, 0],
                       [0 0 0;
                        0 0 0])

const LD_corrs = Correlators([1, 1],
                      [1, 1, 1],
                      [1 1 1;
                       1 1 1])

werner_corrs(v) = convexsum([v, 1-v], [singlet_corrs, mix_corrs])
const v_L = 0.6829; const v_NL = 0.6964; const v_crit = 0.7263

const ket0 = ket(1,2)
const ket1 = ket(2,2)
const ketp = (ket0 + ket1)/sqrt(2)
const ketm = (ket0 - ket1)/sqrt(2)
const phip = (kron(ket0, ket0) + kron(ket1, ket1)) / sqrt(2)
const phim = (kron(ket0, ket0) - kron(ket1, ket1)) / sqrt(2)
const psip = (kron(ket0, ket1) + kron(ket1, ket0)) / sqrt(2)
const psim = (kron(ket0, ket1) - kron(ket1, ket0)) / sqrt(2)

# measurements that saturate the bound on the standard protocol
function bell_diag_corrs(C)
  Zmul = 1/sqrt(1+C^2)
  Xmul = C/sqrt(1+C^2)
  As = [sigmas[3], sigmas[1]]
  Bs = [Zmul * sigmas[3] + Xmul * sigmas[1],
        Zmul * sigmas[3] - Xmul * sigmas[1],
        sigmas[3]]
  rho = ((1+C)/2) * proj(phip) + ((1-C)/2) * proj(phim)

  return Correlators(rho, As, Bs)
end

# POVMs and observables
observable(M::POVMMeasurement, vals::AbstractVector{<:Number}) = vals' * M.matrices
function POVMMeasurement(obsv::T) where {T <: AbstractMatrix{<:Number}}
  decomp = eigen(obsv)
  d = first(size(obsv))
  ops = [proj(decomp.vectors[:, i]) for i in 1:d]
  return POVMMeasurement(ops)
end
function Behaviour(rho::AbstractMatrix{<:Number}, Ms::AbstractVector{<:POVMMeasurement}, Ns::AbstractVector{<:POVMMeasurement})
  iA = length(Ms); iB = length(Ns)
  dA = first(Ms).idim; dB = first(Ns).idim;
  oA = first(Ms).odim; oB = first(Ns).odim;
  @assert all([M.odim == oA && M.idim == dA for M in Ms])
  @assert all([N.odim == oB && N.idim == dB for N in Ns])
  types = [eltype(rho); 
           [[eltype(Mmat) for Mmat in M.matrices] for M in Ms]...;
           [[eltype(Nmat) for Nmat in N.matrices] for N in Ns]...]
  T = promote_type(types...) |> real

  p = Array{T}(undef, oA, oB, iA, iB)
  # TODO use CG representation to reduce number of computations
  for (x, y) in itprod(1:iA, 1:iB)
    M = Ms[x]; N = Ns[y]
    for (a, b) in itprod(1:oA, 1:oB)
      Ma = M.matrices[a]; Nb = N.matrices[b]
      p[a, b, x, y] = tr(rho * kron(Ma, Nb)) |> real
    end
  end
  return Behaviour(p)
end

function Correlators(rho::AbstractMatrix{<:Number}, As::AbstractVector{<:AbstractMatrix}, Bs::AbstractVector{<:AbstractMatrix})
  iA = length(As); iB = length(Bs)
  dA = As |> first |> size |> first; dB = Bs |> first |> size |> first;
  oA = 2; oB = 2;

  rhoA = ptrace(rho, [dA, dB], 2)
  rhoB = ptrace(rho, [dA, dB], 1)
  Eabxy = [tr(rho * kron(As[x], Bs[y])) |> real for x in 1:iA, y in 1:iB]
  Eax = [tr(rhoA * As[x]) |> real for x in 1:iA]
  Eby = [tr(rhoB * Bs[y]) |> real for y in 1:iB]

  return Correlators(Eax, Eby, Eabxy)
end



# operator parametrisations
symbmat(d, sym) = [symbols("$(sym)_{$i;$j}") for i in 1:d, j in 1:d]
diagmat(v::AbstractVector) = sparse(1:length(v), 1:length(v), v)
diagmat(d, sym::AbstractString) = diagmat([symbols("$sym_{$i}") for i in 1:d])
function rotmat(d, x, y, lamxy, lamyx)
  T = promote_type(typeof(lamxy), typeof(lamyx))
  mat = Matrix{T}(I(d))
  mat[x, x] = cos(lamxy)
  mat[x, y] = sin(lamxy)
  mat[y, x] = -exp(im * lamyx) * sin(lamxy)
  mat[y, y] = exp(im * lamyx) * cos(lamxy)
  return mat
end
rotmat(d, x, y, L::AbstractMatrix) = rotmat(d, x, y, L[x, y], L[y, x])

function unitarylist(L::AbstractMatrix)
  d = minimum(size(L))
  ulist = [rotmat(d, m, n, L) for m in 1:(d-1) for n in (m+1):d]
  phases = [exp(im * L[i, i]) for i in 1:d]
  push!(ulist, diagmat(phases))
  
end
unitary(L) = prod(unitarylist(L))
function densmat_unitarylist(L::AbstractMatrix, k=0)
  d = minimum(size(L))
  if !(0 < k < d)
    k = d - 1
  end
  ulist = [rotmat(d, m, n, L) for m in 1:k for n in (m+1):d]
  return ulist
end
densmat_unitary(L, k=0) = prod(densmat_unitarylist(L, k))

function paramet_povm(o, d; sym=raw"\lambda")
  L = symbmat(o * d, sym)
  U = densmatunitary(L, d)  # phases will be lost anyway
  povm = Vector{typeof(U)}(undef, o)
  for a in 1:o
    sqrtMa = U[(a-1)*d+1:a*d, 1:d]
    povm[a] = sqrtMa' * sqrtMa
  end
  return povm, L
end

function paramet_densmat(k, d; Usym=raw"\lambda", psym=raw"p")
  L = symbmat(d, Usym)
  U = densmatunitary(L, d)  # phases will be lost anyway
  rhop = diagmat(d, psym)
  return U * rhop * U', rhop, L
end

# default to Lebesgue measure
randPOVM(d, k::Integer, n::Integer=d) = randPOVM(d, repeat([n], k))
function randPOVM(d, ns::AbstractVector{<:Integer})
    T = promote_type(typeof(d), eltype(ns)) |> float |> complex
    S = zeros(d, d);
    effects = Matrix{T}[]

    for n in ns
        Xi = (randn(d, n) + im*randn(d, n))/sqrt(2)
        Wi = Xi*Xi'
        push!(effects, Wi)
        S += Wi;
    end

    Srinv = S |> pinv |> sqrt
    effects = [Srinv * W * Srinv for W in effects]

    return POVMMeasurement(effects)
end
function randunitary(n::Integer)
  X = (randn(n, n) + im*randn(n, n))/sqrt(2)
  Q, R = qr(X)
  # sign(x) = x/abs(x)
  basis = [sign(R[i, i]) * Q[:,i] for i in 1:n]
  return hcat(basis...)
end

# %%
# Koon Tong's model

psi(theta) = cos(theta) * kron(ket(1,2), ket(1,2)) + sin(theta) * kron(ket(2,2), ket(2,2))
rho(theta) = proj(psi(theta))
Mtld(mu) = cos(mu) * sigmas[3] + sin(mu) * sigmas[1]
singlet_theta = pi/4
singlet_mus = [0, pi/2]
singlet_nus = [pi/4, -pi/4, 0]
singlet_As = [Mtld(mu) for mu in singlet_mus]
singlet_Bs = [Mtld(nu) for nu in singlet_nus]
singlet_Ms = [POVMMeasurement(A) for A in singlet_As]
singlet_Ns = [POVMMeasurement(B) for B in singlet_Bs]

nc_eta_bl = [0.85, 0.935]
nc_eta_tl = [0.85, 0.938]
nc_eta_tr = [0.95, 0.958]

function meas_corrs(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi])
  rhov = rho(theta)
  rhoA = ptrace(rhov, [2,2], 2)
  rhoB = ptrace(rhov, [2,2], 1)

  # mus for A, nus for B
  Atlds = [E(Mtld(mu), rhoA) |> real for mu in mus]
  Btlds = [E(Mtld(nu), rhoB) |> real for nu in nus]
  ABtlds = [E(kron(Mtld(mu), Mtld(nu)), rhov) |> real for mu in mus, nu in nus]

  return Correlators(Atlds, Btlds, ABtlds)
end

function expt_corrs(nc, eta, tldcorrs)
  Atlds, Btlds, ABtlds = tldcorrs 
  iA, iB = length(Atlds), length(Btlds)
  Eax = [-nc-(1-nc)*((1-eta)-eta*Atlds[x]) for x in 1:iA]
  Eby = [-nc-(1-nc)*((1-eta)-eta*Btlds[y]) for y in 1:iB]
  Eabxy = [nc + (1-nc)*(eta^2 * ABtlds[x,y] - eta*(1-eta)*(Atlds[x] + Btlds[y]) + (1-eta)^2) for x in 1:iA, y in 1:iB]
  return Correlators(Eax, Eby, Eabxy)
end

# necessary and sufficient analytic conditions for membership in Q1
genTLM(p::Behaviour; kwargs...) = genTLM(Correlators(p); kwargs...)
function genTLM(C::Correlators; tol=0)
  Eax, Eby, Eabxy = C
  Acheck = isapprox.(Eax.^2, 1, atol=tol)
  Bcheck = isapprox.(Eby.^2, 1, atol=tol)
  if any(Acheck) || any(Bcheck)
    return true
  end

  iA, iB = length(Eax), length(Eby)
  xsit = combinations(1:iA, 2)
  ysit = combinations(1:iB, 2)

  for (xs, ys) in itprod(xsit, ysit)
    terms = [(Eabxy[x,y] - Eax[x]*Eby[y])/sqrt((1-Eax[x]^2)*(1-Eby[y]^2)) for x in xs, y in ys]
    sumval = sum(asin.(terms))
    tests = [abs(sumval - 2*asin(term)) <= pi for term in terms]
    all(tests) || return false
  end
  return true
end

function expt_grads(nc, eta, Atlds, Btlds, ABtlds)
  iA, iB = length(Atlds), length(Btlds)
  etagrad = [ (1-nc)*((1-2*eta)*(Atlds[x] + Btlds[y]) - 2*eta*ABtlds[x,y] + 2 - 2*eta) for x in 1:iA, y in 1:iB ] 
  ncgrad = [ eta*((1-eta)*(Atlds[x] + Btlds[y]) - eta*ABtlds[x,y] + 2 - eta) for x in 1:iA, y in 1:iB ]
  return ncgrad, etagrad
end

function expt_chsh_ncgrads(ncgrad, etagrad, S)
  Qncgrad = - ncgrad[1,3]
  if !isfinite(Qncgrad)
    Qncgrad = 0
  end
  Sncgrad = ncgrad[1,1] + ncgrad[1,2] + ncgrad[2,1] - ncgrad[2,2]
  if !isfinite(Sncgrad)
    Sncgrad = 0
  end
  HgradS = S/(4*sqrt(S^2-4)) * log2( (2+sqrt(S^2-4)) / (2-sqrt(S^2-4)) )
  if !isfinite(HgradS)
    HgradS = 0
  end
  Hncgrad = Sncgrad * HgradS
  return Qncgrad, Hncgrad
end
