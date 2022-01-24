using Revise
using LinearAlgebra, QuantumInformation

includet("helpers.jl")
includet("maxcorr.jl")

const vLl = 999 * 689 * 10^-6 * cos(pi/50)^4
const vNL = 0.6964
vcrit(vL) = (vL+1)/(3-vL)

kd(i,j) = (i == j) ? 1 : 0
E(M, rho) = tr(M * rho)
const sigmas = [[kd(j, 3) kd(j,1)-im*kd(j,2); kd(j,1)+im*kd(j,2) -kd(j,3)] for j in 1:3]
# Generalised Gell-Mann matrices
function su_generators(n::Integer)
  Emats = [sparse([j], [k], [1], n, n) for j in 1:n, k in 1:n]
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
  diags = [sqrt(2/(l*(l+1))) * sum([Emats[j,j] - l*Emats[l+1,l+1] for j in 1:l])
           for l in 1:n-1]
  return vcat(syms, antisyms, diags)
end

# Impt states
const singlet_corrs = Correlators([0, 0],
                                 [0, 0, 0],
                                 [1/sqrt(2) 1/sqrt(2) 1
                                  1/sqrt(2) -1/sqrt(2) 0])

const bound_corrs = Correlators([0, 0],
                         [0, 0, 0],
                         [1//2 1//2 1//2
                          1//2 -1//2 1//2])

const PR_corrs = Correlators([0, 0],
                      [0, 0, 0],
                      [1 1 1;
                       1 -1 1])

const mix_corrs = Correlators([0, 0],
                       [0, 0, 0],
                       [0 0 0
                        0 0 0])

const LD_corrs = Correlators([1, 1],
                      [1, 1, 1],
                      [1 1 1
                       1 1 1])

werner_corrs(v) = convexsum([v, 1-v], [singlet_corrs, mix_corrs])
const v_L = 0.6829; const v_NL = 0.6964; const v_crit = 0.7263

# %%
# Koon Tong's model

psi(theta) = cos(theta) * kron(ket(1,2), ket(1,2)) + sin(theta) * kron(ket(2,2), ket(2,2))
rho(theta) = proj(psi(theta))
Mtld(mu) = cos(mu) * sigmas[3] + sin(mu) * sigmas[1]
mus = [0, pi/2]
nus = [pi/2, 0, -pi/2]

function meas_corrs(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi])
  rhov = rho(theta)
  rhoA = ptrace(rhov, [2,2], 2)
  rhoB = ptrace(rhov, [2,2], 1)

  # mus for A, nus for B
  Atlds = Float64[E(Mtld(mu), rhoA) for mu in mus]
  Btlds = Float64[E(Mtld(nu), rhoB) for nu in nus]
  ABtlds = Float64[E(kron(Mtld(mu), Mtld(nu)), rhov) for mu in mus, nu in nus]

  return Correlators(Atlds, Btlds, ABtlds)
end

function expt_corrs(nc, eta, tldcorrs)
  Atlds, Btlds, ABtlds = tldcorrs 
  iA, iB = length(Atlds), length(Btlds)
  Eax = Float64[-nc-(1-nc)*((1-eta)-eta*Atlds[x]) for x in 1:iA]
  Eby = Float64[-nc-(1-nc)*((1-eta)-eta*Btlds[y]) for y in 1:iB]
  Eabxy = Float64[nc + (1-nc)*(eta^2 * ABtlds[x,y] - eta*(1-eta)*(Atlds[x] + Btlds[y]) + (1-eta)^2) for x in 1:iA, y in 1:iB]
  return Correlators(Eax, Eby, Eabxy)
end

function expt_grads(nc, eta, Atlds, Btlds, ABtlds)
  iA, iB = length(Atlds), length(Btlds)
  etagrad = Float64[ (1-nc)*((1-2*eta)*(Atlds[x] + Btlds[y]) - 2*eta*ABtlds[x,y] + 2 - 2*eta) for x in 1:iA, y in 1:iB ] 
  ncgrad = Float64[ eta*((1-eta)*(Atlds[x] + Btlds[y]) - eta*ABtlds[x,y] + 2 - eta) for x in 1:iA, y in 1:iB ]
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


