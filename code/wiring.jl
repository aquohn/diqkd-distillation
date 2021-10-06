# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

# %%
# Basic definitions and imports

using Revise
using QuantumInformation, LinearAlgebra

includet("helpers.jl")

# %%
# Wirings

num_wirings(c, o, i) = o^(i * o^c) * prod([i^(i * o^(j-1)) for j in 2:c])
num_wirings_fix(c, o, i, f) = o^((i-f)*i * o^c) * prod([i^((i-f) * o^(j-1)) for j in 2:c])
function wiring_prob(CA, CAj, CB, CBj, pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  ppax, ppby, ppabxy = (zeros(size(p)) for p in [pax, pby, pabxy]) |> collect
  c = Integer((CA |> size |> length) / 2)

  # iterate over every possible combination
  for params in Iterators.product(vcat([[1:n for i in 1:c] for n in [oA, oB, iA, iB]]...)...) @sliceup params as c bs c xs c ys c
    pABXY = 1
    for j in 2:c
      if CAj[j][as[1:j-1]..., xs[1:j-1]...] != xs[j] || CBj[j][bs[1:j-1]..., ys[1:j-1]...] != ys[j]
        pABXY = 0
        break
      else
        pABXY *= pabxy[as[j], bs[j], xs[j], ys[j]]
      end
    end

    xp = CA[as..., xs...]
    yp = CB[bs..., ys...]
    ppabxy[as[1], bs[1], xp, yp] += pABXY
  end

  for params in Iterators.product(vcat([[1:n for i in 1:c] for n in [oA, iA]]...)...)
    @sliceup params as c xs c
    pAX = 1
    for j in 1:c
      if CAj[j][as[1:j-1]..., xs[1:j-1]...] != xs[j]
        pAX = 0
        break
      else
        pAX *= pax[as[j], xs[j]]
      end
    end

    xp = CA[as..., xs...]
    ppax[as[1], xp] += pAX
  end

  for params in Iterators.product(vcat([[1:n for i in 1:c] for n in [oB, iB]]...)...)
    @sliceup params bs c ys c
    pBY = 1
    for j in 1:c
      if CBj[j][bs[1:j-1]..., ys[1:j-1]...] != ys[j]
        pBY = 0
        break
      else
        pBY *= pby[bs[j], ys[j]]
      end
    end

    yp = CB[bs..., ys...]
    ppax[bs[1], yp] += pBY
  end

  return ppax, ppby, ppabxy
end

# %%
# Koon Tong's model

psi(theta) = cos(theta) * kron(ket(1,2), ket(1,2)) + sin(theta) * kron(ket(2,2), ket(2,2))
rho(theta) = proj(psi(theta))
Mtld(mu) = cos(mu) * sigmas[3] + sin(mu) * sigmas[1]

# let outcome 1 be associated with eigenvalue -1 or logical 0
function probs_from_corrs(Eax, Eby, Eabxy)
  invcorr = 0.25 * [-1.0 -1.0 1.0 1.0; -1.0 1.0 -1.0 1.0; 1.0 -1.0 -1.0 1.0; 1.0 1.0 1.0 1.0]

  oA, oB = 2, 2
  iA, iB = [length(Eax), length(Eby)]
  pax = Array{Float64}(undef, oA, iA)
  pby = Array{Float64}(undef, oB, iB)
  pabxy = Array{Float64}(undef, oA, oB, iA, iB)

  for x in 1:iA, y in 1:iB
    pabxy[:, :, x, y] = invcorr * [Eax[x]; Eby[y]; Eabxy[x,y]; 1]
  end
  for a in 1:oA, x in 1:iA
    pax[a, x] = sum(pabxy[a, :, x, 1])
  end
  for b in 1:oB, y in 1:iB
    pby[b, y] = sum(pabxy[:, b, 1, y])
  end

  return pax, pby, pabxy
end

function margps_from_jointp(pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  if isnothing(pax)
    pax = [sum(pabxy[a,:,x,1]) for a in 1:oA, x in 1:iA]
  end
  if isnothing(pby)
    pby = [sum(pabxy[:,b,1,y]) for b in 1:oB, y in 1:iB]
  end
  return pax, pby, pabxy
end

function corrs_from_probs(pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  pax, pby, pabxy = margps_from_jointp(pax, pby, pabxy)
  Eax = [pax[2,x] - pax[1,x] for x in 1:iA]
  Eby = [pby[2,y] - pby[1,y] for y in 1:iB]
  Eabxy = [sum([pabxy[a,b,x,y] * ((a == b) ? 1 : -1) for a in 1:oA, b in 1:iA]) for x in 1:iA, y in 1:iB]

  return Eax, Eby, Eabxy
end

function meas_corrs(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi])
  rhov = rho(theta)
  rhoA = ptrace(rhov, [2,2], 2)
  rhoB = ptrace(rhov, [2,2], 1)

  # mus for A, nus for B
  Atlds = Float64[E(Mtld(mu), rhoA) for mu in mus]
  Btlds = Float64[E(Mtld(nu), rhoB) for nu in nus]
  ABtlds = Float64[E(kron(Mtld(mu), Mtld(nu)), rhov) for mu in mus, nu in nus]

  return Atlds, Btlds, ABtlds
end

function expt_corrs(nc, eta, Atlds, Btlds, ABtlds)
  iA, iB = length(Atlds), length(Btlds)

  Eax = Float64[-nc-(1-nc)*((1-eta)-eta*Atlds[x]) for x in 1:iA]
  Eby = Float64[-nc-(1-nc)*((1-eta)-eta*Btlds[y]) for y in 1:iB]
  Eabxy = Float64[nc + (1-nc)*(eta^2 * ABtlds[x,y] - eta*(1-eta)*(Atlds[x] + Btlds[y]) + (1-eta)^2) for x in 1:iA, y in 1:iB]
  return Eax, Eby, Eabxy
end

function expt_grads(nc, eta, Atlds, Btlds, ABtlds)
  etagrad = Float64[ (1-nc)*((1-2*eta)*(Atlds[x] + Btlds[y]) - 2*eta*ABtlds[x,y] + 2 - 2*eta) for x in 1:iA, y in 1:iB ] 
  ncgrad = Float64[ eta*((1-eta)*(Atlds[x] + Btlds[y]) - eta*ABtlds[x,y] + 2 - eta) for x in 1:iA, y in 1:iB ]
  return etagrad, ncgrad
end

function expt_chsh_ncgrads(etagrad, ncgrad)
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


function and_corrs(N, Eax, Eby, Eabxy)
  iA, iB = length(Eax), length(Eby)

  EaxN = [1 - 2*((1-Eax[x])/2)^N for x in 1:iA]
  EbyN = [1 - 2*((1-Eby[y])/2)^N for y in 1:iB]
  EabxyN = [1 - ((1-Eax[x])^N + (1-Eby[y])^N)/(2^(N-1)) + 4^(1-N) * (1-Eax[x]-Eby[y]+Eabxy[x,y])^N for x in 1:iA, y in 1:iB]

  return EaxN, EbyN, EabxyN
end

# %%
# Wiring exhaustive search

# TODO incorporate don't cares

function wiring_iters(iA, oA, iB, oB, c)
  # I love this language lmao
  CAshapes = tuple((iA for j in 1:c)..., (oA for j in 1:c)...)
  CBshapes = tuple((iB for j in 1:c)..., (oB for j in 1:c)...)
  CAjshapes = [tuple((iA for k in 1:j-1)..., (oA for k in 1:j-1)...) for j in 1:c]
  CBjshapes = [tuple((iB for k in 1:j-1)..., (oB for k in 1:j-1)...) for j in 1:c]

  CAiters = fill(1:oA, CAshapes...)
  CBiters = fill(1:oB, CBshapes...)
  CAjiters = [fill(1:iA, CAjshapes[j]...) for j in 1:c]
  CBjiters = [fill(1:iB, CBjshapes[j]...) for j in 1:c]

  # first function is trivial
  CAjiters[1][] = 0:0
  CBjiters[1][] = 0:0

  return CAiters, CBiters, CAjiters, CBjiters
end

function diqkd_wiring_iters(c = 2)
  iA, oA, iB, oB = 2, 2, 3, 2
  CAiters, CBiters, CAjiters, CBjiters = wiring_iters(iA, oA, iB, oB, c)

  # Fix keygen settings x = 1 and y = 3 to be AND-gated
  for params in Iterators.product([1:oA for i in 1:c]..., [1:iA for i in 2:c]...)
    @sliceup params as c xs c-1
    CAiters[1, xs..., as...] = all(as .== 2) ? (2:2) : (1:1)
  end
  for params in Iterators.product([1:oB for i in 1:c]..., [1:iB for i in 2:c]...)
    @sliceup params bs c ys c-1
    CBiters[3, ys..., bs...] = all(bs .== 2) ? (2:2) : (1:1)
  end

  # Fix keygen settings to simply broadcast initial input
  for j in 2:c # j == 1 is a 0-dim array
    for params in Iterators.product([1:oA for i in 1:j-1]..., [1:iA for i in 2:j-1]...)
      @sliceup params as j-1 xs j-2
      CAjiters[j][1, xs..., as...] = 1:1
    end
    for params in Iterators.product([1:oB for i in 1:j-1]..., [1:iB for i in 2:j-1]...)
      @sliceup params bs j-1 ys j-2
      CBjiters[j][3, ys..., bs...] = 3:3
    end
  end

  return CAiters, CBiters, CAjiters, CBjiters
end

function diqkd_wiring_eval(Eabxy, Eax, Eby, HAE, HAB)
  CAiters, CBiters, CAjiters, CBjiters = diqkd_wiring_iters()
  iterarrs = [CAiters, CBiters, CAjiters..., CBjiters...]
  shapes = [size(iterarr) for iterarr in iterarrs]
  lengths = [prod(shape) for shape in shapes]

  for data in Iterators.product(vcat([vec(iterarr) for iterarr in iterarrs]...)...)
    info = sliceup(data, lengths...)
    arrs = [reshape(info[j], shapes[j]) for j in eachindex(info)]
    js = Integer((length(arrs) - 2)/2)
    CA, CB, CAj, CBj = sliceup(arrs, 1, 1, js, js)

    HAEval = HAE(Eax, Eby, Eabxy)
    HABval = HAB(Eax, Eby, Eabxy)
    r = HAEval - HABval

    pax, pby, pabxy = probs_from_corrs(Eax, Eby, Eabxy)
    ppax, ppby, ppabxy = wiring_prob(CA, CB, CAj, CBj, pax, pby, pabxy)
    Epax, Epby, Epabxy = corrs_from_probs(ppax, ppby, ppabxy)

    HAEvalp = HAE(Epax, Epby, Epabxy)
    HABvalp = HAB(Epax, Epby, Epabxy)
    rp = HAEvalp - HABvalp

    # TODO implement figure of merit
  end
end
