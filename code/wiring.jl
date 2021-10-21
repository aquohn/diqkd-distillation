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

struct BoxSequence{T <: Integer}
  boxtype::Symbol
  as::Union{Vector{T}, Nothing}
  bs::Union{Vector{T}, Nothing}
  xs::Union{Vector{T}, Nothing}
  ys::Union{Vector{T}, Nothing}
  function BoxSequence(bt, os::Vector{T}, is::Vector{T}) where T <: Integer
    if (bt == :A)
      new{T}(:A, os, nothing, is, nothing)
    elseif(bt == :B)
      new{T}(:B, nothing, os, nothing, is)
    else
      throw(ArgumentError("Box type must be :A or :B!"))
    end
  end
end

struct Setting{T}
  oA::T
  oB::T
  iA::T
  iB::T
end
Base.iterate(s::Setting) = s.oA, [s.iB, s.iA, s.oB]
Base.iterate(s::Setting, state) = isempty(state) ? nothing : (pop!(state), state)

struct Correlators
  Eax
  Eby
  Eabxy
end
Base.iterate(C::Correlators) = C.Eax, [C.Eabxy, C.Eby]
Base.iterate(C::Correlators, state) = isempty(state) ? nothing : (pop!(state), state)

struct Behaviour
  pax
  pby
  pabxy
end
Base.iterate(P::Behaviour) = P.pax, [P.pabxy, P.pby]
Base.iterate(P::Behaviour, state) = isempty(state) ? nothing : (pop!(state), state)

struct Wiring
  CA
  CAj
  CB
  CBj
end
Base.iterate(W::Wiring) = W.CA, [W.CBj, W.CB, W.CAj]
Base.iterate(W::Wiring, state) = isempty(state) ? nothing : (pop!(state), state)

struct EntropyData
  HAE
  HAB
  HAEp
  HABp
end

struct WiringData
  wiring::Wiring
  r
  rp
  Hdata::EntropyData
end

# TODO rewrite functions to take structs as args

num_wirings(c, o, i) = o^(i * o^c) * prod([i^(i * o^(j-1)) for j in 2:c])
num_wirings_fix(c, o, i, f) = o^((i-f)*i * o^c) * prod([i^((i-f) * o^(j-1)) for j in 2:c])
function wiring_prob(CA, CAj, CB, CBj, pax::Array{T,2}, pby::Array{T,2}, pabxy::Array{T,4}) where T <: Real
  oA, oB, iA, iB = size(pabxy)
  ppax, ppby, ppabxy = (zeros(size(p)) for p in [pax, pby, pabxy]) |> collect
  c = Integer((CA |> size |> length) / 2)

  aseqs = BoxSequence[BoxSequence(:A, [a], [x]) for (a,x) in Iterators.product(1:oA, 1:iA)] |> vec
  while !isempty(aseqs) # depth-first search
    aseq = pop!(aseqs)
    j = length(aseq.as)
    if j == c # all boxes accounted for; start iterating over Bob's boxes
      bseqs = BoxSequence[BoxSequence(:B, [b], [y]) for (b,y) in Iterators.product(1:oB, 1:iB)] |> vec
      while !isempty(bseqs)
        bseq = pop!(bseqs)
        k = length(bseq.bs)
        if k == c
          p = prod([pabxy[aseq.as[i], bseq.bs[i], aseq.xs[i], bseq.ys[i]] for i in 1:c])
          ap = CA[aseq.xs..., aseq.as...]
          bp = CB[bseq.ys..., bseq.bs...]
          ppabxy[ap, bp, aseq.xs[1], bseq.ys[1]] += p
        else
          y = CBj[k+1][bseq.ys..., bseq.bs...]
          for b in 1:oB
            push!(bseqs, BoxSequence(:B, [bseq.bs..., b], [bseq.ys..., y]))
          end
        end
      end
    else # add all possibilities for next box
      x = CAj[j+1][aseq.xs..., aseq.as...]
      for a in 1:oA
        push!(aseqs, BoxSequence(:A, [aseq.as..., a], [aseq.xs..., x]))
      end
    end
  end

  ppax, ppby, ppabxy = margps_from_jointp(nothing, nothing, ppabxy)
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


function and_corrs(N, Eax, Eby, Eabxy)
  iA, iB = length(Eax), length(Eby)

  EaxN = [1 - 2*((1-Eax[x])/2)^N for x in 1:iA]
  EbyN = [1 - 2*((1-Eby[y])/2)^N for y in 1:iB]
  EabxyN = [1 - ((1-Eax[x])^N + (1-Eby[y])^N)/(2^(N-1)) + 4^(1-N) * (1-Eax[x]-Eby[y]+Eabxy[x,y])^N for x in 1:iA, y in 1:iB]

  return EaxN, EbyN, EabxyN
end

# %%
# Wiring exhaustive search

function wiring_iters(iA::T, oA::T, iB::T, oB::T, c::T) where T <: Integer
  CAiters = [1:oA for i in 1:(iA * oA^c)]
  CBiters = [1:oB for i in 1:(iB * oB^c)]
  CAjiters = vcat([fill(T, 0:0, 0)], [[1:iA for i in 1:(iA * oA^(j-1))] for j in 2:c])
  CBjiters = vcat([fill(T, 0:0, 0)], [[1:iB for i in 1:(iB * oB^(j-1))] for j in 2:c])

  return CAiters, CBiters, CAjiters, CBjiters
end

function wiring_policy(CA, CAj, CB, CBj, CAvec, CBvec, CAjvec, CBjvec, iA, oA, iB, oB, c)
  aseqs = BoxSequence[BoxSequence(:A, [a], [x]) for (a,x) in Iterators.product(1:oA, 1:iA)] |> vec
  while !isempty(aseqs) # depth-first search
    aseq = pop!(aseqs)
    j = length(aseq.as)
    if j == c # all boxes accounted for; start iterating over Bob's boxes
      ap = pop!(CAvec)
      CA[aseq.xs..., aseq.as...] = ap
    else # add all possibilities for next box
      x = pop!(CAjvec[j+1])
      CAj[j+1][aseq.xs..., aseq.as...] = x
      for a in 1:oA
        push!(aseqs, BoxSequence(:A, [aseq.as..., a], [aseq.xs..., x]))
      end
    end
  end

  bseqs = BoxSequence[BoxSequence(:B, [b], [y]) for (b,y) in Iterators.product(1:oB, 1:iB)] |> vec
  while !isempty(bseqs)
    bseq = pop!(bseqs)
    k = length(bseq.bs)
    if k == c
      bp = pop!(CBvec)
      CB[bseq.ys..., bseq.bs...] = bp
    else
      y = pop!(CBjvec[k+1])
      CBj[k+1][bseq.ys..., bseq.bs...] = y
      for b in 1:oB
        push!(bseqs, BoxSequence(:B, [bseq.bs..., b], [bseq.ys..., y]))
      end
    end
  end
end

function generate_wirings(CAvec::Vector{T}, CBvec::Vector{T},
    CAjvec::Vector{Vector{T}}, CBjvec::Vector{Vector{T}}, iA, oA, iB, oB, c, policy = wiring_policy) where T <: Integer
  CAvec, CBvec, CAjvec, CBjvec = (deepcopy(v) for v in (CAvec, CBvec, CAjvec, CBjvec))
  CA = zeros(T, [iA for i in 1:c]..., [oA for i in 1:c]...)
  CB = zeros(T, [iB for i in 1:c]..., [oB for i in 1:c]...)
  CAj = [zeros(T, [iA for i in 1:(j-1)]..., [oA for i in 1:(j-1)]...) for j in 1:c]
  CBj = [zeros(T, [iB for i in 1:(j-1)]..., [oB for i in 1:(j-1)]...) for j in 1:c]

  policy(CA, CAj, CB, CBj, CAvec, CBvec, CAjvec, CBjvec, iA, oA, iB, oB, c)

  return CA, CAj, CB, CBj
end

# fixing the wirings for one i removes o^c degrees of freedom from C, and
# o^(j-1) degrees of freedom from Cj[j]
function diqkd_wiring_and_policy(CA, CAj, CB, CBj, CAvec, CBvec, CAjvec, CBjvec, iA, oA, iB, oB, c)
  aseqs = BoxSequence[BoxSequence(:A, [a], [x]) for (a,x) in Iterators.product(1:oA, 1:iA)] |> vec
  while !isempty(aseqs) # depth-first search
    aseq = pop!(aseqs)
    j = length(aseq.as)
    if j == c # all boxes accounted for
      if aseq.xs[1] == 1
        ap = all(as .== 2) ? 2 : 1 # AND gate, 1 = F, 2 = T
      else
        ap = pop!(CAvec)
      end
      CA[aseq.xs..., aseq.as...] = ap
    else # add all possibilities for next box
      if aseq.xs[1] == 1
        x = xs[1]
      else
        x = pop!(CAjvec[j+1])
      end
      for a in 1:oA
        push!(aseqs, BoxSequence(:A, [aseq.as..., a], [aseq.xs..., x]))
      end
    end
  end

  bseqs = BoxSequence[BoxSequence(:B, [b], [y]) for (b,y) in Iterators.product(1:oB, 1:iB)] |> vec
  while !isempty(bseqs)
    bseq = pop!(bseqs)
    k = length(bseq.bs)
    if k == c
      if bseq.ys[1] == 3
        bp = all(bs .== 2) ? 2 : 1
      else
        bp = pop!(CBvec)
      end
      CB[bseq.ys..., bseq.bs...] = bp
    else
      if bseq.ys[1] == 3
        y = ys[1]
      else
        y = pop!(CBjvec[j+1])
      end
      for b in 1:oB
        push!(bseqs, BoxSequence(:B, [bseq.bs..., b], [bseq.ys..., y]))
      end
    end
  end
end

function diqkd_wiring_and_iters(c = 2)
  iA, oA, iB, oB = 2, 2, 3, 2
  CAiters, CBiters, CAjiters, CBjiters = wiring_iters(iA, oA, iB, oB, c)
  CAiters = CAiters[1:end-oA^c]
  CBiters = CBiters[1:end-oB^c]
  CAjiters = [CAjiters[j][1:end-iA^(j-1)] for j in 1:c]
  CBjiters = [CBjiters[j][1:end-iB^(j-1)] for j in 1:c]

  return CAiters, CBiters, CAjiters, CBjiters
end

function diqkd_wiring_eval(pax, pby, pabxy, HAE, HAB; c = 2, iterf = nothing, policy = wiring_policy)
  oA, oB, iA, iB = size(pabxy)
  if isnothing(iterf)
    iterf = () -> wiring_iters(iA, oA, iB, oB, c)
  end
  CAiters, CBiters, CAjiters, CBjiters = iterf()
  iterarrs = [CAiters, CBiters, CAjiters..., CBjiters...]
  shapes = [size(iterarr) for iterarr in iterarrs]
  lengths = [prod(shape) for shape in shapes]
  recs = WiringData[]

  for data in Iterators.product(vcat([vec(iterarr) for iterarr in iterarrs]...)...)
    sliced = sliceup(data, lengths...)
    CAvec, CBvec = [sliced[1:2]...]
    l = Integer(length(sliced[3:end]) / 2)
    CAjvec, CBjvec = sliceup(sliced[3:end], l, l)

    CA, CAj, CB, CBj = generate_wirings(CAvec, CBvec, CAjvec, CBjvec, iA, oA, iB, oB, c, policy)

    HAEval, HAEdata = HAE(pax, pby, pabxy)
    HABval, HABdata = HAB(pax, pby, pabxy)
    r = HAEval - HABval

    ppax, ppby, ppabxy = wiring_prob(CA, CAj, CB, CBj, pax, pby, pabxy)

    HAEvalp, HAEdatap = HAE(ppax, ppby, ppabxy)
    HABvalp, HABdatap = HAB(ppax, ppby, ppabxy)
    rp = HAEvalp - HABvalp

    if rp > r
      push!(recs, WiringData(CA, CAj, CB, CBj, r, rp, EntropyData(HAEdata, HABdata, HAEdatap, HABdatap)))
    end
  end

  return recs
end
