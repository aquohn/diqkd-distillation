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
using Combinatorics
using QuantumInformation, LinearAlgebra

includet("helpers.jl")
includet("nonlocality.jl")

# %%
# Counting and generating wirings
function incr_idx_vec!(minvec::AbstractVector{T}, maxvec::AbstractVector{T}, vec::AbstractVector{T}) where {T <: Integer}
  if !(length(minvec) == length(maxvec) == length(vec))
    throw(ArgumentError("Vectors must be of the same length! Got minvec of $(length(minvec)), maxvec of $(length(maxvec)), and vec of $(length(vec))."))
  end
  for idx in Iterators.reverse(eachindex(vec))
    if vec[idx] < maxvec[idx]
      vec[idx] += 1
      break
    end
    vec[idx] = minvec[idx]
  end
  # manually compare vec[1] before and after to detect overflow
end

# Approach 1: Enumerate LDs and prune with no-retrocausation
num_lds(c, sett::Setting) = num_lds(c, sett.oA, sett.iA) * num_lds(c, sett.oB, sett.iB)
num_lds(c, o, i) = (o * i^c)^(i^c * o)
struct LDWiringIter{T <: Integer}
  c::T
  o::T
  i::T
  _ts
  _ss
  _dtsit
  function LDWiringIter(c::Integer, o::Integer, i::Integer)
    T = promote_type(typeof(c), typeof(o), typeof(i))
    ts = (t for t in itprod(1:o, repeat([1:i], c)...))  # diff targets
    ss = (s for s in itprod(repeat([1:o], c)..., 1:i))  # diff sources
    lenss = length(ss)  # number of diff sources
    dtsit = itprod(repeat([ts], lenss)...)  # iterator over possible deterministic targets
    new{T}(c, o, i, ts, ss, dtsit)
  end
end
function ldwiring_check_prob(ldwit::LDWiringIter{Ti}, chi::Array{Tp}) where {Tp <: Real, Ti <: Integer}
  valid = true
  c, o, i = ldwit.c, ldwit.o, ldwit.i
  for j in 1:c
    rmdims = (1+j+1):(1+c)
    chij = dropdims(sum(chi; dims=rmdims); dims=Tuple(rmdims))
    for t in itprod(1:o, repeat([1:i], j)...)
      for scomm in itprod(repeat([1:o], j-1)...)
        for x in 1:i
          probs = (chij[t..., scomm..., svar..., x]
                   for svar in itprod(repeat([1:o], c-(j-1))...))
          if !(all(p -> p == first(probs), probs))
            valid = false
            @goto checkdone
          end
        end
      end
    end
  end
  @label checkdone
  return valid
end
function _iterate(ldwit::LDWiringIter{T}, dtsitcurr) where T
  c, o, i, ts, ss, dtsit  = ldwit.c, ldwit.o, ldwit.i, ldwit._ts, ldwit._ss, ldwit._dtsit
  while !isnothing(dtsitcurr)
    dts, state = dtsitcurr
    chi = zeros(T, o, repeat([i], c)..., i, repeat([o], c)...)
    for (dt, s) in zip(dts, ss)
      chi[dt..., s...] = 1
    end
    if ldwiring_check_prob(ldwit, chi)
      return chi, state
    end
    dtsitcurr = iterate(dtsit, state)
  end
end
Base.iterate(ldwit::LDWiringIter) = _iterate(ldwit, iterate(ldwit._dtsit))
Base.iterate(ldwit::LDWiringIter, state) = _iterate(ldwit, iterate(ldwit._dtsit, state))

# Approach 2: Enumerate possible wiring maps
num_wiring_maps(c, sett::Setting) = num_wiring_maps(c, sett.oA, sett.iA) * num_wiring_maps(c, sett.oB, sett.iB)
num_wiring_maps(c, o, i) = o^(i * o^c) * prod([i^(i * o^(j-1)) for j in 1:c])
num_wiring_maps_fix(c, sett::Setting, fA, fB) = num_wiring_maps_fix(c, sett.oA, sett.iA, fA) * num_wiring_maps_fix(c, sett.oB, sett.iB, fB)
function num_wiring_maps_fix(c, o, i, f)
  if f > i
    throw(ArgumentError("$f inputs to fix, but there are only $i inputs!"))
  end
  o^((i-f) * o^c) * prod([i^((i-f) * o^(j-1)) for j in 1:c])
end
struct CondWiringMapIter{T <: Integer}
  c::T
  o::T
  i::T
  j::T
  _idxit
  _valit
  function CondWiringMapIter(c::Integer, o::Integer, i::Integer, j::Integer)
    if j > c + 1
      throw(ArgumentError("There are only $(c+1) wiring maps for c = $c!"))
    end
    T = promote_type(typeof(c), typeof(o), typeof(i), typeof(j))
    # remove ugliness when https://github.com/JuliaLang/julia/issues/43921 is
    # closed
    if j == 1
      idxit = [()]
    else
      idxit = itprod(repeat([1:o], j-1)...)  # all sources for the wiring maps
    end
    valit = itprod(repeat([(j <= c) ? (1:i) : (1:o)], length(idxit))...)  # all possible wiring map outputs
    new{T}(c, o, i, j, idxit, valit)
  end
end
_iterate(cwmit::CondWiringMapIter, mapitcurr::Nothing) = nothing
function _iterate(cwmit::CondWiringMapIter{T}, valitcurr) where T
  vals, state = valitcurr
  c, o, i, j = cwmit.c, cwmit.o, cwmit.i, cwmit.j
  W = Array{T}(undef, repeat([o], j-1)...)
  for (val, idx) in zip(vals, cwmit._idxit)
    W[idx...] = val
  end
  return W, state
end
Base.length(cwmit::CondWiringMapIter) = length(cwmit._valit)
Base.iterate(cwmit::CondWiringMapIter) = _iterate(cwmit, iterate(cwmit._valit))
Base.iterate(cwmit::CondWiringMapIter, state) = _iterate(cwmit, iterate(cwmit._valit, state))

struct WiringMapIter{T <: Integer}
  c::T
  o::T
  i::T
  fix::Vector{Tuple{Array{T}, T, T}}
  _prodwmit
  function WiringMapIter(c::Integer, o::Integer, i::Integer, fix::Vector)
    T = promote_type(typeof(c), typeof(o), typeof(i))
    its = [
           repeat([CondWiringMapIter(c, o, i, j)], i) for j in 1:c+1
    ]
    for (cWmap, x, j) in fix
      # TODO check for validity
      its[j][x] = [cWmap]
    end
    prodwmit = itprod([itprod(itj...) for itj in its]...)
    new{T}(c, o, i, fix, prodwmit)
  end
  WiringMapIter(c, o, i) = WiringMapIter(c, o, i, [])
end
Base.length(wmit::WiringMapIter) = length(wmit._prodwmit)
Base.iterate(wmit::WiringMapIter) = iterate(wmit._prodwmit)
Base.iterate(wmit::WiringMapIter, state) = iterate(wmit._prodwmit, state)
# NOTE: returns tuple of tuple of arrays
# for c=2, o=2, i=3 we need 200GB of RAM; I think we should write this out and
# solve slowly using lrs

# Approach 3: Decompose into application and permutation maps
struct MargWiring{Ti <: Integer, Tp <: Real}
  c::Ti
  o::Ti
  i::Ti
  W::SparseMatrixCSC{Tp}
end
MargWiring(c::Integer, o::Integer, i::Integer, Wmap) = MargWiring(Float64, c, o, i, Wmap)
function MargWiring(::Type{Tp}, c::Integer, o::Integer, i::Integer, Wmap) where Tp <: Real
  Ti = promote_type(typeof(c), typeof(o), typeof(i))
  W = vcat(repeat([I((o*i)^c)], i)...)
  for j in 1:c
    sit = itprod(repeat([1:o], j-1)...)
    condmaps = []
    for x in 1:i
      Wmapcurr = Wmap[j][x]
      condWtld = cat((sparsevec([Wmapcurr[s...]], Tp[1], i)' for s in sit)...;
                    dims=(1,2))
      push!(condmaps, condWtld)
    end
    Wj = kron(cat(condmaps...; dims=(1,2)), I(i))
    W = kron(Wj, I((o*i)^(c-j))) * W
  end
  sit = itprod(repeat([1:o], c)...)
  scount = length(sit)
  Wfinal = cat((sparse([(Wmap[c+1][x][s...] for s in sit)...], 1:scount, ones(Tp, scount), o, scount) for x in 1:i)...; dims=(1,2))
  return MargWiring{Ti, Tp}(c, o, i, Wfinal * W)
end

struct Wiring{Ti <: Integer, Tp <: Real}
  n::Ti
  c::Ti
  os::Vector{Ti}
  is::Vector{Ti}
  W::SparseMatrixCSC{Tp}
end
function Wiring(margWs::AbstractVector{MargWiring{Ti, Tp}}) where {Ti, Tp}
  cs = [margW.c for margW in margWs]
  if !(all(c -> c == first(cs), cs))
    throw(ArgumentError("Marginal wirings take different numbers of boxes!"))
  end
  os = [margW.o for margW in margWs]
  is = [margW.i for margW in margWs]
  W = kron((margW.W for margW in margWs)...)
  return Wiring(length(margWs), first(cs), os, is, W)
end

# TODO implement AbstractVector iface, maybe value as type parameter?
struct BehaviourVec{Ti <: Integer, Tp <: Real}
  n::Ti
  os::Vector{Ti}
  is::Vector{Ti}
  p::Vector{Tp}
end
function BehaviourVec(parr::AbstractArray{Tp}) where {Tp <: Real}
    dims = size(parr)
    Ti = eltype(dims)
    n = Ti(length(dims) // 2)
    dimvec = Ti[dims...]
    os, is = dimvec[1:n], dimvec[n+1:2*n]
    argit = itprod(vcat(([1:os[j], 1:is[j]] for j in n:-1:1)...)...)
    p = Vector{Tp}(undef, length(argit))
    for (pidx, argval) in zip(eachindex(p), argit)
      arg = Vector{Ti}(undef, 2*n)
      for j in 1:n
        arg[j] = argval[2*(n-j)+1]
        arg[j+n] = argval[2*(n-j)+2]
      end
      p[pidx] = parr[arg...]
    end
    return BehaviourVec{Ti, Tp}(n, os, is, p)
end
function BehaviourVec(behav::Behaviour)
  oA, oB, iA, iB = size(behav.pabxy)
  BehaviourVec([oA, oB], [iA, iB], behav.pabxy)
end
function BehaviourVec(mat::AbstractMatrix)
  o, i = size(mat)
  BehaviourVec([o], [i], mat)
end
function Array(bvec::BehaviourVec{Ti, Tp}) where {Ti, Tp}
  os, is, p = bvec.os, bvec.is, bvec.p
  n = length(os)  # TODO check value
  parr = Array{Tp}(undef, os..., is...)
  argit = itprod(vcat(([1:os[j], 1:is[j]] for j in n:-1:1)...)...)
  for (pval, argval) in zip(p, argit)
    arg = Vector{Ti}(undef, 2*n)
    for j in 1:n
      arg[j] = argval[2*(n-j)+1]
      arg[j+n] = argval[2*(n-j)+2]
    end
    parr[arg...] = pval
  end
  return parr
end

Base.:*(margwir::MargWiring, b::BehaviourVec) = Wiring([margwir]) * b
function Base.:*(wiring::Wiring{Tiw, Tpw}, b::BehaviourVec{Tib, Tpb}) where {Tiw, Tpw, Tib, Tpb}
  c, n, os, is = wiring.c, b.n, b.os, b.is
  @assert wiring.n == n
  @assert all(wiring.os .== os)
  @assert all(wiring.is .== is)
  Ti = promote_type(Tiw, Tib)
  Tp = promote_type(Tpw, Tpb)
  kronp = kron(repeat([b.p], c)...)
  permvec = Vector{Ti}(undef, n*c)
  for j in 1:n
    permvec[((j-1)*n+1):j*n] = j:n:c*n
  end
  invpermvec = invperm(permvec)
  dims = vcat((repeat([o*i], c) for (o, i) in zip(os, is))...)
  permp = permutesystems(kronp, dims, permvec)
  permpp = wiring.W * permp
  return BehaviourVec{Ti, Tp}(n, os, is, permpp)
end
Base.:(==)(bv1::BehaviourVec, bv2::BehaviourVec) = all(iszero.(bv1.p - bv2.p))

# n stars separated by k bars
struct StarsBarsNN{T <: Integer}
  n::T
  k::T
  _combit
  function StarsBarsNN(n::Integer, k::Integer)
    T = promote_type(typeof(n), typeof(k))
    combit = combinations(1:T(n+k-1), T(k-1))
    new{T}(n, k, combit)
  end
end
Base.length(snb::StarsBarsNN) = binomial(snb.n + snb.k - 1, snb.k - 1)
Base.iterate(snb::StarsBarsNN) = _iterate(snb, iterate(snb._combit))
Base.iterate(snb::StarsBarsNN, state) = _iterate(snb, iterate(snb._combit, state))
_iterate(snb::StarsBarsNN, combval::Nothing) = nothing
function _iterate(snb::StarsBarsNN{T}, combval) where T
  n, k = snb.n, snb.k
  comb, state = combval
  counts = Vector{T}(undef, k)
  counts[1] = comb[1] - 1
  for idx in 2:k-1
    counts[idx] = comb[idx] - comb[idx-1] - 1
  end
  counts[k] = n+k - comb[k-1] - 1
  return counts, state
end

function num_wiring_matrices(c, o, i)
  mat_counts = BigInt[]
  for j in 1:c
    tcount = o^(j-1)
    tcountf = factorial(big(tcount))
    snb = StarsBarsNN(tcount, i)
    jmat_count = 0
    for counts in snb
      factor = prod(factorial.(big.(counts)))
      jmat_count += BigInt(tcountf // factor)
    end
    push!(mat_counts, jmat_count)
  end
  ftcount = o^c
  snb = StarsBarsNN(ftcount, o)
  final_count = 0
  ftcountf = factorial(big(ftcount))
  for counts in snb
    factor = prod(factorial.(big.(counts)))
    final_count += BigInt(ftcountf // factor)
  end
  push!(mat_counts, final_count)
  return prod(mat_counts)^i
end

struct WiringData
  wiring::Wiring
  r::Real
  rp::Real
  Hdata # object holding additional details about the computed rates
end

# %%
# Specific wirings
const eg_Wmap = ((fill(1), fill(2)), ([2, 1], [2, 2]), ([1 1; 1 2], [2 1; 1 1]))

# binary AND (a = 1 unless all as = 2, then a = 2)
function and_Wmap(c::Integer, i::Integer)
  Ti = promote_type(typeof(c), typeof(i))
  Wmap = [[fill(Ti(x), repeat([2], j-1)...) for x in 1:i] for j in 1:c]
  push!(Wmap, [ones(Ti, repeat([2], c)...) for x in 1:i])
  for x in 1:i
    Wmap[c+1][x][repeat([2], c)...] = 2
  end
  return Wmap
end

# take the first output and ignore the rest
function first_Wmap(c::Integer, o::Integer, i::Integer)
  Ti = promote_type(typeof(c), typeof(i))
  Wmap = [[fill(Ti(x), repeat([o], j-1)...) for x in 1:i] for j in 1:c]
  push!(Wmap, [Array{Ti}(undef, repeat([o], c)...) for x in 1:i])
  for x in 1:i, a in 1:o
      Wmap[c+1][x][a, repeat([:], c-1)...] .= a
  end
  return Wmap
end

function and_corrs(N, corrs::Correlators)
  Eax, Eby, Eabxy = corrs
  iA, iB = length(Eax), length(Eby)

  EaxN = [1 - 2*((1-Eax[x])/2)^N for x in 1:iA]
  EbyN = [1 - 2*((1-Eby[y])/2)^N for y in 1:iB]
  EabxyN = [1 - ((1-Eax[x])^N + (1-Eby[y])^N)/(2.0^(N-1)) + (1-Eax[x]-Eby[y]+Eabxy[x,y])^N/(4.0^(N-1)) for x in 1:iA, y in 1:iB]

  return Correlators(EaxN, EbyN, EabxyN)
end

function sel_and_behav(N, behav::Behaviour, xs, ys)
  # TODO use wiring formalism
end
