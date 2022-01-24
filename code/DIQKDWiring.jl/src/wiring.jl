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

num_lds(c, sett::Setting) = num_lds(c, sett.oA, sett.iA) * num_lds(c, sett.oB, sett.iB)
num_lds(c, o, i) = (o * i^c)^(i^c * o)
struct LDWiringIter{T <: Integer}
  c::T
  o::T
  i::T
  tmin::Vector{T}
  tmax::Vector{T}
  smin::Vector{T}
  smax::Vector{T}
  function LDWiringIter(c::Integer, o::Integer, i::Integer)
    T = promote_type(typeof(c), typeof(o), typeof(i))
    new{T}(c, o, i, repeat([1], c+1), vcat(repeat([i], c), o), repeat([1], c+1), vcat(i, repeat([o], c)))
  end
end
function ldwiring_check_prob(ldwit::LDWiringIter{Ti}, chi::Array{Tp}) where {Tp <: Real, Ti <: Integer}
  valid = true
  c, o, i = ldwit.c, ldwit.o, ldwit.i
  for j in 1:c
    println(chi)
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
Base.iterate(ldwit::LDWiringIter) = iterate(ldwit, nothing)
function Base.iterate(ldwit::LDWiringIter{T}, initstate) where T
  c, o, i = ldwit.c, ldwit.o, ldwit.i
  ts = (t for t in itprod(1:o, repeat([1:i], c)...))  # diff targets
  ss = (s for s in itprod(repeat([1:o], c)..., 1:i))  # diff sources
  lenss = length(ss)  # number of diff sources
  dtsit = itprod(repeat([ts], lenss)...)  # iterator over possible deterministic targets
  if isnothing(initstate)
    dtsitval = iterate(dtsit)
  else
    dtsitval = iterate(dtsit, initstate)
  end
  while !isnothing(dtsitval)
    dts, state = dtsitval
    chi = zeros(T, o, repeat([i], c)..., i, repeat([o], c)...)
    for (dt, s) in zip(dts, ss)
      chi[dt..., s...] = 1
    end
    if ldwiring_check_prob(ldwit, chi)
      return chi, state
    end
    dtsitval = iterate(dtsit, state)
  end
end

num_wiring_maps(c, sett::Setting) = num_wiring_maps(c, sett.oA, sett.iA) * num_wiring_maps(c, sett.oB, sett.iB)
num_wiring_maps(c, o, i) = o^(i * o^c) * prod([i^(i * o^(j-1)) for j in 1:c])
num_wiring_maps_fix(c, sett::Setting, fA, fB) = num_wiring_maps_fix(c, sett.oA, sett.iA, fA) * num_wiring_maps_fix(c, sett.oB, sett.iB, fB)
function num_wiring_maps_fix(c, o, i, f)
  if f > i
    throw(ArgumentError("$f inputs to fix, but there are only $i inputs!"))
  end
  o^((i-f) * o^c) * prod([i^((i-f) * o^(j-1)) for j in 1:c])
end

# %%
# Wiring exhaustive search


# %%
# Specific wirings

function and_corrs(N, corrs::Correlators)
  Eax, Eby, Eabxy = corrs
  iA, iB = length(Eax), length(Eby)

  EaxN = [1 - 2*((1-Eax[x])/2)^N for x in 1:iA]
  EbyN = [1 - 2*((1-Eby[y])/2)^N for y in 1:iB]
  EabxyN = [1 - ((1-Eax[x])^N + (1-Eby[y])^N)/(2.0^(N-1)) + (1-Eax[x]-Eby[y]+Eabxy[x,y])^N/(4.0^(N-1)) for x in 1:iA, y in 1:iB]

  return Correlators(EaxN, EbyN, EabxyN)
end
