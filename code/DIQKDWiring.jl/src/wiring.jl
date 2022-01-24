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
# Counting wirings

num_lds(c, sett::Setting) = num_lds(c, sett.oA, sett.iA) * num_lds(c, sett.oB, sett.iB)
num_lds(c, o, i) = (o * i^c)^(i^c * o)
num_wirings(c, sett::Setting) = num_wirings(c, sett.oA, sett.iA) * num_wirings(c, sett.oB, sett.iB)
num_wirings(c, o, i) = o^(i * o^c) * prod([i^(i * o^(j-1)) for j in 1:c])
num_wirings_fix(c, sett::Setting, fA, fB) = num_wirings_fix(c, sett.oA, sett.iA, fA) * num_wirings_fix(c, sett.oB, sett.iB, fB)
function num_wirings_fix(c, o, i, f)
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
