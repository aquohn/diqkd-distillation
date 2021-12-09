using Revise
using LinearAlgebra, QuantumInformation

includet("helpers.jl")
includet("maxcorr.jl")

vLl = 999 * 689 * 10^-6 * cos(pi/50)^4
vNL = 0.6964
vcrit(vL) = (vL+1)/(3-vL)
