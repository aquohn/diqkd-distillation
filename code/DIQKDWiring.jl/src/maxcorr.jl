using Revise
using Symbolics, LinearAlgebra, QuantumInformation

includet("helpers.jl")
includet("nonlocality.jl")

function pauli_to_reg(v)
  Mv = [1 0 0 1; 0 1 -im 0; 0 1 im 0; 1 0 0 -1] * v
  return [Mv[1] Mv[2]; Mv[3] Mv[4]]
end

function reg_to_pauli(M)
  R = [0.5 0 0 0.5; 0 0.5 0.5 0; 0 0.5im 0.5im 0; 0.5 0 0 -0.5]
  return R * [M[1,1], M[1,2], M[2,1], M[2,2]]
end

function pauli_to_sv(v)
  p = sum([abs(z)^2 for z in v])
  m = v[1]^2 - sum([z^2 for z in v[2:4]])
  pm = sqrt(abs(p)^2 - abs(m)^2)
  return sqrt(p + pm), sqrt(p - pm)
end

#=
Symbolics.@variables p[1:2, 1:2]
PAB = [p[1,1] p[1,2]; p[2,1] p[2,2]]
PA = [p[1,1]+p[1,2] 0; 0 p[2,1]+p[2,2]]
PB = [p[1,1]+p[2,1] 0; 0 p[1,2]+p[2,2]]
Ptld = sqrt(PA) * PAB * sqrt(PB)
lambdap, lambdam = Ptld |> reg_to_pauli |> pauli_to_sv
=#

function maxcorrs(pax::Array{T}, pby::Array{T}, pabxy::Array{T}) where T <: Real
  oA, oB, iA, iB = size(pabxy)
  corrs = Array{T}(undef, iA, iB)
  for x in 1:iA, y in 1:iB
    PAB = Array{T}(undef, oA, oB)
    PA = zeros(T, oA, oA)
    PB = zeros(T, oB, oB)
    for a in 1:oA, b in 1:oB
      PAB[a, b] = abs(pabxy[a, b, x, y])
    end
    for a in 1:oA
      PA[a, a] = abs(pax[a,x])
    end
    for b in 1:oB
      PB[b, b] = abs(pby[b,y])
    end

    Ptld = pinv(PA^(0.5)) * PAB * pinv(PB^(0.5))
    corrs[x,y] = svd(Ptld).S[2]
  end

  return corrs
end

function maxcorrs(rhoAB::AbstractArray{T}, dA::Integer) where T <: Complex
  dAdB = size(rhoAB)[1]; dB = Integer(dAdB / dA)
  rhoA = ptrace(rhoAB, [dA, dB], 2)
  rhoB = ptrace(rhoAB, [dA, dB], 1)

  rhotld = kron(I(dA), pinv(rhoB^(0.5))) * rhoAB * kron(pinv(rhoA^(0.5)), I(dB))
  rhotldsvd = svd(rhotld)
  svdbasis = [kron(reshape(rhotldsvd.U[:,i], dA, dA), reshape(rhotldsvd.V[:,i], dB, dB)) for i in 1:length(rhotldsvd.S)]
  lambdas = [tr(basis' * rhotld) for basis in svdbasis]

  return lambdas
end

function maxcorrs_analyse(rhoAB::AbstractArray{T}, dA::Integer) where T <: Complex
  dAdB = size(rhoAB)[1]; dB = Integer(dAdB / dA)
  rhoA = ptrace(rhoAB, [dA, dB], 2)
  rhoB = ptrace(rhoAB, [dA, dB], 1)

  rhotld = kron(I(dA), pinv(rhoB^(0.5))) * rhoAB * kron(pinv(rhoA^(0.5)), I(dB))
  rhotldsvd = svd(rhotld)
  display(rhotldsvd)

  svdbasis = [kron(reshape(rhotldsvd.U[:,i], dA, dA), reshape(rhotldsvd.V[:,i], dB, dB)) for i in 1:length(rhotldsvd.S)]
  lambdas = [tr(basis' * rhotld) for basis in svdbasis]

  println("rhotld")
  display(rhotld)
  println()

  println("kron(rhoA, rhoB)")
  display(kron(rhoA, rhoB))
  println()

  println("lambdas .* svdbasis")
  display(sum(lambdas .* svdbasis))
  println()

  println("svdbasis and lambdas")
  for idx in eachindex(svdbasis)
    println(lambdas[idx])
    display(svdbasis[idx])
    println()
  end

  #= svdvecbasis = [kron(rhotldsvd.U[:,i], rhotldsvd.V[:,i]) for i in 1:length(rhotldsvd.S)]
  rhotldvec = vec(rhotld)
  veclambdas = [basis' * rhotldvec for basis in svdvecbasis]
  println("svdvecbasis and lambdas")
  for idx in eachindex(svdvecbasis)
    println(veclambdas[idx])
    display(svdvecbasis[idx])
    println()
  end =#

  return rhotldsvd, lambdas, svdbasis
end
