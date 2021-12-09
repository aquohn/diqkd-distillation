using DataFrames
using CSV
includet("wiring.jl")

function wiring_table(s::Setting{IdxT}, c::IdxT, ::Type{T} = Float32) where {T <: Real, IdxT <: Integer}
  oA, oB, iA, iB = (s...,)
  probs = [Symbol("p($a,$b|$x,$y)") => T[] for a in 1:iA for b in 1:iB for x in 1:oA for y in 1:oB]
  CAs = [Symbol("CA$seq") => IdxT[] for seq in
         Iterators.product(vcat([1:iA for i in 1:c], [1:oA for i in 1:c])...)] |> vec
  CAjs = vcat([[Symbol("CA$j$seq") => IdxT[] for seq in
                Iterators.product(vcat([1:iA for i in 1:(j-1)], [1:oA for i in 1:(j-1)])...)] for j in 2:c]...) |> vec
  CBs = [Symbol("CB$seq") => IdxT[] for seq in
         Iterators.product(vcat([1:iB for i in 1:c], [1:oB for i in 1:c])...)] |> vec
  CBjs = vcat([[Symbol("CB$j$seq") => IdxT[] for seq in
                Iterators.product(vcat([1:iB for i in 1:(j-1)], [1:oB for i in 1:(j-1)])...)] for j in 2:c]...) |> vec
  return DataFrame(vcat(probs, CAs, CAjs, CBs, CBjs))
end

function store_wiring(df, behav, wiring)
  oA, oB, iA, iB = size(behav.pabxy)
  c = length(wiring.CAj)
  vals = Dict{Symbol, Union{eltype(pabxy), eltype(CA)}}()

  for a in 1:iA, b in 1:iB, x in 1:oA, y in 1:oB
    vals[Symbol("p($a,$b|$x,$y)")] = pabxy[a,b,x,y]
  end
  for seq in Iterators.product(vcat([1:iA for i in 1:c], [1:oA for i in 1:c])...)
    vals[Symbol("CA$seq")] = CA[seq...]
  end
  for seq in Iterators.product(vcat([1:iB for i in 1:c], [1:oB for i in 1:c])...)
    vals[Symbol("CB$seq")] = CB[seq...]
  end
  for j in 2:c
    for seq in Iterators.product(vcat([1:iA for i in 1:(j-1)], [1:oA for i in 1:(j-1)])...)
      vals[Symbol("CA$j$seq")] = CAj[j][seq...]
    end
    for seq in Iterators.product(vcat([1:iB for i in 1:(j-1)], [1:oB for i in 1:(j-1)])...)
      vals[Symbol("CB$j$seq")] = CBj[j][seq...]
    end
  end
end

# TODO dump to CSV
