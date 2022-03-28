using Permutations

macro printvals(syms...)
  l = length(syms)
  quote
    $([(i == l) ? :(print(String($(Meta.quot(syms[i]))) * " = "); println($(syms[i]))) :
       :(print(String($(Meta.quot(syms[i]))) * " = "); print($(syms[i])); print(", ")) for i in eachindex(syms)]...)
  end |> esc
end

macro loadstruct(s)
  quote
    for f in fieldnames(typeof($s))
      eval(:($f = $$s.$f))
    end
  end |> esc
end

macro sliceup(arr, info...)
  l = Integer(length(info)/2)
  names = Vector{Any}(undef, l)
  steps = Vector{Any}(undef, l)
  for i in eachindex(names)
    names[i] = info[2*i-1]
    steps[i] = info[2*i]
  end

  quote
    curr = 1
    $([:(
         next = curr + $(steps[j]) - 1;
         $(names[j]) = $arr[curr:next]; 
         curr = next + 1
        ) 
       for j in 1:l]...)
  end |> esc
end

function sliceup(seq, steps...)
  l = length(steps)
  sliced = Vector{Vector{eltype(seq)}}(undef, l)
  curr = 1
  for j in 1:l
    next = curr + steps[j] - 1
    sliced[j] = [seq[curr:next]...]
    curr = next + 1
  end
  return sliced
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
const itprod = Iterators.product
const itrept = Iterators.repeated

# TODO PR to QuantumInformation.jl permutesystems
permutespaces(D, dims, systems::AbstractVector{<:Integer}) = permutespaces(D, dims, Permutation(systems))
permutespaces(M::AbstractMatrix, dims::AbstractVector{<:Integer}, P::Permutation) = permutespaces(M, [(d,d) for d in dims], P)
function permutespaces(M::AbstractMatrix, dims::AbstractVector{<:Tuple{<:Integer, <:Integer}}, P::Permutation)
  @assert length(P) == length(dims)
  rdims, cdims = zip(dims...) |> collect
  @assert (prod(rdims), prod(cdims)) == size(M)

  n = length(P)
  Pvec = P.data
  fullP = [Pvec; Pvec .+ n]
  reversed_indices = tuple(collect(2*n:-1:1)...)
  rrdims = reverse(rdims)
  rcdims = reverse(cdims)
  tensor = reshape(M, [rrdims...; rcdims...]...)

  # reverse tensor to match dims and P
  reversed_tensor = permutedims(tensor, reversed_indices)
  reversed_transposed_tensor = permutedims(reversed_tensor, fullP)
  transposed_tensor = permutedims(reversed_transposed_tensor, reversed_indices)
  return reshape(transposed_tensor, size(M))
end

function permutespaces(v::AbstractVector, dims::AbstractVector{<:Integer}, P::Permutation)
  @assert prod(dims) == length(v)
  @assert length(P) == length(dims)

  # v = kron(V[1], V[2], ...), length(V[j]) == dims[j]
  reversed_indices = tuple(collect(length(P):-1:1)...)
  tensor = reshape(v, reverse(dims)...)
  # tensor[..., j2, j1] = V[1][j1] * V[2][j2] * ...

  # reverse tensor to match dims and P
  reversed_tensor = permutedims(tensor, reversed_indices)
  reversed_transposed_tensor = permutedims(reversed_tensor, P.data)
  transposed_tensor = permutedims(reversed_transposed_tensor, reversed_indices)
  return reshape(transposed_tensor, size(v))
end
