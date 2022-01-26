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

#= function permutesystems(v::AbstractVector{T}, dims::Vector{Int}, P::Permutation) where T
  @assert length(v) == prod(dims)
  @assert length(P) == length(dims)
  
end =#

permutesystems(v::AbstractVector, dims::Vector{Int}, systems::Vector{Int}) = permutesystems(v, dims, Permutation(systems))
function permutesystems(v::AbstractVector, dims::Vector{Int}, P::Permutation)
  @assert prod(dims) == length(v)
  @assert length(P) == length(dims)

  # TODO something wrong here
  # if v = kron(V[1], V[2], ...), length(V[j]) == dims[j]
  reversed_indices = tuple(collect(length(P):-1:1)...)
  tensor = reshape(v, reverse(dims)...)

  # reverse tensor to match dims and P
  reversed_tensor = permutedims(tensor, reversed_indices)
  reversed_transformed_tensor = permutedims(reversed_tensor, P.data)
  transformed_tensor = permutedims(reversed_transformed_tensor, reversed_indices)
  return reshape(transformed_tensor, size(v))
end
