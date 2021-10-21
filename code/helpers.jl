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

kd(i,j) = (i == j) ? 1 : 0
E(M, rho) = tr(M * rho)
sigmas = [[kd(j, 3) kd(j,1)-im*kd(j,2); kd(j,1)+im*kd(j,2) -kd(j,3)] for j in 1:3]
