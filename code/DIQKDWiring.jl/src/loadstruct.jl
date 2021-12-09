struct BellSetting
  iA::Int
  oA::Int
  iB::Int
  oB::Int
end

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

function load_test()
  s = BellSetting(2,2,2,2)
  for f in fieldnames(typeof(s))
    @eval $f = $s.$f
  end
  @printvals(iA, oA, iB, oB)
end

function load_macro_test()
  s = BellSetting(2,2,2,2)
  @loadstruct s
  @printvals iA oA iB oB
end

struct Test
       t1
       t2
end

t = Test(1,2)

