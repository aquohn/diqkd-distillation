using DynamicPolynomials, TSSOS

includet("upper.jl")

function generate_constrs(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

  # expr >= 0
  ineqconstrs = [M..., N..., O..., P...]
  if !isnothing(pJ)
    ineqconstrs = vcat(ineqconstrs, vec(pJ))
  end

  # expr == 0
  eqconstrs = []
  # pJ normalisation
  if !isnothing(pJ)
    for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
      push!(eqconstrs, sum(pJ[:, a, x, b, y, e]))
    end
  end
  # P normalisation
  push!(eqconstrs, sum(P) - 1)
  # POVM normalisation
  for kA in 1:dA, x in 1:iA
    push!(eqconstrs, sum(M[kA, :, x]) - 1)
  end
  for kB in 1:dB, y in 1:iB
    push!(eqconstrs, sum(N[kB, :, y]) - 1)
  end
  for kE in 1:dE
    push!(eqconstrs, sum(O[kE, :]) - 1)
  end
  # behaviour constrs
  for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)
    constr = 0
    for (kA, kB) in itprod(1:dA, 1:dB)
      constr += M[kA, a, x] * N[kB, b, y] * sum(P[kA, kB, :])
    end
    push!(eqconstrs, constr - p[a,b,x,y])
  end

  return ineqconstrs, eqconstrs
end

# polynomial approximations
function gauss_radau_objpolys(probs::Array{Tuple{T1, T2}}, m; grmode=:loglb, termmode=:zmin, kwargs...) where {T1, T2}
  # NOTE logub + zmin blows up
  if grmode == :logub
    T, W = logub_gaussradau(m)
  else  # default: loglb
    T, W = loglb_gaussradau(m)
  end
  if termmode == :gr
    numf = gr_num
    denf = gr_den
    cs = [W[i]/log(2) for i in 1:m]
    polypart = 0
  else  # default: zmin
    numf = zmin_num
    denf = zmin_den
    cs = [W[i]/(T[i]*log(2)) for i in 1:m]
    polypart = sum(cs)
  end

  Texp = promote_type(T1, T2, eltype(cs))
  objpolys = Array{Tuple{Texp, Texp}}(undef, m, size(probs)...)
  it = itprod((1:s for s in size(probs))...)
  for i in 1:m
    for idxs in it
      p1, p2 = probs[idxs...]
      objpolys[i, idxs...] = (cs[i] * numf(p1, p2, T[i]), denf(p1, p2, T[i]))
    end
  end

  return objpolys, polypart
end

function apply_epigraph!(eqconstrs, ratpolys, polypart, R, idxit)
  obj = sum(R) + polypart
  for idxs in idxit
    poly = ratpolys[idxs...]
    push!(eqconstrs, poly[1] - poly[2] * R[idxs...])
  end
  return obj
end

function poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer; pJ=nothing, kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE]
  vars = [M..., N..., O..., P..., R...]

  Jvar = false
  if isnothing(pJ)
    Jvar = true
    @polyvar pJ[1:oJ, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE]
    ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p, pJ)
    vars = vcat(vars, vec(pJ))
  else
    ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)
  end

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  pvgv = [pin[x,y] * p[a, b, x, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  probs = full_probs(us, pin, p, pJ, pvgv, M, N, O, P)
  objpolys, polypart = gauss_radau_objpolys(probs, m; kwargs...)
  idxit = itprod(1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
  obj = apply_epigraph!(eqconstrs, objpolys, polypart, R, idxit) 

  return obj, ineqconstrs, eqconstrs, vars
end

function simple_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer, pJ; kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE]
  vars = [M..., N..., O..., P..., R...]
  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  pvv = [pin[x,y] * p[a, b, x, y]
          for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  pvgv = [pvv[a, x, b, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  pvve = [pin[x,y] * sum([M[kA, a, x] * N[kB, b, y] * O[kE, e] * P[kA, kB, kE]
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)])
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  pvvje = [pJ[j, a, x, b, y, e] * pvve[a, x, b, y, e] 
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]
  pjge = [sum([pJ[j, a, x, b, y, e] * pvv[a, x, b, y]
              for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]) 
         for (j, e) in itprod(1:oJ, 1:oE)]
  pv2je = [sum(pvvje[:, :, b, y, j, e])
         for (b, y, j, e) in itprod(1:oB, 1:iB, 1:oJ, 1:oE)]
  fvvje = [pJ[j, a, x, b, y, e] / (pvgv[a, x, b, y] * pjge[j, e]) 
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]

  T, W = loglb_gaussradau(m)
  obj = sum(R) + sum([pvvje[idxs...] * fvvje[idxs...]
                      for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)])

  cs = [W[i]/(T[i]*log(2)) for i in 1:m]
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
    a, x, b, y, j, e = idxs
    for i in 1:m
      den = zmin_den(pvvje[idxs...], pv2je[b,y,j,e], T[i])
      num = cs[i] * zmin_num(pvvje[idxs...], pv2je[b,y,j,e], T[i])
      push!(eqconstrs, num - R[i, idxs...] * den)
    end
  end

  return obj, ineqconstrs, eqconstrs, vars
end

# heuristic: minimise I(V1:V2|E) or H(V1|E) - H(V1|V2) first as a heuristic guess for ABE behaviour

function CMI_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer; kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE]
  vars = [M..., N..., O..., P..., R...]

  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  probs = CMI_probs(us, pin, M, N, O, P)
  objpolys, polypart = gauss_radau_objpolys(probs, m; kwargs...)
  idxit = itprod(1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
  obj = apply_epigraph!(eqconstrs, objpolys, polypart, R, idxit) 

  return obj, ineqconstrs, eqconstrs, vars
end

function HAgE_symengine_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer; kwargs...)
  @expanduppersett us
  M = [symbols("M_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:oA, 1:iA)]
  N = [symbols("N_{$(join(idxs, ';'))}") for idxs in itprod(1:dB, 1:oB, 1:iB)]
  O = [symbols("O_{$(join(idxs, ';'))}") for idxs in itprod(1:dE, 1:oE)]
  P = [symbols("P_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:dB, 1:dE)]
  R = [symbols("R_{$(join(idxs, ';'))}") for idxs in itprod(1:m, 1:oA, 1:iA, 1:oE)]


end

function HAgE_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer; kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oE]
  vars = [M..., N..., O..., P..., R...]

  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  probs = HAgE_probs(us, pin, M, N, O, P)
  objpolys, polypart = gauss_radau_objpolys(probs, m; kwargs...)
  idxit = itprod(1:m, 1:oA, 1:iA, 1:oE)
  obj = apply_epigraph!(eqconstrs, objpolys, polypart, R, idxit) 

  return obj, ineqconstrs, eqconstrs, vars
end

function cs_tssos_minimal(setupf, TS, params...)
  obj, ineqconstrs, eqconstrs, vars = setupf(params...)
  poly_prob = [obj, ineqconstrs..., eqconstrs...]
  opt, sol, data = cs_tssos_first(poly_prob, vars, "min", numeq=length(eqconstrs), TS=TS)
end
