using JuMP, Ipopt

includet("upper.jl")
sprod(args...) = Expr(:call, :*, args...)
ssum(args...) = Expr(:call, :+, args...)
sprod(args::AbstractArray) = Expr(:call, :*, args...)
ssum(args::AbstractArray) = Expr(:call, :+, args...)

# constraints
function generate_lin_constrs(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

  # expr >= 0
  ineqconstrs = []

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

  return ineqconstrs, eqconstrs
end

function generate_nonlin_constr_expressions(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

  # expr >= 0
  ineqconstrs = Expr[]

  # expr == 0
  eqconstrs = Expr[]
  # behaviour constrs
  for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)
    constrterms = Expr[]
    for (kA, kB) in itprod(1:dA, 1:dB)
      sumterm = sprod(M[kA, a, x], N[kB, b, y], sum(P[kA, kB, :]))
      push!(constrterms, sumterm)
    end
    constr = ssum(constrterms..., -p[a,b,x,y])
    push!(eqconstrs, :($constr == 0))
  end

  return ineqconstrs, eqconstrs
end

sz_min_den(p1, p2, t) = ssum(sprod(p1, 1-t), sprod(p2, t))
sz_min_num(p1) = sprod(p1, p1)  # absorbed minus sign

# TODO move probs functions to upper and switch between Expr and arithmetic modes

function CMI_probs_expr(us, pin, pvgv, M, N, O, P)
  @expanduppersett us
  pvve = [sprod(pin[x,y], ssum([sprod(M[kA, a, x], N[kB, b, y], O[kE, e], P[kA, kB, kE])
                                for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)]))
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  py = [ssum(pin[:,y]) for y in 1:iB]
  pv2e = [sprod(py[y], ssum([sprod(N[kB, b, y], O[kE, e], P[kA, kB, kE])
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)])
                for (b, y, e) in itprod(1:oB, 1:iB, 1:oE))]

  Texp = promote_type(eltype.([pvve, pv2e, px])...)
  probs = Array{Tuple{Texp, Texp}}(undef, oA, iA, oB, iB, oE)
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
    a, x, b, y, e = idxs
    probs[idxs...] = (pvve[idxs...], sprod(pvgv[a, x, b, y], pv2e[b, y, j, e]))
  end

  return probs
end

function HAgE_probs_expr(us, pin, M, N, O, P)
  @expanduppersett us
  px = [ssum(pin[:,x]) for x in 1:iA]
  pv1e = [sprod(px[x], ssum(
               [sprod(M[kA, a, x], O[kE, e], P[kA, kB, kE])
                for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)]...))
          for (a, x, e) in itprod(1:oA, 1:iA, 1:oE)]
  pe = [sum([sprod(O[kE, e], P[kA, kB, kE])
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)])
          for e in 1:oE]

  Texp = promote_type(eltype.([px, pv1e, pe])...)
  probs = Array{Tuple{Texp, Texp}}(undef, oA, iA, oE)
  for idxs in itprod(1:oA, 1:iA, 1:oE)
    a, x, e = idxs
    probs[idxs...] = (pv1e[idxs...], pe[e])
  end

  return probs
end

function full_probs_expr(us, pin, p, pJ, pvgv, M, N, O, P)
  @expanduppersett us
  pvve = [sprod(pin[x,y], ssum([sprod(M[kA, a, x], N[kB, b, y], O[kE, e], P[kA, kB, kE])
                                for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)]))
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  pvv = [sprod(pin[x,y], p[a, b, x, y])
          for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  pvvje = [sprod(pJ[j, a, x, b, y, e], pvve[a, x, b, y, e])
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]
  pjge = [ssum([sprod(pJ[j, a, x, b, y, e], pvv[a, x, b, y])
              for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]) 
         for (j, e) in itprod(1:oJ, 1:oE)]
  pv2je = [ssum(pvvje[:, :, b, y, j, e])
         for (b, y, j, e) in itprod(1:oB, 1:iB, 1:oJ, 1:oE)]

  Texp = promote_type(eltype.([pvve, pvv, pvvje, pjge, pv2je])...)
  probs = Array{Tuple{Texp, Texp}}(undef, oA, iA, oB, iB, oJ, oE)
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
    a, x, b, y, j, e = idxs
    pvvje1 = sprod(pvvje[idxs...], pJ[j, a, x, b, y, e])
    pvvje2 = sprod(pvgv[a, x, b, y], pjge[j, e], pv2je[b, y, j, e])
    probs[idxs...] = (pvvje1, pvvje2)
  end

  return objpolys
end


function poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer, pJ=nothing)
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
  objpolys = gauss_radau_objpolys(probs, m)

  obj = sum(R)
  for idxs in itprod(1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
    poly = objpolys[idxs...]
    push!(eqconstrs, poly[1] - poly[2] * R[idxs...])
  end

  return obj, ineqconstrs, eqconstrs, vars
end

function simple_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer, pJ)
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
      den = z_min_den(pvvje[idxs...], pv2je[b,y,j,e], T[i])
      num = cs[i] * z_min_num(pvvje[idxs...])
      push!(eqconstrs, num - R[i, idxs...] * den)
    end
  end

  return obj, ineqconstrs, eqconstrs, vars
end

# heuristic: minimise I(V1:V2|E) or H(V1|E) - H(V1|V2) first as a heuristic guess for ABE behaviour

function CMI_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer)
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
  objpolys = gauss_radau_objpolys(probs, m)

  obj = sum(R)
  for idxs in itprod(1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
    poly = objpolys[idxs...]
    push!(eqconstrs, poly[1] - poly[2] * R[idxs...])
  end

  return obj, ineqconstrs, eqconstrs, vars
end

function HAgE_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer)
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
  HAgE_probs = HAgE_(us, pin, M, N, O, P)
  objpolys = gauss_radau_objpolys(probs, m)

  obj = sum(R)
  for idxs in itprod(1:m, 1:oA, 1:iA, 1:oE)
    poly = objpolys[idxs...]
    push!(eqconstrs, poly[1] - poly[2] * R[idxs...])
  end

  return obj, ineqconstrs, eqconstrs, vars
end
