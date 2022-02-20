using Distributions
using FastGaussQuadrature, DynamicPolynomials, TSSOS

includet("nonlocality.jl")

function reparam_gaussradau_generic(ts, ws, a, b, c, d, flip = false)
  v = flip ? b : a
  k = flip ? -1 : 1
  q = (b - a)//(d - c)
  newts = [k * (t - c) * q for t in ts] .+ v
  newws = abs(k * q) .* ws
  if flip
    return reverse(newts), reverse(newws)
  else
    return newts, newws
  end
end
function reparam_gaussradau(m, a, b, flip = false)
  t, w = gaussradau(m)
  return reparam_gaussradau_generic(t, w, a, b, -1, 1, flip)
end
loglb_gaussradau(m) = reparam_gaussradau(m, 0, 1, true)
logub_gaussradau(m) = reparam_gaussradau(m, 0, 1, false)

struct UpperSetting{T <: Integer}
  sett::Setting{T}
  oE::T
  dA::T
  dB::T
  dE::T
  oJ::Union{T, Nothing}
  function UpperSetting(sett::Setting{Ts},
      oE::Integer, dA::Integer, dB::Integer, dE::Integer
    ) where {Ts <: Integer}
    T = promote_type(Ts, typeof(oE), typeof(dA), typeof(dB), typeof(dE))
    new{T}(sett, oE, dA, dB, dE)
  end
end
function UpperSetting(iA::Integer, oA::Integer, iB::Integer, oB::Integer,
    oE::Integer, dA::Integer, dB::Integer, dE::Integer
  )
  return UpperSetting(Setting(iA, oA, iB, oB), oE, dA, dB, dE)
end
function Base.iterate(us::UpperSetting)
  return us.sett, reverse([us.oE, us.dA, us.dB, us.dE])
end
Base.iterate(us::UpperSetting, state) = isempty(state) ? nothing : (pop!(state), state)
macro expanduppersett(us)
  quote
    @eval begin
      sett, oE, dA, dB, dE = us
      iA, oA, iB, oB = sett
      oJ = isnothing(us.oJ) ? oA*iA*oB*iB*oE : us.oJ
    end
  end |> esc
end

# generates a probability map for J|V1V2E uniformly at random
function uniform_pJ(us, type::Type{T} = Float64) where T <: Real
  @expanduppersett us
  pJ = Array{T}(undef, oJ, oA, iA, oB, iB, oE)
  for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
    pJ[:, a, x, b, y, e] = rand(Dirichlet(oJ, 1.0))
  end
  return pJ
end

function logvar_vars(us::UpperSetting)
  @expanduppersett us
  pJ_V1V2E_count = oJ * oA*iA*oB*iB*oE 
  pk_count = dA * dB * dE
  corr_param_count = dA*oA*iA + dB*oB*iB + dE*oE
  z_count = oJ * iA*oA*iB*oB + oJ * oA*iA*oB*iB*oE
  tot = pJ_V1V2E_count + pk_count + corr_param_count + z_count
  return pJ_V1V2E_count, pk_count, corr_param_count, z_count, tot
end

z_min_term(p1, p2, t) = z_min_num(p1) / z_min_den(p1, p2, z)
z_min_den(p1, p2, t) = p1*(1-t) + p2*t
z_min_num(p1) = p1^2  # absorbed minus signed

function gauss_radau_objpolys(us, pin, p, m, pJ, pvgv, M, N, O, P)
  @expanduppersett us
  pvve = [pin[x,y] * sum([M[kA, a, x] * N[kB, b, y] * O[kE, e] * P[kA, kB, kE]
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)])
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  pvv = [pin[x,y] * p[a, b, x, y]
          for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  T, W = loglb_gaussradau(m)

  objpolys = []
  for i in 1:m
    ci = W[i]/(T[i]*log(2))  # bring in the minus sign
    for (a, x, b, y, j) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ)
      pvvj1 = sum([pvve[a, x, b, y, e] * pJ[j, a, x, b, y, e] for e in 1:oE])
      pvvj2 = sum([pvgv[a, x, b, y] * sum([pvve[ap, xp, b, y, e] * pJ[j, ap, xp, b, y, e]
                                           for (ap, xp) in itprod(1:oA, 1:iA) for e in 1:oE])])
      push!(objpolys, [ci * z_min_num(pvvj1), z_min_den(pvvj1, pvvj2, T[i])])
    end

    for (a, x, b, y, e, j) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE, 1:oJ)
      pvvje1 = pvve[a, x, b, y, e] * pJ[j, a, x, b, y, e]
      pvvje2 = pvve[a, x, b, y, e] *
      sum([pvve[ap, xp, bp, yp, e] * pJ[j, ap, xp, bp, yp, e]
             for (ap, xp, bp, yp) in itprod(1:oA, 1:iA, 1:oB, 1:iB)])
      push!(objpolys, [ci * z_min_num(pvvje1), z_min_den(pvvje1, pvvje2, T[i])])
    end
  end

  return objpolys
end

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
    push!(eqconstr, constr - p[a,b,x,y])
  end

  return ineqconstrs, eqconstrs
end

function logarithmic_obj(us::UpperSetting, pin::AbstractArray, p::Behaviour, pJ=nothing)
  @expanduppersett us
  if isnothing(pJ)
    @polyvar pJ[1:oJ, 1:oA, 1:iA, 1:oB, 1:iB]
  end
end

function poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer, pJ=nothing)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]

  Jvar = false
  if isnothing(pJ)
    Jvar = true
    @polyvar pJ[1:oJ, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE]
    ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p, pJ)
  else
    ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)
  end

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  pvgv = [pin[x,y] * p[a, b, x, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  objpolys = gauss_radau_objpolys(us, pin, p, m, pJ, pvgv, M, N, O, P)

  @polyvar R[1:2, 1:m]
  obj = sum(R)
  for i in 1:m
    vvjpolys, vvjepolys = objpolys[2i-1], objpolys[2i]
    push!(eqconstrs, vvjpolys[1] + R[1,i] * vvjpolys[2])
    push!(eqconstrs, vvjepolys[1] + R[2,i] * vvjepolys[2])
  end

  return obj, ineqconstrs, eqconstrs
end

function simple_poly_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer, pJ)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]

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
  pje = [sum([pJ[j, a, x, b, y, e] * pvv[a, x, b, y]
              for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]) 
         for (j, e) in itprod(1:oJ, 1:oE)]
  pv2je = [sum(pvvje[:, :, b, y, j, e])
         for (b, y, j, e) in itprod(1:oB, 1:iB, 1:oJ, 1:oE)]
  fvvje = [pJ[j, a, x, b, y, e] / (pvgv[a, x, b, y] * pje[j, e]) 
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]

  T, W = loglb_gaussradau(m)
  @polyvar R[1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE, 1:oJ]
  obj = sum(R) + sum([pvvje[idxs...] * fvvje[idxs...]
                      for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)])
  for i in 1:m
    ci = T[i] / (W[i] * log(2))
    for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
      a, x, b, y, j, e = idxs
      den = z_min_den(pvvje[idxs...], pv2je[b,y,j,e], T[i])
      num = ci * z_min_num(pvvje[idxs...])
      push!(eqconstrs, num - R[i, idxs...] * den)
    end
  end

  return obj, ineqconstrs, eqconstrs
end

function rational_setup(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer, pJ=nothing)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]

  Jvar = false
  if isnothing(pJ)
    Jvar = true
    @polyvar pJ[1:oJ, 1:oA, 1:iA, 1:oB, 1:iB]
    ineqconstr, eqconstr = generate_constrs(us, M, N, O, P, pJ)
  else
    ineqconstr, eqconstr = generate_constrs(us, M, N, O, P)
  end

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  pvgv = [pin[x,y] * p[a, b, x, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  objpolys = gauss_radau_objpolys(us, pin, p, m, pJ, pvgv, M, N, O, P)

  for polys in objpolys
    obj += (polys[1]/polys[2]) + (polys[3]/polys[4])
  end

  return obj, ineqconstr, eqconstr
end

