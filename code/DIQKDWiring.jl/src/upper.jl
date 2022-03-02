using Distributions, FastGaussQuadrature

includet("helpers.jl")
includet("nonlocality.jl")

# symbolic operations
sprod(args...) = Expr(:call, :*, args...)
ssum(args...) = Expr(:call, :+, args...)
sprod(args) = Expr(:call, :*, args...)
ssum(args) = Expr(:call, :+, args...)

# arithmetic operations
aprod(args...) = prod(args)
asum(args...) = sum(args)
aprod(args) = prod(args)
asum(args) = sum(args)

function modeops(mode)
  if mode == :arith
    return asum, aprod
  elseif mode == :sym
    return ssum, sprod
  end
end

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

# ineqconstrs <= 0
min_lagrangian(obj, mus, ineqconstrs, lambdas, eqconstrs) = obj + mus' * ineqconstrs + lambdas' * eqconstrs
lagrangian_grad(L, vars, diff) = [diff(L, var) for var in vars]
lagrangian_hessian(dL, vars, diff) = [diff(dvL, var) for dvL in dL, var in vars]

struct UpperSetting{T <: Integer}
  sett::Setting{T}
  oE::T
  dA::T
  dB::T
  dE::T
  oJ::T
  function UpperSetting(sett::Setting{Ts},
      oE::Integer, dA::Integer, dB::Integer, dE::Integer, oJ::Union{Tj, Nothing} = nothing
    ) where {Ts <: Integer, Tj <: Integer}
    T = promote_type(Ts, typeof(oE), typeof(dA), typeof(dB), typeof(dE))
    if !isnothing(oJ)
      T = promote_type(Tj, T)
    else
      iA, oA, iB, oB = sett
      oJ = oA*iA*oB*iB*oE
    end
    new{T}(sett, oE, dA, dB, dE, oJ)
  end
end
function UpperSetting(iA::Integer, oA::Integer, iB::Integer, oB::Integer,
    oE::Integer, dA::Integer, dB::Integer, dE::Integer, oJ::Union{Tj, Nothing} = nothing
  ) where {Tj <: Integer}
  return UpperSetting(Setting(iA, oA, iB, oB), oE, dA, dB, dE, oJ)
end
Base.iterate(us::UpperSetting) = us.sett, reverse([us.oE, us.dA, us.dB, us.dE, us.oJ])
Base.iterate(us::UpperSetting, state) = isempty(state) ? nothing : (pop!(state), state)
macro expanduppersett(us)
  quote
    @eval begin
      sett, oE, dA, dB, dE, oJ = $us
      iA, oA, iB, oB = sett
    end
  end |> esc
end

# count number of variables
function logvar_varcount(us::UpperSetting)
  @expanduppersett us
  pJ_V1V2E_count = oJ * oA*iA*oB*iB*oE 
  pk_count = dA * dB * dE
  corr_param_count = dA*oA*iA + dB*oB*iB + dE*oE
  z_count = oJ * iA*oA*iB*oB + oJ * oA*iA*oB*iB*oE
  tot = pJ_V1V2E_count + pk_count + corr_param_count + z_count
  return pJ_V1V2E_count, pk_count, corr_param_count, z_count, tot
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
function uniform_pin(us::UpperSetting)
  iA, iB = us.sett.iA, us.sett.iB
  return fill(1//iA * 1//iB, iA, iB)
end

function generate_leqconstrs(us, M, N, O, P, pJ = nothing)
  @expanduppersett us

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

  return eqconstrs
end

function generate_nleqconstr_exprs(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

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
    push!(eqconstrs, constr)
  end

  return eqconstrs
end


function generate_constrs(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

  # expr <= 0
  ineqconstrs = -1 .* [M..., N..., O..., P...]
  if !isnothing(pJ)
    ineqconstrs = vcat(ineqconstrs, -1 .* vec(pJ))
  end

  # expr == 0
  # linear constraints
  eqconstrs = generate_leqconstrs(us, M, N, O, P, pJ)

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

# Rational approximations
zmin_term(p1, p2, t) = zmin_num(p1, p2, t) / zmin_den(p1, p2, t)
zmin_den(p1, p2, t) = p1*(1-t) + p2*t
zmin_num(p1, p2, t) = -p1^2

szmin_term(p1, p2, t) = Expr(:call, :/, szmin_num(p1, p2, t), szmin_den(p1, p2, t))
szmin_den(p1, p2, t) = ssum(sprod(p1, 1-t), sprod(p2, t))
szmin_num(p1, p2, t) = sprod(-1, p1, p1)

gr_term(p1, p2, t) = gr_num(p1, p2, t) / gr_den(p1, p2, t)
gr_den(p1, p2, t) = t*(p1 - p2) + p2
gr_num(p1, p2, t) = p1*(p1 - p2)

function gr_relent_polypairs(probs::Array{Tuple{T1, T2}}, m; grmode=:logub, termmode=:gr, kwargs...) where {T1, T2}
  # NOTE logub + zmin blows up
  # TODO check lzmin
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
    cs = [-W[i]/(T[i]*log(2)) for i in 1:m]
    polypart = -sum(cs)
  end

  Texp = promote_type(T1, T2, eltype(cs))
  polypairs = Array{Tuple{Texp, Texp}}(undef, m, size(probs)...)
  it = itprod((1:s for s in size(probs))...)
  for i in 1:m
    for idxs in it
      p1, p2 = probs[idxs...]
      polypairs[i, idxs...] = (cs[i] * numf(p1, p2, T[i]), denf(p1, p2, T[i]))
    end
  end

  return polypairs, polypart
end

function apply_epigraph!(eqconstrs, polypairs, polypart, R, idxit)
  obj = sum(R) + polypart
  for idxs in idxit
    poly = polypairs[idxs...]
    push!(eqconstrs, poly[1] - poly[2] * R[idxs...])
  end
  return obj
end

# generate probability expressions
# bound directions shown for zmin with loglb

# pvve log(pvve/pv1|e pv2e) <= pvve zmin(pvve/pv1|e pv2e)
function CMI_probs(us, pin, pvgv, M, N, O, P, mode=:arith)
  @expanduppersett us
  msum, mprod = modeops(mode)

  pvve = [mprod(pin[x,y], msum(mprod(M[kA, a, x], N[kB, b, y], O[kE, e], P[kA, kB, kE])
                                for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)))
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  py = [msum(pin[:,y]) for y in 1:iB]
  pv2e = [mprod(py[y], msum(mprod(N[kB, b, y], O[kE, e], P[kA, kB, kE])
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE))
                for (b, y, e) in itprod(1:oB, 1:iB, 1:oE))]

  Texp = promote_type(eltype.([pvve, pv2e, px])...)
  probs = Array{Tuple{Texp, Texp}}(undef, oA, iA, oB, iB, oE)
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
    a, x, b, y, e = idxs
    probs[idxs...] = (pvve[idxs...], mprod(pvgv[a, x, b, y], pv2e[b, y, j, e]))
  end

  return probs
end

# -pv1e log(pv1e/pe) >= pv1e zmin(pv1e/pe)
function HAgE_probs(us, pin, M, N, O, P, mode=:arith)
  @expanduppersett us
  msum, mprod = modeops(mode)

  px = [msum(pin[:,x]) for x in 1:iA]
  pv1e = [mprod(px[x], msum(
               [mprod(M[kA, a, x], O[kE, e], P[kA, kB, kE])
                for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)]...))
          for (a, x, e) in itprod(1:oA, 1:iA, 1:oE)]
  pe = [msum(mprod(O[kE, e], P[kA, kB, kE])
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE))
          for e in 1:oE]

  Texp = promote_type(eltype.([px, pv1e, pe])...)
  probs = Array{Tuple{Texp, Texp}}(undef, oA, iA, oE)
  for idxs in itprod(1:oA, 1:iA, 1:oE)
    a, x, e = idxs
    probs[idxs...] = (pv1e[idxs...], pe[e])
  end

  return probs
end

# pvve log(pvve/pv1|v2 pj|e pv2je) <= pvve zmin(pvve/pv1|e pv2e)
function full_probs(us, pin, p, pJ, pvgv, M, N, O, P, mode=:arith)
  @expanduppersett us
  msum, mprod = modeops(mode)

  pvve = [mprod(pin[x,y], msum(mprod(M[kA, a, x], N[kB, b, y], O[kE, e], P[kA, kB, kE])
                                for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE)))
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  pvv = [mprod(pin[x,y], p[a, b, x, y])
          for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  pvvje = [mprod(pJ[j, a, x, b, y, e], pvve[a, x, b, y, e])
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]
  pjge = [msum(mprod(pJ[j, a, x, b, y, e], pvv[a, x, b, y])
              for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)) 
         for (j, e) in itprod(1:oJ, 1:oE)]
  pv2je = [msum(pvvje[:, :, b, y, j, e])
         for (b, y, j, e) in itprod(1:oB, 1:iB, 1:oJ, 1:oE)]

  Texp = promote_type(eltype.([pvve, pvv, pvvje, pjge, pv2je])...)
  probs = Array{Tuple{Texp, Texp}}(undef, oA, iA, oB, iB, oJ, oE)
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
    a, x, b, y, j, e = idxs
    pvvje1 = mprod(pvvje[idxs...], pJ[j, a, x, b, y, e])
    pvvje2 = mprod(pvgv[a, x, b, y], pjge[j, e], pv2je[b, y, j, e])
    probs[idxs...] = (pvvje1, pvvje2)
  end

  return probs
end

