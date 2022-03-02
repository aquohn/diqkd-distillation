using DynamicPolynomials, TSSOS
using HomotopyContinuation, SemialgebraicSets

includet("upper.jl")
includet("quantum.jl")

# polynomial approximations
function poly_setup(us::UpperSetting, pin::AbstractArray, p, m::Integer; pJ=nothing, kwargs...)
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

  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  pvgv = [pin[x,y] * p[a, b, x, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  probs = full_probs(us, pin, p, pJ, pvgv, M, N, O, P)
  polypairs, polypart = gr_relent_polypairs(probs, m; kwargs...)
  idxit = itprod(1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)
  obj = apply_epigraph!(eqconstrs, polypairs, polypart, R, idxit) 

  return obj, ineqconstrs, eqconstrs, vars
end

function simple_poly_setup(us::UpperSetting, pin::AbstractArray, p, m::Integer, pJ; kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE]
  vars = [M..., N..., O..., P..., R...]
  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)

  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  pvv = [pin[x,y] * p[a, b, x, y]
          for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  pvgv = [pvv[a, x, b, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  pvve = [pin[x,y] * sum(M[kA, a, x] * N[kB, b, y] * O[kE, e] * P[kA, kB, kE]
                          for (kA, kB, kE) in itprod(1:dA, 1:dB, 1:dE))
          for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)]
  pvvje = [pJ[j, a, x, b, y, e] * pvve[a, x, b, y, e] 
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]
  pjge = [sum(pJ[j, a, x, b, y, e] * pvv[a, x, b, y]
              for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)) 
         for (j, e) in itprod(1:oJ, 1:oE)]
  pv2je = [sum(pvvje[:, :, b, y, j, e])
         for (b, y, j, e) in itprod(1:oB, 1:iB, 1:oJ, 1:oE)]
  fvvje = [pJ[j, a, x, b, y, e] / (pvgv[a, x, b, y] * pjge[j, e]) 
           for (a, x, b, y, j, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE)]

  T, W = loglb_gaussradau(m)
  obj = sum(R) + sum(pvvje[idxs...] * fvvje[idxs...]
                      for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oJ, 1:oE))

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

function CMI_poly_setup(us::UpperSetting, pin::AbstractArray, p, m::Integer; kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE]
  vars = [M..., N..., O..., P..., R...]

  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)

  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  probs = CMI_probs(us, pin, M, N, O, P)
  polypairs, polypart = gr_relent_polypairs(probs, m; kwargs...)
  idxit = itprod(1:m, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
  obj = apply_epigraph!(eqconstrs, polypairs, polypart, R, idxit) 

  return obj, ineqconstrs, eqconstrs, vars
end

function HAgE_poly_setup(us::UpperSetting, pin::AbstractArray, p, m::Integer; kwargs...)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar O[1:dE, 1:oE]
  @polyvar P[1:dA, 1:dB, 1:dE]
  @polyvar R[1:m, 1:oA, 1:iA, 1:oE]
  vars = [M..., N..., O..., P..., R...]

  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)

  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  probs = HAgE_probs(us, pin, M, N, O, P)
  polypairs, polypart = gr_relent_polypairs(probs, m; kwargs...)
  idxit = itprod(1:m, 1:oA, 1:iA, 1:oE)
  obj = apply_epigraph!(eqconstrs, polypairs, polypart, R, idxit) 

  return obj, ineqconstrs, eqconstrs, vars
end

function cs_tssos_minimal(setupf, TS, params...)
  obj, ineqconstrs, eqconstrs, vars = setupf(params...)
  poly_prob = [obj, ineqconstrs..., eqconstrs...]
  opt, sol, data = cs_tssos_first(poly_prob, vars, "min", numeq=length(eqconstrs), TS=TS)
end

# Polynomial systems of equations for behaviours
extract_params(measAs::AbstractVector{<:POVMMeasurement}, measBs::AbstractVector{<:POVMMeasurement}, state::AbstractVector) = extract_params(measAs, measBs, proj(state))
function extract_params(measAs::AbstractVector{<:POVMMeasurement}, measBs::AbstractVector{<:POVMMeasurement}, state::AbstractMatrix)
  @assert all(ispovm.(measAs))
  @assert all(ispovm.(measBs))
  iA = length(meass)
  oA = maximum([meas.odim for meas in measAs])
  iB = length(meass)
  oB = maximum([meas.odim for meas in measAs])
  d = first(size(state))

  decompA = first(measAs).matrices |> first |> eigen
  decompB = first(measBs).matrices |> first |> eigen

end

function generic_behav_square_setup(us::UpperSetting)
  @expanduppersett us
  @polyvar rM[1:dA, 1:oA, 1:iA]
  @polyvar rN[1:dB, 1:oB, 1:iB]
  @polyvar rP[1:dA, 1:dB]
  @polyvar rp[1:oA, 1:oB, 1:iA, 1:iB]
  vars = [rM..., rN..., rP...]
  params = [rp...]

  ineqconstrs = []
  M = rM .^ 2
  N = rN .^ 2
  P = rP .^ 2
  p = rp .^ 2

  eqconstrs = [sum(P) - 1]
  for kA in 1:dA, x in 1:iA
    push!(eqconstrs, sum(M[kA, :, x]) - 1)
  end
  for kB in 1:dB, y in 1:iB
    push!(eqconstrs, sum(N[kB, :, y]) - 1)
  end
  for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)
    constr = 0
    for (kA, kB) in itprod(1:dA, 1:dB)
      constr += M[kA, a, x] * N[kB, b, y] * sum(P[kA, kB, :])
    end
    push!(eqconstrs, constr - p[a,b,x,y])
  end

  M0 = [1//sqrt(oA) for i in 1:length(M)]
  N0 = [1//sqrt(oB) for i in 1:length(N)]
  P0 = [1//sqrt(dA*dB) for i in 1:length(P)]
  p0 = [1//sqrt(oA*oB) for i in 1:length(p)]

  return ineqconstrs, eqconstrs, vars, params, [M0; N0; P0], p0
end

function generic_behav_setup(us::UpperSetting)
  @expanduppersett us
  @polyvar M[1:dA, 1:oA, 1:iA]
  @polyvar N[1:dB, 1:oB, 1:iB]
  @polyvar P[1:dA, 1:dB]
  @polyvar p[1:oA, 1:oB, 1:iA, 1:iB]
  vars = [M..., N..., P...]
  params = [p...]
  ineqconstrs = -1 .* vars

  eqconstrs = [sum(P) - 1]
  for kA in 1:dA, x in 1:iA
    push!(eqconstrs, sum(M[kA, :, x]) - 1)
  end
  for kB in 1:dB, y in 1:iB
    push!(eqconstrs, sum(N[kB, :, y]) - 1)
  end
  for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)
    constr = 0
    for (kA, kB) in itprod(1:dA, 1:dB)
      constr += M[kA, a, x] * N[kB, b, y] * sum(P[kA, kB, :])
    end
    push!(eqconstrs, constr - p[a,b,x,y])
  end

  M0 = [1//(oA) for i in 1:length(M)]
  N0 = [1//(oB) for i in 1:length(N)]
  P0 = [1//(dA*dB) for i in 1:length(P)]
  p0 = [1//(oA*oB) for i in 1:length(p)]

  return ineqconstrs, eqconstrs, vars, params, [M0; N0; P0], p0
end

