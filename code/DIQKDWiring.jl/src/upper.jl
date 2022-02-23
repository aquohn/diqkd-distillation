using Distributions, FastGaussQuadrature

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
  oJ::T
  function UpperSetting(sett::Setting{Ts},
      oE::Integer, dA::Integer, dB::Integer, dE::Integer, oJ::Union{Tj, Nothing} = nothing
    ) where {Ts <: Integer, Tj <: Integer}
    T = promote_type(Ts, typeof(oE), typeof(dA), typeof(dB), typeof(dE))
    if !isnothing(oJ)
      T = promote_type(Tj, T)
    else
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
      sett, oE, dA, dB, dE, oJ = us
      iA, oA, iB, oB = sett
    end
  end |> esc
end

# count number of variables
function logvar_vars(us::UpperSetting)
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

z_min_term(p1, p2, t) = z_min_num(p1) / z_min_den(p1, p2, z)
z_min_den(p1, p2, t) = p1*(1-t) + p2*t
z_min_num(p1) = p1^2  # absorbed minus sign

