using DynamicPolynomials, TSSOS

struct UpperSetting{T <: Integer}
  sett::Setting{T}
  oE::T
  dA::T
  dB::T
  dE::T
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

function logvar_vars(us::UpperSetting)
  sett, oE, dA, dB, dE = us
 iA, oA, iB, oB = sett
  oJ = oA*iA*oB*iB*oE
  pJ_V1V2E_count = oJ * oA*iA*oB*iB*oE 
  pk_count = dA * dB * dE
  corr_param_count = dA*oA*iA + dB*oB*iB + dE*oE
  z_count = oJ * iA*oA*iB*oB + oJ * oA*iA*oB*iB*oE
  tot = pJ_V1V2E_count + pk_count + corr_param_count + z_count
  return pJ_V1V2E_count, pk_count, corr_param_count, z_count, tot
end

