using StaticArrays, SparseArrays
using Combinatorics

struct Setting{T <: Integer}
  iA::T
  oA::T
  iB::T
  oB::T
  function Setting(iA::Integer, oA::Integer, iB::Integer, oB::Integer)
    T = promote_type(typeof(iA), typeof(oA), typeof(iB), typeof(iB))
    new{T}(iA, oA, iB, oB)
  end
end
Base.iterate(s::Setting) = s.iA, reverse([s.oA, s.iB, s.oB])
Base.iterate(s::Setting, state) = isempty(state) ? nothing : (pop!(state), state)

function is_ns(pabxy, tol=0)
  oA, oB, iA, iB = size(pabxy)
  paxs = [[sum(pabxy[a,:,x,y]) for a in 1:oA, x in 1:iA] for y in 1:iB]
  pbys = [[sum(pabxy[:,b,x,y]) for b in 1:oB, y in 1:iB] for x in 1:iA]
  fpax, rpaxs = Iterators.peel(paxs)
  fpby, rpbys = Iterators.peel(pbys)

  Acheck = [isapprox(fpax, pax, atol=tol) for pax in rpaxs]
  Bcheck = [isapprox(fpby, pby, atol=tol) for pby in rpbys]
  if all(Acheck) || all(Bcheck)
    return true
  else
    return false
  end
end

struct Correlators{Sax, Sby, Sabxy, T <: Real}
  Eax::SArray{Sax, T}
  Eby::SArray{Sby, T}
  Eabxy::SArray{Sabxy, T}
  function Correlators(Eax::AbstractArray, Eby::AbstractArray, Eabxy::AbstractArray)
    corrarrs = [Eax, Eby, Eabxy]
    T = promote_type(eltype(Eax), eltype(Eby), eltype(Eabxy))
    if !(T <: Real)
      throw(ArgumentError("All arrays must contain real values!"))
    end
    corrsarrs = [SArray{Tuple{size(E)...}, T}(E) for E in corrarrs]
    corrsarrshapes = [typeof(corr).parameters[1] for corr in corrsarrs]
    new{corrsarrshapes..., T}(corrsarrs...)
  end
end
Base.iterate(C::Correlators) = C.Eax, reverse([C.Eby, C.Eabxy])
Base.iterate(C::Correlators, state) = isempty(state) ? nothing : (pop!(state), state)

function convexsum(ws::AbstractVector{T}, Cs::AbstractVector{Tc}) where {T <: Real, Tc <: Correlators}
  Eax = sum(ws .* [C.Eax for C in Cs])
  Eby = sum(ws .* [C.Eby for C in Cs])
  Eabxy = sum(ws .* [C.Eabxy for C in Cs])
  return Correlators(Eax, Eby, Eabxy)
end

struct Behaviour{Sax, Sby, Sabxy, T <: Real} <: AbstractArray{T, 4}
  pax::SArray{Sax, T}
  pby::SArray{Sby, T}
  pabxy::SArray{Sabxy, T}
  function Behaviour(pabxy::AbstractArray{T}) where T <: Real
    oA, oB, iA, iB = size(pabxy)
    pax = T[sum(pabxy[a,:,x,1]) for a in 1:oA, x in 1:iA]
    pby = T[sum(pabxy[:,b,1,y]) for b in 1:oB, y in 1:iB]
    parrs = [pax, pby, pabxy]
    psarrs = [SArray{Tuple{size(p)...}, T}(p) for p in parrs]
    psarrshapes = [typeof(p).parameters[1] for p in psarrs]
    new{psarrshapes..., T}(psarrs...)
  end
end
Base.iterate(P::Behaviour) = P.pax, reverse([P.pby, P.pabxy])
Base.iterate(P::Behaviour, state) = isempty(state) ? nothing : (pop!(state), state)
Base.getindex(P::Behaviour, i::Integer) = P.pabxy[i]
Base.getindex(P::Behaviour, i::Integer...) = P.pabxy[i...]
Base.size(P::Behaviour) = Base.size(P.pabxy)
Base.:(==)(P1::Behaviour, P2::Behaviour) = all(iszero.(P1.pabxy - P2.pabxy))
# TODO constructor that checks if pabxy is normalised?

function convexsum(ws::AbstractVector{T}, Cs::AbstractVector{Tp}) where {T <: Real, Tp <: Behaviour}
  pabxy = sum(ws .* [P.pabxy for P in Ps])
  return Behaviour(pabxy)
end

function Correlators(behav::Behaviour)
  pax, pby, pabxy = behav
  oA, oB, iA, iB = size(pabxy)
  Eax = [pax[2,x] - pax[1,x] for x in 1:iA]
  Eby = [pby[2,y] - pby[1,y] for y in 1:iB]
  Eabxy = [sum(pabxy[a,b,x,y] * ((a == b) ? 1 : -1) for a in 1:oA, b in 1:oB) for x in 1:iA, y in 1:iB]

  return Correlators(Eax, Eby, Eabxy)
end
function Behaviour(corrs::Correlators{Sax, Sby, Sabxy, T}) where {Sax, Sby, Sabxy, T}
  # assumes binary outcomes
  # let outcome 1 be associated with eigenvalue -1 or logical 0
  Eax, Eby, Eabxy = corrs
  invcorr = [-1 -1 1 1; -1 1 -1 1; 1 -1 -1 1; 1 1 1 1] .// 4

  oA, oB = 2, 2
  iA, iB = [length(Eax), length(Eby)]
  pabxy = Array{T}(undef, oA, oB, iA, iB)

  for x in 1:iA, y in 1:iB
    pabxy[:, :, x, y] = invcorr * [Eax[x]; Eby[y]; Eabxy[x,y]; 1]
  end
  behav = Behaviour(pabxy)
  # TODO check for normalisation
  return behav
end

maxchsh(p::Behaviour) = maxchsh(Correlators(p))
function maxchsh(C::Correlators)
  Eax, Eby, Eabxy = C
  iA, iB = length(Eax), length(Eby)
  iA, iB = length(Eax), length(Eby)
  xsit = combinations(1:iA, 2)
  ysit = combinations(1:iB, 2)

  maxS = 0
  for (xs, ys) in itprod(xsit, ysit)
    Es = [Eabxy[x, y] for x in xs, y in ys]
    Esum = sum(Es)
    for E in Es
      curr = abs(Esum - 2*E)
      if curr > maxS
        maxS = curr
      end
    end
  end

  return maxS
end

# TODO wrong
function nl_extr_count(sett::Setting)
  iA, oA, iB, oB = sett
  count = 0
  for (gx, gy) in itprod(2:iA, 2:iB)
    detAs = max(1, 2 * (iA - gx))
    detBs = max(1, 2 * (iB - gy))
    choices = max(1, ((gx-1)*(gy-1) - 1))
    count += detAs * detBs * choices * factorial(iA) * factorial(iB)
  end
  return count
end
