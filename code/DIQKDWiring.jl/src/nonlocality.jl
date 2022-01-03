using StaticArrays

struct Setting{T <: Integer}
  oA::T
  oB::T
  iA::T
  iB::T
end
Base.iterate(s::Setting) = s.oA, reverse([s.oB, s.iA, s.iB])
Base.iterate(s::Setting, state) = isempty(state) ? nothing : (pop!(state), state)

struct Correlators{Sax, Sby, Sabxy, T <: Real}
  Eax::SArray{Sax, T}
  Eby::SArray{Sby, T}
  Eabxy::SArray{Sabxy, T}
  function Correlators(Eax::AbstractArray, Eby::AbstractArray, Eabxy::AbstractArray)
    corrarrs = [Eax, Eby, Eabxy]
    T = promote_type(eltype.(corrarrs)...)
    if !(T <: Real)
      throw(ArgumentError("All arrays must contain real values!"))
    end
    corrsarrs = [SArray{Tuple{size(E)...}}(E) for E in corrarrs]
    corrsarrshapes = [typeof(corr).parameters[1] for corr in corrsarrs]
    new{corrsarrshapes..., T}(corrsarrs...)
  end
end
Base.iterate(C::Correlators) = C.Eax, reverse([C.Eby, C.Eabxy])
Base.iterate(C::Correlators, state) = isempty(state) ? nothing : (pop!(state), state)

struct Behaviour{Sax, Sby, Sabxy, T <: Real}
  pax::SArray{Sax, T}
  pby::SArray{Sby, T}
  pabxy::SArray{Sabxy, T}
  function Behaviour(pabxy::AbstractArray{T}) where T <: Real
    oA, oB, iA, iB = size(pabxy)
    pax = T[sum(pabxy[a,:,x,1]) for a in 1:oA, x in 1:iA]
    pby = T[sum(pabxy[:,b,1,y]) for b in 1:oB, y in 1:iB]
    parrs = [pax, pby, pabxy]
    psarrs = [SArray{Tuple{size(p)...}}(p) for p in parrs]
    psarrshapes = [typeof(p).parameters[1] for p in psarrs]
    new{psarrshapes..., T}(psarrs...)
  end
end
Base.iterate(P::Behaviour) = P.pax, reverse([P.pby, P.pabxy])
Base.iterate(P::Behaviour, state) = isempty(state) ? nothing : (pop!(state), state)
# TODO constructor that checks if pabxy is normalised?

function Correlators(behav::Behaviour)
  pax, pby, pabxy = behav
  oA, oB, iA, iB = size(pabxy)
  Eax = [pax[2,x] - pax[1,x] for x in 1:iA]
  Eby = [pby[2,y] - pby[1,y] for y in 1:iB]
  Eabxy = [sum([pabxy[a,b,x,y] * ((a == b) ? 1 : -1) for a in 1:oA, b in 1:iA]) for x in 1:iA, y in 1:iB]

  return Correlators(Eax, Eby, Eabxy)
end
function Behaviour(corrs::Correlators)
  # assumes binary outcomes
  # let outcome 1 be associated with eigenvalue -1 or logical 0
  Eax, Eby, Eabxy = corrs
  invcorr = 0.25 * [-1.0 -1.0 1.0 1.0; -1.0 1.0 -1.0 1.0; 1.0 -1.0 -1.0 1.0; 1.0 1.0 1.0 1.0]

  oA, oB = 2, 2
  iA, iB = [length(Eax), length(Eby)]
  pabxy = Array{Float64}(undef, oA, oB, iA, iB)

  for x in 1:iA, y in 1:iB
    pabxy[:, :, x, y] = invcorr * [Eax[x]; Eby[y]; Eabxy[x,y]; 1]
  end
  behav = Behaviour(pabxy)
  # TODO check for normalisation
  return behav
end


struct Wiring
  CA
  CAj
  CB
  CBj
end
Base.iterate(W::Wiring) = W.CA, reverse([W.CAj, W.CB, W.CBj])
Base.iterate(W::Wiring, state) = isempty(state) ? nothing : (pop!(state), state)

struct EntropyData
  HAE
  HAB
  HAEp
  HABp
end

struct WiringData
  wiring::Wiring
  r::Real
  rp::Real
  Hdata # object holding additional details about the computed rates
end

# TODO rewrite functions to take structs as args

kd(i,j) = (i == j) ? 1 : 0
E(M, rho) = tr(M * rho)
sigmas = [[kd(j, 3) kd(j,1)-im*kd(j,2); kd(j,1)+im*kd(j,2) -kd(j,3)] for j in 1:3]

