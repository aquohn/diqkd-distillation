struct Setting{T}
  oA::T
  oB::T
  iA::T
  iB::T
end
Base.iterate(s::Setting) = s.oA, reverse([s.oB, s.iA, s.iB])
Base.iterate(s::Setting, state) = isempty(state) ? nothing : (pop!(state), state)

struct Correlators
  Eax
  Eby
  Eabxy
end
Base.iterate(C::Correlators) = C.Eax, reverse([C.Eby, C.Eabxy])
Base.iterate(C::Correlators, state) = isempty(state) ? nothing : (pop!(state), state)

struct Behaviour
  pax
  pby
  pabxy
end
Base.iterate(P::Behaviour) = P.pax, reverse([P.pby, P.pabxy])
Base.iterate(P::Behaviour, state) = isempty(state) ? nothing : (pop!(state), state)

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
  r
  rp
  Hdata::EntropyData
end

# TODO rewrite functions to take structs as args

kd(i,j) = (i == j) ? 1 : 0
E(M, rho) = tr(M * rho)
sigmas = [[kd(j, 3) kd(j,1)-im*kd(j,2); kd(j,1)+im*kd(j,2) -kd(j,3)] for j in 1:3]

function margps_from_jointp(pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  if isnothing(pax)
    pax = [sum(pabxy[a,:,x,1]) for a in 1:oA, x in 1:iA]
  end
  if isnothing(pby)
    pby = [sum(pabxy[:,b,1,y]) for b in 1:oB, y in 1:iB]
  end
  return pax, pby, pabxy
end

# only for binary outcomes
# let outcome 1 be associated with eigenvalue -1 or logical 0
function probs_from_corrs(Eax, Eby, Eabxy)
  invcorr = 0.25 * [-1.0 -1.0 1.0 1.0; -1.0 1.0 -1.0 1.0; 1.0 -1.0 -1.0 1.0; 1.0 1.0 1.0 1.0]

  oA, oB = 2, 2
  iA, iB = [length(Eax), length(Eby)]
  pax = Array{Float64}(undef, oA, iA)
  pby = Array{Float64}(undef, oB, iB)
  pabxy = Array{Float64}(undef, oA, oB, iA, iB)

  for x in 1:iA, y in 1:iB
    pabxy[:, :, x, y] = invcorr * [Eax[x]; Eby[y]; Eabxy[x,y]; 1]
  end
  for a in 1:oA, x in 1:iA
    pax[a, x] = sum(pabxy[a, :, x, 1])
  end
  for b in 1:oB, y in 1:iB
    pby[b, y] = sum(pabxy[:, b, 1, y])
  end

  return pax, pby, pabxy
end

function corrs_from_probs(pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  pax, pby, pabxy = margps_from_jointp(pax, pby, pabxy)
  Eax = [pax[2,x] - pax[1,x] for x in 1:iA]
  Eby = [pby[2,y] - pby[1,y] for y in 1:iB]
  Eabxy = [sum([pabxy[a,b,x,y] * ((a == b) ? 1 : -1) for a in 1:oA, b in 1:iA]) for x in 1:iA, y in 1:iB]

  return Eax, Eby, Eabxy
end

