using Revise
using Printf
using LazySets, Polyhedra, Symbolics, LRSLib, Combinatorics, SparseArrays

# TODO: optimise for cache efficiency - iterate from last to first indices
# TODO remove lazysets

includet("helpers.jl")
includet("nonlocality.jl")
includet("keyrates.jl")
includet("maxcorr.jl")

const chshsett = Setting(2,2,2,2)
const qkdsett = Setting(2,2,3,2)

cg_length(s::Setting) = s.oA*(s.iA-1)*s.oB*(s.iB-1) + s.oA*(s.iA-1) + s.oB*(s.iB-1)
ld_length(s::Setting) = s.oA^s.iA * s.oB^s.iB
positivities(s::Setting) = s.oA * s.oB * s.iA * s.iB

# generating 

# create a lookup table for translating function arguments (args) to vectors
# allows a function of integers (really a tensor of indices) to be represented
# as a vector
function func_vec(::Type{T}, ::Type{Ti}, argranges) where {T <: Real, Ti <: Integer}
  shape = [rang.stop for rang in argranges]
  nargvals = Ti(prod(shape))
  F = Array{SparseVector{T, Ti}}(undef, shape...)
  idx = 1
  for arg in itprod(argranges...)
    F[arg...] = sparsevec([idx], 1, nargvals)
    idx += 1
  end
  return F, nargvals
end

# LRSLib seems quite slow; DefaultLibrary{T}() may work faster
full_polytope(sett::Setting, polylib=DefaultLibrary{Rational{Int}}()) = full_polytope(2, [sett.oA, sett.oB], [sett.iA, sett.iB], polylib)
full_polytope(n::Integer, o::Integer, i::Integer, polylib=DefaultLibrary{Rational{Int}}()) = full_polytope(n, repeat([o], n), repeat([i], n), polylib)
full_polytope(::Type{T}, n::Integer, o::Integer, i::Integer, polylib=DefaultLibrary{T}()) where {T} = full_polytope(T, n, repeat([o], n), repeat([i], n), polylib)
full_polytope(n, os, is, polylib=DefaultLibrary{Rational{Int}}()) = full_polytope(Rational{Int}, n, os, is, polylib)
function full_polytope(::Type{T}, n::Integer, os::AbstractVector{To}, is::AbstractVector{Ti}, polylib=DefaultLibrary{T}()) where {T <: Real, Ti <: Integer, To <: Integer}
  # HalfSpace(a,b) => a \dot x \leq b
  # HyperPlane(a,b) => a \dot x = b

  Tn = promote_type(Ti, To, typeof(n))
  # ranges for the tuples of indices specifying a probability
  itupranges = [1:i for i in is]
  otupranges = [1:o for o in os]
  tupranges  = vcat(otupranges, itupranges)
  P, ntups = func_vec(T, Tn, tupranges)
  SV = SparseVector{T, Tn}

  lnormconstrs = vec([-P[tup...] for tup in itprod(tupranges...)])
  unormconstrs = vec([sum(P[otup..., itup...] for otup in itprod(otupranges...)) for itup in itprod(itupranges...)])
  nsconstrs = SV[]
  for p in 1:n  # create ns constrs for player p
    # iterate over everyone else's indices
    currtups = deepcopy(tupranges)
    currtups[p] = 0:0
    currtups[n + p] = 0:0
    o = os[p]; i = is[p]
    icombs = collect(combinations(1:i, 2))

    # choose two different inputs for player p and sum over all his outputs
    # the resulting sum should be the same for all choices
    for tup in itprod(currtups...)
      params = [tup...]
      for (i1, i2) in icombs
        constr = spzeros(T, ntups)
        for oval in 1:o
          params[p] = oval
          params[n + p] = i1
          constr .+= P[params...]
          params[n + p] = i2
          constr .-= P[params...]
        end
        push!(nsconstrs, constr)
      end
    end
  end

  ineqconstrs = [Polyhedra.HalfSpace(constr, 0) for constr in lnormconstrs]
  eqconstrs = vcat([Polyhedra.HyperPlane(constr, 0) for constr in nsconstrs],
                   [Polyhedra.HyperPlane(constr, 1) for constr in unormconstrs])
  hr = hrep(eqconstrs, ineqconstrs)
  return polyhedron(hr, polylib)
end

function cg_polytope(sett::Setting, ::Type{T} = Float64) where T <: Real
  oA, oB, iA, iB = sett
  vars = Symbolics.@variables pax[1:oA-1, 1:iA], pby[1:oB-1, 1:iB], pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB]
  allvars = vcat([vec(ps) for ps in vars]...)

  atob_nsconstr = [sum(pabxy[:, b, x, y] |> collect) <= pby[b, y] for b in 1:oB-1 for x in 1:iA for y in 1:iB]
  btoa_nsconstr = [sum(pabxy[a, :, x, y] |> collect) <= pax[a, x] for a in 1:oA-1 for x in 1:iA for y in 1:iB]

  lnormconstr = vcat([T(0) <= v for v in allvars],
                     [T(0) <= T(1) - sum(pax[:, x] |> collect) - sum(pby[:, y] |> collect) + sum(pabxy[:, :, x, y] |> collect) for x in 1:iA for y in 1:iB]) 
  unormconstr = vcat([sum(pax[:, x] |> collect) <= T(1) for x in 1:iA],
                     [sum(pby[:, y] |> collect) <= T(1) for y in 1:iB],
                     [sum(pabxy[:, :, x, y] |> collect) <= T(1) for x in 1:iA for y in 1:iB])

  allconstrs = vcat(atob_nsconstr, btoa_nsconstr, unormconstr, lnormconstr)
  return HPolytope(allconstrs, allvars, N=T) |> remove_redundant_constraints
end

# TODO inefficient but simple
function ld_polytope(sett::Setting, ::Type{T} = Float64) where T <: Real
  oA, oB, iA, iB = sett
  ld_pts = Vector{Vector{T}}()
  for params in itprod((1:oA for x in 1:iA)..., (1:oB for y in 1:iB)...)
    pabxy = zeros(T, oA, oB, iA, iB)
    pax = zeros(T, oA, iA)
    pby = zeros(T, oB, iB)

    @sliceup(params, as, iA, bs, iB)
    for x in eachindex(as)
      pax[as[x], x] = 1
    end
    for y in eachindex(bs)
      pby[bs[y], y] = 1
    end

    for x in 1:iA, y in 1:iB
      pabxy[as[x], bs[y], x, y] = 1
    end

    push!(ld_pts, full_to_cg(pax, pby, pabxy))
  end

  return VPolytope(ld_pts)
end

# converting
function cg_to_full(v::AbstractVector{T}, sett::Setting) where {T <: Real}
  oA, oB, iA, iB = sett
  pabxy = Array{T, 4}(undef, oA, oB, iA, iB)
  pax = Array{T, 2}(undef, oA, iA)
  pby = Array{T, 2}(undef, oB, iB)

  @sliceup(v, pax_vec, (oA-1) * iA, pby_vec, (oB-1) * iB, pabxy_vec, ((oA-1) * (oB-1) * iA * iB))

  pax[1:oA-1, 1:iA] = reshape(pax_vec, oA-1, iA)
  pby[1:oB-1, 1:iB] = reshape(pby_vec, oB-1, iB)

  for x in eachindex(pax[oA, :])
    pax[oA, x] = 1 - sum(pax[1:oA-1, x])
  end
  for y in eachindex(pby[oB, :])
    pby[oB, y] = 1 - sum(pby[1:oB-1, y])
  end

  pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB] = reshape(pabxy_vec, oA-1, oB-1, iA, iB)
  for x in 1:iA, y in 1:iB
    for a in 1:(oA-1)
      pabxy[a, oB, x, y] = pax[a,x] - sum(pabxy[a, 1:(oB-1), x, y])
    end

    for b in 1:oB
      pabxy[oA, b, x, y] = pby[b,y] - sum(pabxy[1:(oA-1), b, x, y])
    end
  end

  return Behaviour(pabxy)
end

full_to_cg(p::Behaviour) = full_to_cg(p.pax, p.pby, p.pabxy)
function full_to_cg(pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  pax_vec = vec(pax[1:oA-1, 1:iA])
  pby_vec = vec(pby[1:oB-1, 1:iB])
  pabxy_vec = vec(pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB])
  return vcat(pax_vec, pby_vec, pabxy_vec)
end

# printing
function print_full(pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  w = maximum([ceil(log10(d)) for d in [oA, oB, iA, iB]]) |> Int
  cw = w + 2 + w + 1
  corner = "a, b/x, y|"
  fw = max(length(corner), cw) |> Int
  l = fw + (iA * iB) * cw

  # P(w, w|w, w) -> 8 + 4w chars
  lpad(corner, fw) |> print
  for x in 1:iA, y in 1:iB
    print(lpad(string(x), w) * ", " * lpad(string(y), w) * "|")
  end
  println("\n" * "-" ^ l)
  for a in 1:oA, b in 1:oB
    print(rpad(lpad(string(a), w) * ", " * lpad(string(b), w), fw-1) * "|")
    for x in 1:iA, y in 1:iB
      pstr = (round(pabxy[a,b,x,y], sigdigits=cw-2-1) |> string) * "|"
      print(lpad(pstr, cw))
    end
    println("\n" * "-" ^ l)
  end
end

function print_cg_constr(constr, sett::Setting)
  oA, oB, iA, iB = sett
  pabxy = fill(NaN, oA, oB, iA, iB)
  pax = fill(NaN, oA, iA)
  pby = fill(NaN, oB, iB)

  v, c = constr.a, constr.b
  @sliceup(v, pax_vec, (oA-1) * iA, pby_vec, (oB-1) * iB, pabxy_vec, ((oA-1) * (oB-1) * iA * iB))

  pax[1:oA-1, 1:iA] = reshape(pax_vec, oA-1, iA)
  pby[1:oB-1, 1:iB] = reshape(pby_vec, oB-1, iB)
  pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB] = reshape(pabxy_vec, oA-1, oB-1, iA, iB)

  for a in 1:oA-1, x in 1:iA
    if pax[a,x] != 0
      print(@sprintf "%+.3f P(a = %u|x = %u) " pax[a,x] a x)
    end
  end
  for b in 1:oB-1, y in 1:iB
    if pby[b,y] != 0
      print(@sprintf "%+.3f P(b = %u|y = %u) " pby[b,y] b y)
    end
  end
  for a in 1:oA-1, b in 1:oB-1, x in 1:iA, y in 1:iB
    if pabxy[a,b,x,y] != 0
      print(@sprintf "%+.3f P(%u,%u|%u,%u) " pabxy[a,b,x,y] a b x y)
    end
  end
  println(@sprintf "<= %.3f" c)
end

# debugging
function cg_debug(behavs, sett::Setting)
  oA, oB, iA, iB = sett
  for behav in behavs
    fullbehav = cg_to_full(behav, sett)
    pabxy = fullbehav.pabxy
    if !(all([all(ps .>= 0) && all(ps .<= 1) for ps in fullbehav])
         && all([sum(pabxy[:, b, x, y]) == pby[b, y] for b in 1:oB for x in 1:iA for y in 1:iB])
         && all([sum(pabxy[a, :, x, y]) == pax[a, x] for a in 1:oA for x in 1:iA for y in 1:iB])
         && all([sum(pabxy[:, :, x, y]) == 1 for x in 1:iA for y in 1:iB]))
      println(behav); println(fullbehav); println()
    end
  end
end

# analysis
function find_overlapping_constrs(constrs1, constrs2)
  T1, T2 = eltype(constrs1), eltype(constrs2)
  overlaps, rem1 = T2[], T1[]
  for constr1 in constrs1
    matchconstr = nothing
    for constr2 in constrs2
      if issubset(constr1, constr2) || issubset(constr2, constr1)
        matchconstr = constr2
        break
      end
    end;
    if !isnothing(matchconstr)
      push!(overlaps, matchconstr)
    else
      push!(rem1, constr1)
    end
  end
  rem2 = [constr for constr in filter(c -> !(c in overlaps), constrs2)]
  return overlaps, rem1, rem2
end

function qkd_analysis(vs, testf = (Q, S, rho) -> abs(S) == 4)
  for v in vs
    probs = cg_to_full(v, qkdsett)
    corrs = corrs_from_probs(probs)
    Q, S, rhos = QBER(corrs), CHSH(corrs), maxcorrs(probs)
    if testf(Q, S, rhos)
      println(v)
      print(@sprintf "Q = %.3f, S = %.3f, " Q S)
      println(@sprintf "rho = %.3f" maximum(rhos))
    end
  end
end

function qkd_find_lintersections(::Type{T} = Float64) where T <: Real
  qkd_v_maxmix = T[1//2, 1//2, 1//2, 1//2, 1//2, 1//4, 1//4, 1//4, 1//4, 1//4, 1//4]
  qkd_ldpoly = ld_polytope(qkdsett, T)
  qkd_ldconstr = constraints_list(qkd_ldpoly)
  qkd_poly = cg_polytope(qkdsett, T)
  qkd_v = vertices_list(qkd_poly)
  CT, VT = eltype(qkd_ldconstr), eltype(qkd_v)
  constrs_to_extrv = Dict{CT, Vector{Tuple{VT, T}}}()

  ns_extremal = filter(v -> !(v in qkd_ldpoly), qkd_v)
  for constr in qkd_ldconstr
    matches = Tuple{VT, T}[]
    for v in ns_extremal
      Symbolics.@variables q
      vb = q .* v + (1-q) .* qkd_v_maxmix
      boundval = dot(constr.a, vb)
      qval = T(Symbolics.solve_for(boundval ~ constr.b, q))
      if 0 <= qval <= 1
        vbval = qval .* v + qval .* qkd_v_maxmix
        push!(matches, (v, qval))
      end
    end
    constrs_to_extrv[constr] = matches
  end

  return constrs_to_extrv
end

function qkd_find_prs(::Type{T} = Float64) where T <: Real
  linters = qkd_find_lintersections(T)
  nlinters = typeof(linters)()
  for (facet, boundvs) in linters
    nlboundvs = filter(v -> !(v[1] in qkd_ldpoly) && isapprox(0.5, v[2]), boundvs)
    if !isempty(nlboundvs)
      nlinters[facet] = nlboundvs
    end
  end
  return nlinters
end

function qkd_iso(::Type{T} = Float64) where T <: Real
  cg_maxmix_qkd = T[1//2, 1//2, 1//2, 1//2, 1//2, 1//4, 1//4, 1//4, 1//4, 1//4, 1//4]

  cg_best_qkd = T[1//2, 1//2, 1//2, 1//2, 1//2, 1//2, 1//2, 1//2, 0//1, 1//2, 1//2]
  cg_boundary_qkd = 1//2 .* cg_best_qkd + 1//2 .* cg_maxmix_qkd
  best_qkd = cg_to_full(cg_best_qkd, qkdsettg)
  maxmix_qkd = cg_to_full(cg_maxmix_qkd, qkdsettg)
  boundary_qkd = cg_to_full(cg_boundary_qkd, qkdsettg)

  println("PR-equivalent for QKD")
  print_full(best_qkd...)
  println("Maximally mixed")
  print_full(maxmix_qkd...)
  println("Boundary?")
  print_full(boundary_qkd...)

  qkd_ldpoly = ld_polytope(qkdsett, T)
  qkd_ldconstr = constraints(qkd_ldpoly)
  bestconstr = nothing; boundconstr = nothing
  for constr in qkd_ldconstr
    boundval = dot(constr.a, cg_boundary_qkd)
    if boundval > constr.b
      print_cg_constr(constr, qkdsett)
      boundconstr = constr
      println("Boundary value achieves $boundval)")
    end
    bestval = dot(constr.a, cg_best_qkd)
    if bestval > constr.b
      print_cg_constr(constr, qkdsett)
      bestconstr = constr
      println("Best value achieves $bestval)")
    end
  end

  for frac in 0.5:0.05:1
    v = frac .* cg_best_qkd + (1-frac) .* cg_maxmix_qkd
    probs = cg_to_full(v, qkdsett)
    corrs = corrs_from_probs(probs)
    Q, S, rhos = QBER(corrs), CHSH(corrs), maxcorr(probs)
    print(@sprintf "Q = %.3f, S = %.3f, " Q S)
    println(@sprintf "rho = %.3f" maximum(rhos))
  end

  for v in vertices_list(qkd_ldpoly)
    val = dot(bestconstr.a, v) - bestconstr.b
    println("$v: $val")
  end

  # TODO find intersection numerically
end

# Couplers
# "Entaglement swapping for generalized non-local correlations"; Short, Popescu
# and Gisin

function general_couplers_sparse(::Type{T}, c::Integer, o::Integer, i::Integer, polylib=DefaultLibrary{T}()) where {T <: Real}
  tupranges = [1:o-1, repeat([1:o], c)..., repeat([1:i], c)...]
  Ti = promote_type(typeof(c), typeof(o), typeof(i))
  chi, nidxs = func_vec(T, Ti, tupranges)
  SV = SparseVector{T, Ti}
  allbs_iter = itprod(repeat([itprod(repeat([1:o], c)...)], i^c)...)
  ys_iter = itprod(repeat([1:i], c)...)

  lnormconstrs = SV[]
  unormconstrs = SV[]
  for allbs in allbs_iter
    ps = SV[]
    allbs_arr = reshape([allbs...], tuple(repeat([i], c)...))
    for bp in 1:o-1
      p = spzeros(T, nidxs)
      for ys in ys_iter
        bs = allbs_arr[ys...]
        p += chi[bp, bs..., ys...]
      end
      push!(ps, p)
      push!(lnormconstrs, -p)
    end
    push!(unormconstrs, sum(ps))
  end

  ineqconstrs = vcat([Polyhedra.HalfSpace(constr, 0) for constr in lnormconstrs], [Polyhedra.HalfSpace(constr, 1) for constr in unormconstrs])
  hr = hrep(ineqconstrs)
  poly = polyhedron(hr, polylib)
  vr = vrep(poly)
  return vr
end
general_couplers_sparse(c, o, i, polylib=DefaultLibrary{Rational{Int}}()) = general_couplers_sparse(Rational{Int}, c, o, i, polylib)

function couplers_hrep(::Type{T}, c::Integer, o::Integer, i::Integer, behavs) where {T <: Real}
  argranges = [1:o-1, repeat([1:o], c)..., repeat([1:i], c)...]
  Ti = promote_type(typeof(c), typeof(o), typeof(i))
  chi, nidxvals = func_vec(T, Ti, argranges)
  SV = SparseVector{T, Ti}
  bs_iter = itprod(repeat([1:o], c)...)
  ys_iter = itprod(repeat([1:i], c)...)

  lnormconstrs = SV[]
  unormconstrs = SV[]
  for behav in behavs
    ps = SV[]
    for bp in 1:o-1
      p = spzeros(T, nidxvals)
      for (bs, ys) in itprod(bs_iter, ys_iter)
        p += chi[bp, bs..., ys...] * behav[bs..., ys...]
      end
      push!(ps, p)
      push!(lnormconstrs, -p)
    end
    push!(unormconstrs, sum(ps))
  end

  ineqconstrs = vcat([Polyhedra.HalfSpace(constr, 0) for constr in lnormconstrs], [Polyhedra.HalfSpace(constr, 1) for constr in unormconstrs])
  return hrep(ineqconstrs)
end
couplers_hrep(c::Integer, o::Integer, i::Integer, behavs) = couplers_hrep(Rational{Int64}, c, o, i, behavs)
function couplers_poly(::Type{T}, c::Integer, o::Integer, i::Integer, behavs, polylib=DefaultLibrary{T}()) where {T <: Real}
  hr = couplers_hrep(T, c, o, i, behav)
  poly = polyhedron(hr, polylib)
  vrep(poly)
  return poly
end
couplers_poly(c::Integer, o::Integer, i::Integer, behavs, polylib=DefaultLibrary{Int}()) = couplers_poly(Rational{Int}, c, o, i, behavs, polylib)

struct GeneralExtrBehaviourIter{T <: Integer}
  n::T
  os::Vector{T}
  is::Vector{T}
  _it
  _inputit
  function GeneralExtrBehaviourIter(os::AbstractVector{<:Integer}, is::AbstractVector{<:Integer})
    T = promote_type(eltype(os), eltype(is))
    @assert length(os) == length(is)
    n = length(os)
    _it = itprod(repeat([itprod([1:o for o in os]...)], prod(is))...)
    _inputit = itprod([1:i for i in is]...)
    new{T}(n, os, is, _it, _inputit)
  end
end
Base.length(gebit::GeneralExtrBehaviourIter) = length(gebit._it)
Base.iterate(gebit::GeneralExtrBehaviourIter) = _iterate(gebit, iterate(gebit._it))
Base.iterate(gebit::GeneralExtrBehaviourIter, state) = _iterate(gebit, iterate(gebit._it, state))
_iterate(gebit::GeneralExtrBehaviourIter, itval::Nothing) = nothing
function _iterate(gebit::GeneralExtrBehaviourIter{T}, itval) where T
  pvec, state = itval
  n, os, is = gebit.n, gebit.os, gebit.is
  p = zeros(T, os..., is...)
  allouts = reshape([pvec...], tuple(is...))
  for ins in gebit._inputit
    outs = allouts[ins...]
    p[outs..., ins...] = 1
  end
  return p, state
end

function ns_couplers_hrep(::Type{T}, c::Integer, o::Integer, i::Integer) where {T <: Real}
  poly = full_polytope(T, c, o, i)
  vs = [reshape(v, tuple([repeat([o], c); repeat([i], c)]...)) for v in points(poly)]
  return couplers_hrep(T, c, o, i, vs)
end
function general_couplers_hrep(::Type{T}, c::Integer, o::Integer, i::Integer) where {T <: Real}
  return couplers_hrep(T, c, o, i, GeneralExtrBehaviourIter(repeat([o], c), repeat([i], c)))
end

#=
const chsh_poly = cg_polytope(chshsett)
const chsh_v = vertices_list(chsh_poly)
const chsh_ldpoly = cg_polytope(chshsett)
const chsh_ldconstr = vertices_list(chsh_poly)

const qkd_poly = cg_polytope(qkdsett)
const qkd_v = vertices_list(qkd_poly)
const qkd_ldpoly = ld_polytope(qkdsett)
const qkd_ldconstr = constraints_list(qkd_ldpoly)
=#

#=
-0.200 P(a = 1|x = 1) -0.200 P(b = 1|y = 1) +0.200 P(1,1|1,1) +0.200 P(1,1|1,2) +0.200 P(1,1|2,1) -0.200 P(1,1|2,2) <= 0.000
[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
Q = 1.000, S = 4.000, rho = 1.000
[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5]
Q = 1.000, S = 4.000, rho = 1.000
a, b/x, y|1, 1|1, 2|1, 3|2, 1|2, 2|2, 3|
[0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
Q = 0.500, S = 4.000, rho = 1.000
a, b/x, y|1, 1|1, 2|1, 3|2, 1|2, 2|2, 3|
[0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
Q = 0.500, S = 4.000, rho = 1.000
a, b/x, y|1, 1|1, 2|1, 3|2, 1|2, 2|2, 3|
[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0]
Q = 0.000, S = 4.000, rho = 1.000
a, b/x, y|1, 1|1, 2|1, 3|2, 1|2, 2|2, 3|
[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5]
Q = 0.000, S = 4.000, rho = 1.000
a, b/x, y|1, 1|1, 2|1, 3|2, 1|2, 2|2, 3|
=#

# halfspace -> A
# A:               1  -1   0   1  -1   0  -1   0   0   0  -1   0   0   0  -1   0   0
# halfspace:       1   0  -1   1   0   1   0   0   0   1   0   0   0   1   0   0   1
# halfspace * -1: -1   0   1  -1   0  -1   0   0   0  -1   0   0   0  -1   0   0  -1
# move constant over to get A
