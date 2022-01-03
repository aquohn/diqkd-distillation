using Revise
using Printf
using LazySets, Polyhedra, Symbolics, LRSLib, Combinatorics

# TODO: optimise for cache efficiency - iterate from last to first indices

includet("helpers.jl")
includet("nonlocality.jl")
includet("keyrates.jl")
includet("maxcorr.jl")

const chshset = (2,2,2,2)
const qkdset = (2,2,3,2)

cg_length(iA, oA, iB, oB) = oA*(iA-1)*oB*(iB-1) + oA*(iA-1) + oB*(iB-1)
ld_length(iA, oA, iB, oB) = oA^iA * oB^iB
positivities(iA, oA, iB, oB) = oA * oB * iA * iB

# generating 

function expr_to_vec(::Type{T}, expr, veclen, numdict) where {T <: Real}
    avec = zeros(T, veclen)
    nums = Num.(Symbolics.get_variables(expr))
    subdict = Dict(num => 0 for num in nums)
    for num in nums
      subdict[num] = 1
      val = substitute(expr, subdict)
      seqn = numdict[num]
      avec[seqn] = val
      subdict[num] = 0
    end
    return avec
end

function full_polytope(n, i, o, polylib=LRSLib.Library())
  full_polytope(Float64, n, i, o, polylib)
end

function full_polytope(::Type{T}, n, i, o, polylib=LRSLib.Library()) where {T <: Real}
  # HalfSpace(a,b) => a \dot x \leq b
  # HyperPlane(a,b) => a \dot x = b

  otupranges = [1:o for j in 1:n]
  itupranges = [1:i for j in 1:n]
  tupranges  = vcat(otupranges, itupranges)
  vars = Symbolics.@variables P[tupranges...]

  iT = typeof(n)
  numdict = Dict{Num, iT}()
  tupseqn::iT = 0
  for tup in Iterators.product(tupranges...)
    tupseqn += 1
    numdict[P[tup...]] = tupseqn
  end

  lnormconstrs = vec([-P[tup...] for tup in Iterators.product(tupranges...)])
  unormconstrs = vec([sum([P[otup..., itup...] for otup in Iterators.product(otupranges...)]) for itup in Iterators.product(itupranges...)])
  icombs = collect(combinations(1:i, 2))
  nsconstrs = Num[]
  for p in 1:n
    currtups = deepcopy(tupranges)
    currtups[p] = 0:0
    currtups[p + n] = 0:0
    for tup in Iterators.product(currtups...)
      params = [tup...]
      for is in icombs
        constr = 0
        for oval in 1:o
          params[p] = oval
          params[n + p] = is[1]
          constr += P[params...]
          params[n + p] = is[2]
          constr -= P[params...]
        end
        push!(nsconstrs, constr)
      end
    end
  end

  ineqconstrs = [Polyhedra.HalfSpace(expr_to_vec(T, constr, tupseqn, numdict), 0) for constr in lnormconstrs]
  println(ineqconstrs)
  eqconstrs = vcat([Polyhedra.HyperPlane(expr_to_vec(T, constr, tupseqn, numdict), 0) for constr in nsconstrs],
                   [Polyhedra.HyperPlane(expr_to_vec(T, constr, tupseqn, numdict), 1) for constr in unormconstrs])
  hr = hrep(eqconstrs, ineqconstrs)
  return polyhedron(hr, polylib)
end

function cg_polytope(iA, oA, iB, oB, ::Type{T} = Float64) where T <: Real
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
function ld_polytope(iA, oA, iB, oB, ::Type{T} = Float64) where T <: Real
  ld_pts = Vector{Vector{T}}()
  for params in Iterators.product((1:oA for x in 1:iA)..., (1:oB for y in 1:iB)...)
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
function cg_to_full(v::AbstractVector{T}, iA, oA, iB, oB) where {T <: Real}
  pabxy = Array{T, 4}(undef, oA, oB, iA, iB)
  pax = Array{T, 2}(undef, oA, iA)
  pby = Array{T, 2}(undef, oB, iB)

  @sliceup(v, pax_vec, (oA-1) * iA, pby_vec, (oB-1) * iB, pabxy_vec, ((oA-1) * (oB-1) * iA * iB))

  pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB] = reshape(pabxy_vec, oA-1, oB-1, iA, iB)
  pax[1:oA-1, 1:iA] = reshape(pax_vec, oA-1, iA)
  pby[1:oB-1, 1:iB] = reshape(pby_vec, oB-1, iB)

  for x in eachindex(pax[oA, :])
    pax[oA, x] = 1 - sum(pax[1:oA-1, x])
  end
  for y in eachindex(pby[oB, :])
    pby[oB, y] = 1 - sum(pby[1:oB-1, y])
  end

  for x in 1:iA, y in 1:iB
    for a in 1:(oA-1)
      pabxy[a, oB, x, y] = pax[a,x] - sum(pabxy[a, 1:(oB-1), x, y])
    end

    for b in 1:oB
      pabxy[oA, b, x, y] = pby[b,y] - sum(pabxy[1:(oA-1), b, x, y])
    end
  end

  return pax, pby, pabxy
end

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
  w = maximum([ceil(log10(d)) for d in [oA, oB, iA, iB]]) |> Int64
  cw = w + 2 + w + 1
  corner = "a, b/x, y|"
  fw = max(length(corner), cw) |> Int64
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

function print_cg_constr(constr, iA, oA, iB, oB)
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
function cg_debug(behavs, iA, oA, iB, oB)
  for behav in behavs
    fullbehav = cg_to_full(behav, iA, oA, iB, oB)
    pax, pby, pabxy = fullbehav
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
    probs = Behaviour(cg_to_full(v, qkdset...)...)
    corrs = Correlators(corrs_from_probs(probs...)...)
    Q, S, rhos = QBER(corrs...), CHSH(corrs...), maxcorrs(probs...)
    if testf(Q, S, rhos)
      println(v)
      print(@sprintf "Q = %.3f, S = %.3f, " Q S)
      println(@sprintf "rho = %.3f" maximum(rhos))
    end
  end
end

function qkd_find_lintersections(::Type{T} = Float64) where T <: Real
  qkd_v_maxmix = T[1//2, 1//2, 1//2, 1//2, 1//2, 1//4, 1//4, 1//4, 1//4, 1//4, 1//4]
  qkd_ldpoly = ld_polytope(qkdset..., T)
  qkd_ldconstr = constraints_list(qkd_ldpoly)
  qkd_poly = cg_polytope(qkdset..., T)
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
  best_qkd = cg_to_full(cg_best_qkd, qkdset...)
  maxmix_qkd = cg_to_full(cg_maxmix_qkd, qkdset...)
  boundary_qkd = cg_to_full(cg_boundary_qkd, qkdset...)

  println("PR-equivalent for QKD")
  print_full(best_qkd...)
  println("Maximally mixed")
  print_full(maxmix_qkd...)
  println("Boundary?")
  print_full(boundary_qkd...)

  qkd_ldpoly = ld_polytope(qkdset..., T)
  qkd_ldconstr = constraints(qkd_ldpoly)
  bestconstr = nothing; boundconstr = nothing
  for constr in qkd_ldconstr
    boundval = dot(constr.a, cg_boundary_qkd)
    if boundval > constr.b
      print_cg_constr(constr, qkdset...)
      boundconstr = constr
      println("Boundary value achieves $boundval)")
    end
    bestval = dot(constr.a, cg_best_qkd)
    if bestval > constr.b
      print_cg_constr(constr, qkdset...)
      bestconstr = constr
      println("Best value achieves $bestval)")
    end
  end

  for frac in 0.5:0.05:1
    v = frac .* cg_best_qkd + (1-frac) .* cg_maxmix_qkd
    probs = Behaviour(cg_to_full(v, qkdset...)...)
    corrs = Correlators(corrs_from_probs(pax, pby, pabxy)...)
    Q, S, rhos = QBER(corrs...), CHSH(corrs...), maxcorr(probs...)
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

# hrep for polytope of couplers for two binary-output boxes
function SPGcouplers_hrep(setting=(2,2,2,2), ::Type{T}=Rational{Int64}) where {T <: Real}
  poly = cg_polytope(setting..., T)
  vs = vertices_list(poly)
  hss = Vector{Polyhedra.HalfSpace{T, Vector{T}}}()
  for v in vs
    pax, pby, pabxy = cg_to_full(v, chshset...)
    hsu = Polyhedra.HalfSpace(vec(pabxy), 1)
    hsl = Polyhedra.HalfSpace(-1 .* vec(pabxy), 0)
    push!(hss, hsu)
    push!(hss, hsl)
  end
  return hrep(hss)
end

function qkdset_couplers()
  hr = SPGcouplers_hrep(qkdset)
  poly = polyhedron(hr, LRSLib.Library())
  removehredundancy!(poly)
  # hr = hrep(poly)
  # hmat = LRSLib.RepMatrix(hr)
  # LRSLib.getoutputlinset(hmat)
  # redhss = [Polyhedra.HalfSpace(LRSLib.extractrow(hmat, i)...) for i in 1:LRSLib.nhreps(hmat)]
  # redpoly = polyhedron(hrep(redhss), LRSLib.Library())
  return poly
end
# poly, hr, vr, hmat, vmat = qkdset_couplers()

const chsh_poly = cg_polytope(chshset...)
const chsh_v = vertices_list(chsh_poly)
const chsh_ldpoly = cg_polytope(chshset...)
const chsh_ldconstr = vertices_list(chsh_poly)

const qkd_poly = cg_polytope(qkdset...)
const qkd_v = vertices_list(qkd_poly)
const qkd_ldpoly = ld_polytope(qkdset...)
const qkd_ldconstr = constraints_list(qkd_ldpoly)

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