using Revise
using Printf
using LazySets, Polyhedra, Symbolics, CDDLib, LRSLib

# TODO: optimise for cache efficiency - iterate from last to first indices

includet("helpers.jl")
includet("maxcorr.jl")
includet("keyrates.jl")

cg_length(iA, oA, iB, oB) = oA*(iA-1)*oB*(iB-1) + oA*(iA-1) + oB*(iB-1)
ld_length(iA, oA, iB, oB) = oA^iA * oB^iB
positivities(iA, oA, iB, oB) = oA * oB * iA * iB

function full_polytope(iA, oA, iB, oB, ::Type{T} = Float64) where T <: Real
  vars = Symbolics.@variables pax[1:oA, 1:iA], pby[1:oB, 1:iB], pabxy[1:oA, 1:oB, 1:iA, 1:iB]
  allvars = vcat([vec(ps) for ps in vars]...)

  atob_nsconstr = [sum(pabxy[:, b, x, y] |> collect) <= pby[b, y] for b in 1:oB for x in 1:iA for y in 1:iB]
  btoa_nsconstr = [sum(pabxy[a, :, x, y] |> collect) <= pax[a, x] for a in 1:oA for x in 1:iA for y in 1:iB]

  lnormconstr = [0 <= v for v in allvars]
  unormconstr = vcat([sum(pax[:, x] |> collect) <= 1 for x in 1:iA],
                     [sum(pby[:, y] |> collect) <= 1 for y in 1:iB],
                     [sum(pabxy[:, :, x, y] |> collect) <= 1 for x in 1:iA for y in 1:iB])

  allconstrs = vcat(atob_nsconstr, btoa_nsconstr, unormconstr, lnormconstr)
  return HPolytope(allconstrs, allvars) |> remove_redundant_constraints
end

function cg_polytope(iA, oA, iB, oB, ::Type{T} = Float64) where T <: Real
  vars = Symbolics.@variables pax[1:oA-1, 1:iA], pby[1:oB-1, 1:iB], pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB]
  allvars = vcat([vec(ps) for ps in vars]...)

  atob_nsconstr = [sum(pabxy[:, b, x, y] |> collect) <= pby[b, y] for b in 1:oB-1 for x in 1:iA for y in 1:iB]
  btoa_nsconstr = [sum(pabxy[a, :, x, y] |> collect) <= pax[a, x] for a in 1:oA-1 for x in 1:iA for y in 1:iB]

  lnormconstr = vcat([0 <= v for v in allvars],
                     [0 <= 1 - sum(pax[:, x] |> collect) - sum(pby[:, y] |> collect) + sum(pabxy[:, :, x, y] |> collect) for x in 1:iA for y in 1:iB]) 
  unormconstr = vcat([sum(pax[:, x] |> collect) <= 1 for x in 1:iA],
                     [sum(pby[:, y] |> collect) <= 1 for y in 1:iB],
                     [sum(pabxy[:, :, x, y] |> collect) <= 1 for x in 1:iA for y in 1:iB])

  allconstrs = vcat(atob_nsconstr, btoa_nsconstr, unormconstr, lnormconstr)
  return HPolytope(allconstrs, allvars) |> remove_redundant_constraints
end

# TODO inefficient but simple
function ld_polytope(iA, oA, iB, oB, ::Type{T} = Float64) where T <: Real
  ld_pts = Vector{Vector{T}}()
  for params in Iterators.product((1:oA for x in 1:iA)..., (1:oB for y in 1:iB)...)
    pabxy = zeros(oA, oB, iA, iB)
    pax = zeros(oA, iA)
    pby = zeros(oB, iB)

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

function cg_to_full(v, iA, oA, iB, oB)
  pabxy = fill(NaN, oA, oB, iA, iB)
  pax = fill(NaN, oA, iA)
  pby = fill(NaN, oB, iB)

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

function print_cg_constr(v, c, iA, oA, iB, oB)
  pabxy = fill(NaN, oA, oB, iA, iB)
  pax = fill(NaN, oA, iA)
  pby = fill(NaN, oB, iB)

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

function correlator(pab)
  oA, oB = size(pab)
  corr = 0
  for a in 1:oA, b in 1:oB
    corr += (a == b ? 1 : -1) * pab[a,b]
  end
  return corr
end

function cg_chsh(v)
  iA, oA, iB, oB = 2, 2, 2, 2
  pax, pby, pabxy = cg_to_full(v, iA, oA, iB, oB)

  sum([correlator(pabxy[:,:,x,y]) for x in 1:iA for y in 1:iB] .* [1, 1, 1, -1])
end

function cg_QSrho(v, iA, oA, iB, oB)
  pax, pby, pabxy = cg_to_full(v, iA, oA, iB, oB)
  Eabxy = [correlator(pabxy[:,:,x,y]) for x in 1:iA, y in 1:iB]
  Q = (1 - Eabxy[1,3]) / 2 # QBER H(A|B)
  S = Eabxy[1,1] + Eabxy[1,2] + Eabxy[2,1] - Eabxy[2,2]
  rhos = maxcorrs(pax, pby, pabxy)
  return Q, S, rhos
end

function qkd_analysis(vs)
  iA, oA, iB, oB = 2, 2, 3, 2
  for v in vs
    Q, S, rhos = cg_QSrho(v, iA, oA, iB, oB)
    if (abs(0.5 - Q) == 0.5 && abs(S) == 4)
      println(v)
      print(@sprintf "Q = %.3f, S = %.3f, " Q S)
      println(@sprintf "rho = %.3f" maximum(rhos))
    end
  end
end

function qkd_iso(::Type{T} = Float64) where T <: Real
  qkdset = (2,2,3,2)

  cg_best_qkd = T[1//2, 1//2, 1//2, 1//2, 1//2, 1//2, 1//2, 1//2, 0//1, 1//2, 1//2]
  cg_maxmix_qkd = T[1//2, 1//2, 1//2, 1//2, 1//2, 1//4, 1//4, 1//4, 1//4, 1//4, 1//4]
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
      print_cg_constr(constr.a, constr.b, qkdset...)
      boundconstr = constr
      println("Boundary value achieves $boundval)")
    end
    bestval = dot(constr.a, cg_best_qkd)
    if bestval > constr.b
      print_cg_constr(constr.a, constr.b, qkdset...)
      bestconstr = constr
      println("Best value achieves $bestval)")
    end
  end

  for frac in 0.5:0.05:1
    v = frac .* cg_best_qkd + (1-frac) .* cg_maxmix_qkd
    Q, S, rhos = cg_QSrho(v, qkdset...)
    print(@sprintf "Q = %.3f, S = %.3f, " Q S)
    println(@sprintf "rho = %.3f" maximum(rhos))
  end

  for v in vertices_list(qkd_ldpoly)
    val = dot(bestconstr.a, v) - bestconstr.b
    println("$v: $val")
  end

  # TODO find intersection numerically
end

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

function find_couplers(iA, oA, iB, oB, n)
  poly = cg_polytope(iA, oA, iB, oB)
  vs = vertices_list(poly)
  l = length(vs) 

  # compute coupler for each party independently
  bsysdims = ((oB for i in 1:n)..., (iB for i in 1:n)...)
  bsysranges = (1:d for d in bsysdims)
  vars = Symbolics.@variables C[1:oB-1, bsysranges...]
  allvars = vcat([vec(arr) for arr in vars]...)

  unormconstrs = Vector{Num}(undef, l * oB)
  lnormconstrs = Vector{Num}(undef, l * oB)

  total_pbp = 0
  for idx in eachindex(vs)
    v = vs[idx]
    pax, pby, pabxy = cg_to_full(v, iA, oA, iB, oB)
    pby = pbys[idx]
    constridx = (idx - 1) * oB
    iter = Iterators.product(bsysranges...)
    pbsys = fill(Num(1), bsysdims)
    for bys in iter # tabulate p(bs|ys) 
      # TODO use commutativity to tabulate more efficiently
      for i in 1:n # find individual p(b|y) and accumulate
        b = bys[i]
        y = bys[i+n]
        pbsys[bys...] *= pby[b,y]
      end
    end

    iter = Iterators.product(bsysranges...)
    total_pbp = 0
    for bp in 1:oB-1 # compute the probability of an overall output bp
      pbp = 0
      for bys in iter
        pbp += C[bp, bys...] * pbsys[bys...]
      end

      unormconstrs[constridx + bp] = pbp <= 1
      lnormconstrs[constridx + bp] = 0 <= pbp
      total_pbp += pbp
    end
    unormconstrs[constridx + oB] = total_pbp <= 1
    lnormconstrs[constridx + oB] = 0 <= total_pbp
  end

  allconstrs = vcat(unormconstrs, lnormconstrs)
  return HPolytope(allconstrs, allvars) |> remove_redundant_constraints
end

function cg_bobcouplers(vs, iA, oA, iB, oB, n)
  ps = cg_to_full.(vs, iA, oA, iB, oB)
  pbys = [p[2] for p in ps]
  return find_couplers(pbys, n)
end

function cg_alicecouplers(vs, iA, oA, iB, oB, n)
  ps = cg_to_full.(vs, iA, oA, iB, oB)
  pbys = [p[3] for p in ps]
  return find_couplers(pbys, n)
end

chshset = (2,2,2,2)
chsh_poly = cg_polytope(chshset...)
chsh_v = vertices_list(chsh_poly)

qkdset = (2,2,3,2)
qkd_poly = cg_polytope(qkdset...)
qkd_v = vertices_list(qkd_poly)
qkd_ldpoly = ld_polytope(qkdset...)
qkd_ldconstr = constraints(qkd_ldpoly)
