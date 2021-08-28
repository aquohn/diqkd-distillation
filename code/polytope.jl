using LazySets, Polyhedra, Symbolics, CDDLib, LRSLib

# TODO: optimise for cache efficiency - iterate from last to first indices

function full_polytope(iA, oA, iB, oB)
  vars = @variables pabxy[1:oA, 1:oB, 1:iA, 1:iB], pax[1:oA, 1:iA], pby[1:oB, 1:iB]
  allvars = vcat([vec(ps) for ps in vars]...)

  atob_nsconstr = [sum(pabxy[:, b, x, y] |> collect) <= pby[b, y] for b in 1:oB for x in 1:iA for y in 1:iB]
  btoa_nsconstr = [sum(pabxy[a, :, x, y] |> collect) <= pax[a, x] for a in 1:oA for x in 1:iA for y in 1:iB]

  lnormconstr = [0 <= v for v in allvars]
  unormconstr = vcat([sum(pabxy[:, :, x, y] |> collect) <= 1 for x in 1:iA for y in 1:iB], 
                     [sum(pax[:, x] |> collect) <= 1 for x in 1:iA],
                     [sum(pby[:, y] |> collect) <= 1 for y in 1:iB])

  allconstrs = vcat(atob_nsconstr, btoa_nsconstr, unormconstr, lnormconstr)
  return HPolytope(allconstrs, allvars) |> remove_redundant_constraints
end

function cg_polytope(iA, oA, iB, oB)
  vars = @variables pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB], pax[1:oA-1, 1:iA], pby[1:oB-1, 1:iB]
  allvars = vcat([vec(ps) for ps in vars]...)

  atob_nsconstr = [sum(pabxy[:, b, x, y] |> collect) <= pby[b, y] for b in 1:oB-1 for x in 1:iA for y in 1:iB]
  btoa_nsconstr = [sum(pabxy[a, :, x, y] |> collect) <= pax[a, x] for a in 1:oA-1 for x in 1:iA for y in 1:iB]

  lnormconstr = vcat([0 <= v for v in allvars],
                     [0 <= 1 - sum(pax[:, x] |> collect) - sum(pby[:, y] |> collect) + sum(pabxy[:, :, x, y] |> collect) for x in 1:iA for y in 1:iB]) 
  unormconstr = vcat([sum(pabxy[:, :, x, y] |> collect) <= 1 for x in 1:iA for y in 1:iB], 
                     [sum(pax[:, x] |> collect) <= 1 for x in 1:iA],
                     [sum(pby[:, y] |> collect) <= 1 for y in 1:iB])

  allconstrs = vcat(atob_nsconstr, btoa_nsconstr, unormconstr, lnormconstr)
  return HPolytope(allconstrs, allvars) |> remove_redundant_constraints
end

cg_length(iA, oA, iB, oB) = oA*(iA-1)*oB*(iB-1) + oA*(iA-1) + oB*(iB-1)
ld_length(iA, oA, iB, oB) = oA^iA * oB^iB

function cg_to_full(v, iA, oA, iB, oB)
  pabxy = fill(NaN, oA, oB, iA, iB)
  pax = fill(NaN, oA, iA)
  pby = fill(NaN, oB, iB)

  curr = 1
  next = ((oA-1) * (oB-1) * iA * iB)
  pabxy_range = curr:next
  pabxy[1:oA-1, 1:oB-1, 1:iA, 1:iB] = reshape(v[pabxy_range], oA-1, oB-1, iA, iB)

  curr = next + 1
  next = curr + (oA-1) * iA - 1
  pax_range = curr:next
  pax[1:oA-1, 1:iA] = reshape(v[pax_range], oA-1, iA)

  curr = next + 1
  next = curr + (oB-1) * iB - 1
  pby_range = curr:next
  pby[1:oB-1, 1:iB] = reshape(v[pby_range], oB-1, iB)

  for x in eachindex(pax[oA, :])
    pax[oA, x] = 1 - sum(pax[1:oA-1, x])
  end
  for y in eachindex(pax[oB, :])
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

  return pabxy, pax, pby
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
  pabxy, pax, pby = cg_to_full(v, iA, oA, iB, oB)

  sum([correlator(pabxy[:,:,x,y]) for x in 1:iA for y in 1:iB] .* [1, 1, 1, -1])
end

function cg_find_probs(behavs, iA, oA, iB, oB)
  for behav in behavs
    fullbehav = cg_to_full(behav, iA, oA, iB, oB)
    pabxy, pax, pby = fullbehav
    if !(all([all(ps .>= 0) && all(ps .<= 1) for ps in fullbehav])
         && all([sum(pabxy[:, b, x, y]) == pby[b, y] for b in 1:oB for x in 1:iA for y in 1:iB])
         && all([sum(pabxy[a, :, x, y]) == pax[a, x] for a in 1:oA for x in 1:iA for y in 1:iB])
         && all([sum(pabxy[:, :, x, y]) == 1 for x in 1:iA for y in 1:iB]))
      println(behav); println(fullbehav); println()
    end
  end
end

function find_couplers(pbys, n)
  oB, iB = size(pbys[1])
  l = length(pbys) 

  bsysdims = ((oB for i in 1:n)..., (iB for i in 1:n)...)
  bsysranges = (1:d for d in bsysdims)
  vars = @variables C[1:oB-1, bsysranges...]
  allvars = vcat([vec(arr) for arr in vars]...)

  unormconstrs = Vector{Num}(undef, l * oB)
  lnormconstrs = Vector{Num}(undef, l * oB)

  total_pbp = 0
  for idx in eachindex(pbys)
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
chsh_maxvals = [0.5,0.5,0.5,0,0.5,0.5,0.5,0.5]
chsh_poly = cg_polytope(chshset...)
chsh_v = vertices_list(chsh_poly)

qkdset = (2,2,3,2)
qkd_poly = cg_polytope(qkdset...)
qkd_v = vertices_list(qkd_poly)
