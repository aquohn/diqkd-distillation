using JuMP, Ipopt
import MathOptInterface as MOI

includet("upper.jl")

# constraints
function generate_leqconstrs(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

  # expr == 0
  eqconstrs = []
  # pJ normalisation
  if !isnothing(pJ)
    for (a, x, b, y, e) in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
      push!(eqconstrs, sum(pJ[:, a, x, b, y, e]))
    end
  end
  # P normalisation
  push!(eqconstrs, sum(P) - 1)
  # POVM normalisation
  for kA in 1:dA, x in 1:iA
    push!(eqconstrs, sum(M[kA, :, x]) - 1)
  end
  for kB in 1:dB, y in 1:iB
    push!(eqconstrs, sum(N[kB, :, y]) - 1)
  end
  for kE in 1:dE
    push!(eqconstrs, sum(O[kE, :]) - 1)
  end

  return eqconstrs
end

function generate_nleqconstr_exprs(us, M, N, O, P, p, pJ = nothing)
  @expanduppersett us

  # expr == 0
  eqconstrs = Expr[]
  # behaviour constrs
  for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)
    constrterms = Expr[]
    for (kA, kB) in itprod(1:dA, 1:dB)
      sumterm = sprod(M[kA, a, x], N[kB, b, y], sum(P[kA, kB, :]))
      push!(constrterms, sumterm)
    end
    constr = ssum(constrterms..., -p[a,b,x,y])
    push!(eqconstrs, :($constr == 0))
  end

  return eqconstrs
end

function J_JuMP_init(us::UpperSetting, pin::AbstractArray, p::Behaviour, M, N, O, P)
  @expanduppersett us
  mdl = Model()
  @variable mdl 0 <= pJ[1:oJ, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE] <= 1
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
    @constraint mdl sum(pJ[:, idxs...]) == 1
  end

  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  pvgv = [pin[x,y] * p[a, b, x, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  prob_exprs = full_probs(us, pin, p, pJ, pvgv, M, N, O, P, :sym)

  return mdl, leqconstrs, nleqconstr_exprs, prob_exprs
end

# heuristic: minimise I(V1:V2|E) or H(V1|E) - H(V1|V2) first as a heuristic guess for ABE behaviour

function CMI_JuMP_init(us::UpperSetting, pin::AbstractArray, p::Behaviour, m::Integer)
  @expanduppersett us
  mdl = Model()

  @variable mdl 0 <= M[1:dA, 1:oA, 1:iA] <= 1
  @variable mdl 0 <= N[1:dB, 1:oB, 1:iB] <= 1
  @variable mdl 0 <= O[1:dE, 1:oE]       <= 1
  @variable mdl 0 <= P[1:dA, 1:dB, 1:dE] <= 1

  leqconstrs = generate_leqconstrs(us, M, N, O, P, p)
  nleqconstr_exprs = generate_nleqconstr_exprs(us, M, N, O, P, p)
  pby = [sum([pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA)])
          for (b, y) in itprod(1:oB, 1:iB)]
  prob_exprs = CMI_probs(us, pin, M, N, O, P, :sym)

  return mdl, leqconstrs, nleqconstr_exprs, prob_exprs
end

function HAgE_JuMP_init(us::UpperSetting, pin::AbstractArray, p::Behaviour)
  @expanduppersett us
  mdl = Model()

  @variable mdl 0 <= M[1:dA, 1:oA, 1:iA] <= 1
  @variable mdl 0 <= N[1:dB, 1:oB, 1:iB] <= 1
  @variable mdl 0 <= O[1:dE, 1:oE]       <= 1
  @variable mdl 0 <= P[1:dA, 1:dB, 1:dE] <= 1

  leqconstrs = generate_leqconstrs(us, M, N, O, P, p)
  nleqconstr_exprs = generate_nleqconstr_exprs(us, M, N, O, P, p)
  prob_exprs = HAgE_probs(us, pin, M, N, O, P, :sym)

  return mdl, leqconstrs, nleqconstr_exprs, prob_exprs
end

function log_JuMP_setup(mdl, leqconstrs, nleqconstr_exprs, prob_exprs)
  for constr in leqconstrs
    @constraint mdl constr == 0
  end
  for constrexpr in nleqconstr_exprs
    add_NL_constraint(mdl, constrexpr)
  end

  obj_terms = [:($p1 * log($p1 / $p2)) for (p1, p2) in prob_exprs]
  obj_expr = sprod(1/log(2), ssum(obj_terms))
  set_NL_objective(mdl, MOI.MIN_SENSE, obj_expr)

  return mdl
end


function rat_JuMP_setup(mdl, leqconstrs, nleqconstr_exprs, prob_exprs, m)
  for constr in leqconstrs
    @constraint mdl constr == 0
  end
  for constrexpr in nleqconstr_exprs
    add_NL_constraint(mdl, constrexpr)
  end

  T, W = loglb_gaussradau(m)
  cs = [W[i]/(T[i]*log(2)) for i in 1:m]
  obj_terms = [ssum([sprod(cs[i], sz_min_term(p1, p2, T[i])) for i in 1:m])
               for (p1, p2) in prob_exprs]
  obj_expr = sprod(1/log(2), ssum(obj_terms))
  set_NL_objective(mdl, MOI.MIN_SENSE, obj_expr)

  return mdl
end
