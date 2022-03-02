using JuMP, Ipopt, SymEngine
import MathOptInterface as MOI

includet("upper.jl")

# constraints
function J_JuMP_init(us::UpperSetting, pin::AbstractArray, p, M, N, O, P)
  @expanduppersett us
  mdl = Model()
  @variable mdl 0 <= pJ[1:oJ, 1:oA, 1:iA, 1:oB, 1:iB, 1:oE] <= 1
  for idxs in itprod(1:oA, 1:iA, 1:oB, 1:iB, 1:oE)
    @constraint mdl sum(pJ[:, idxs...]) == 1
  end

  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  pvgv = [pin[x,y] * p[a, b, x, y] / pby[b, y]
           for (a, x, b, y) in itprod(1:oA, 1:iA, 1:oB, 1:iB)]
  prob_exprs = full_probs(us, pin, p, pJ, pvgv, M, N, O, P, :sym)

  return mdl, leqconstrs, nleqconstr_exprs, prob_exprs
end

# heuristic: minimise I(V1:V2|E) or H(V1|E) - H(V1|V2) first as a heuristic guess for ABE behaviour

function SymEngine_vars(us::UpperSetting, m::Integer)
  @expanduppersett us
  M = [symbols("M_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:oA, 1:iA)]
  N = [symbols("N_{$(join(idxs, ';'))}") for idxs in itprod(1:dB, 1:oB, 1:iB)]
  O = [symbols("O_{$(join(idxs, ';'))}") for idxs in itprod(1:dE, 1:oE)]
  P = [symbols("P_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:dB, 1:dE)]
  R = [symbols("R_{$(join(idxs, ';'))}") for idxs in itprod(1:m, 1:oA, 1:iA, 1:oE)]

  return M, N, O, P, R
end

function CMI_SymEngine_init(us::UpperSetting, pin::AbstractArray, p, m::Integer)
  @expanduppersett us

  M = [symbols("M_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:oA, 1:iA)]
  N = [symbols("N_{$(join(idxs, ';'))}") for idxs in itprod(1:dB, 1:oB, 1:iB)]
  O = [symbols("O_{$(join(idxs, ';'))}") for idxs in itprod(1:dE, 1:oE)]
  P = [symbols("P_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:dB, 1:dE)]
  vars = [M..., N..., O..., P...]

  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)
  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  probs = CMI_probs(us, pin, M, N, O, P)

  return ineqconstrs, eqconstrs, probs, vars
end

function HAgE_SymEngine_init(us::UpperSetting, pin::AbstractArray, p)
  @expanduppersett us

  M = [symbols("M_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:oA, 1:iA)]
  N = [symbols("N_{$(join(idxs, ';'))}") for idxs in itprod(1:dB, 1:oB, 1:iB)]
  O = [symbols("O_{$(join(idxs, ';'))}") for idxs in itprod(1:dE, 1:oE)]
  P = [symbols("P_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:dB, 1:dE)]
  vars = [M..., N..., O..., P...]

  ineqconstrs, eqconstrs = generate_constrs(us, M, N, O, P, p)
  probs = HAgE_probs(us, pin, M, N, O, P)

  return ineqconstrs, eqconstrs, probs, vars
end

function log_SymEngine_setup(ineqconstrs, eqconstrs, probs, vars)
  mus = [symbols("mu_{$i}") for i in 1:length(ineqconstrs)]
  lambdas = [symbols("lambda_{$i}") for i in 1:length(eqconstrs)]
  obj = (1/log(2)) * sum(p1 * log(p1/p2) for (p1, p2) in probs)
  L = min_lagrangian(obj, mus, ineqconstrs, lambdas, eqconstrs) |> simplify

  return mus, lambdas, obj, L
end

function rat_SymEngine_setup(ineqconstrs, eqconstrs, probs, vars; kwargs...)
  mus = [symbols("mu_{$i}") for i in 1:length(ineqconstrs)]
  lambdas = [symbols("lambda_{$i}") for i in 1:length(eqconstrs)]
  polypairs, polypart = gr_relent_polypairs(probs, m; kwargs...)
  obj = sum(poly[1]/poly[2] for poly in polypairs) + polypart
  L = min_lagrangian(obj, mus, ineqconstrs, lambdas, eqconstrs) |> simplify

  return mus, lambdas, obj, L
end


function CMI_JuMP_init(us::UpperSetting, pin::AbstractArray, p, m::Integer)
  @expanduppersett us
  mdl = Model()

  @variable mdl 0 <= M[1:dA, 1:oA, 1:iA] <= 1
  @variable mdl 0 <= N[1:dB, 1:oB, 1:iB] <= 1
  @variable mdl 0 <= O[1:dE, 1:oE]       <= 1
  @variable mdl 0 <= P[1:dA, 1:dB, 1:dE] <= 1

  leqconstrs = generate_leqconstrs(us, M, N, O, P)
  nleqconstr_exprs = generate_nleqconstr_exprs(us, M, N, O, P, p)
  pby = [sum(pin[x, y] * p[a, b, x, y] for (a, x) in itprod(1:oA, 1:iA))
          for (b, y) in itprod(1:oB, 1:iB)]
  prob_exprs = CMI_probs(us, pin, M, N, O, P, :sym)

  return mdl, leqconstrs, nleqconstr_exprs, prob_exprs
end

function HAgE_JuMP_init(us::UpperSetting, pin::AbstractArray, p)
  @expanduppersett us
  mdl = Model()

  @variable mdl 0 <= M[1:dA, 1:oA, 1:iA] <= 1
  @variable mdl 0 <= N[1:dB, 1:oB, 1:iB] <= 1
  @variable mdl 0 <= O[1:dE, 1:oE]       <= 1
  @variable mdl 0 <= P[1:dA, 1:dB, 1:dE] <= 1

  leqconstrs = generate_leqconstrs(us, M, N, O, P)
  nleqconstr_exprs = generate_nleqconstr_exprs(us, M, N, O, P, p)
  prob_exprs = HAgE_probs(us, pin, M, N, O, P, :sym)

  return mdl, leqconstrs, nleqconstr_exprs, prob_exprs
end

function log_JuMP_setup(mdl, leqconstrs, nleqconstr_exprs, prob_exprs)
  for constr in leqconstrs
    @constraint mdl constr == 0
  end
  for constrexpr in nleqconstr_exprs
    add_NL_constraint(mdl, :($constrexpr == 0))
  end

  obj_terms = [:($p1 * log($p1 / $p2)) for (p1, p2) in prob_exprs]
  obj_expr = sprod(1/log(2), ssum(obj_terms))
  set_NL_objective(mdl, MOI.MIN_SENSE, obj_expr)

  return mdl
end


function rat_JuMP_setup(mdl, leqconstrs, nleqconstr_exprs, prob_exprs, m, mode=:loglb)
  for constr in leqconstrs
    @constraint mdl constr == 0
  end
  for constrexpr in nleqconstr_exprs
    add_NL_constraint(mdl, :($constrexpr == 0))
  end

  # TODO use gr_relent_polypairs
  if mode == :logub
    T, W = logub_gaussradau(m)
  else
    T, W = loglb_gaussradau(m)
  end
  cs = [W[i]/(T[i]*log(2)) for i in 1:m]
  obj_terms = [ssum(sprod(cs[i], szmin_term(p1, p2, T[i])) for i in 1:m)
               for (p1, p2) in prob_exprs]
  obj_expr = sprod(1/log(2), ssum(obj_terms))
  set_NL_objective(mdl, MOI.MIN_SENSE, obj_expr)

  return mdl
end
