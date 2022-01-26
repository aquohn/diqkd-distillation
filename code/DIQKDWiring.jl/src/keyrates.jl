using Revise
using Zygote, JuMP
using IntervalArithmetic, IntervalRootFinding
using NaNMath

import MathOptInterface as MOI
import Juniper, NLopt, Ipopt, Cbc

includet("helpers.jl")
includet("nonlocality.jl")

using PythonCall
const pysys = pyimport("sys")
pushfirst!(PyList(pysys.path), Py(@__DIR__))
const qre = pyimport("qre")
const optims = pyimport("optims")

struct EntropyData
  HAE
  HAB
  HAEp
  HABp
end

h(x) = (x == 0 || x == 1) ? 0.0 : -x*log2(x) - (1-x)*log2(1-x)
r(Q,S) = 1-h((1+sqrt((S/2)^2 - 1))/2)-h(Q)

# %%
# Asymmetric CHSH
phi(x) = h(0.5 + 0.5*x)
g(q, alpha, s) = 1 + phi(sqrt((1-2*q)^2 +4*q*(1-q)*((s^2/4)-alpha^2))) - phi((s^2/4)-alpha^2)
function dg(q, alpha, s)
  R1 = sqrt((1-2*q)^2 + 4*q*(1-q)*((s^2/4) - alpha^2))
  R2 = sqrt((s^2/4) - alpha^2)
  return s*q*(q-1)/(2*R1) * log((1-R1)/(1+R1))/log(2) - s/(4*R2) * log((1-R2)/(1+R2))/log(2)
end

function sstar(q, alpha)
  starfn = s -> h(q) - dg(q, alpha, s) * (s - 2) - g(q, alpha, s)
  starl = 2*sqrt(1+alpha^2-alpha^4)
  staru = 2*sqrt(1+alpha^2)
  roots(starfn, starl..staru)
end

function gbar(q, alpha, s)
  if abs(alpha) < 1 && s < sstar(q, alpha)
    return h(q) + dg(q, alpha, sstar(q, alpha)) * (abs(s) - 2)
  else
    return g(q, alpha, s)
  end
end

function gbar_opt(q, alpha, s, sstar)
  if abs(alpha) < 1 && s < sstar
    return h(q) + dg(q, alpha, sstar) * (abs(s) - 2)
  else
    return g(q, alpha, s)
  end
end

gchsh(s) = 1-phi(sqrt(s^2/4 - 1))

QBER(C::Correlators) = (1 - C.Eabxy[1,3]) / 2
CHSH(C::Correlators) = C.Eabxy[1,1] + C.Eabxy[1,2] + C.Eabxy[2,1] - C.Eabxy[2,2]
HAB_oneway(behav::Behaviour) = HAB_oneway(Correlators(behav))
function HAB_oneway(corrs::Correlators)
  return h(QBER(corrs)), nothing
end
HAE_CHSH(behav::Behaviour) = HAE_CHSH(Correlators(behav))
function HAE_CHSH(corrs::Correlators)
  S = CHSH(corrs)
  return gchsh(max(S, 2.0)), nothing
end
HAE_CHSHa(behav::Behaviour) = HAE_CHSHa(Correlators(behav))
function HAE_CHSHa(corrs::Correlators)
  Eabxy = corrs.Eabxy
  corrp = Eabxy[1,1] + Eabxy[1,2]
  corrm = Eabxy[2,1] + Eabxy[2,2]

  star_mdl = asym_chsh_star_model(corrp, corrm)
  nostar_mdl = asym_chsh_nostar_model(corrp, corrm)
  bigalpha_mdl = asym_chsh_bigalpha_model(corrp, corrm)
  mdls = [star_mdl, nostar_mdl, bigalpha_mdl]

  optimize!.(mdls)
  haes = [objective_value(mdl) for mdl in mdls]
  return maximum(haes)
end

nl_solver = optimizer_with_attributes(Ipopt.Optimizer)
mip_solver = optimizer_with_attributes(Cbc.Optimizer)
uniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_solver, "mip_solver" => mip_solver)

isres = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:GN_ISRES)
direct = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:GN_ORIG_DIRECT)
directl = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:GN_ORIG_DIRECT_L)

slsqp = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:LD_SLSQP)
ccsaq = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:LD_CCSAQ)

# optimisation looking at |alpha| < 1 and s < sstar
function asym_chsh_star_model(corrp, corrm; optim=isres)
  if (isnothing(optim))
    mdl = JuMP.Model()
  else
    mdl = JuMP.Model(optim)
  end

  @variable(mdl, -1 <= alpha <= 1)
  @variable(mdl, 0 <= q <= 0.5)
  @variable(mdl, 0 <= sstar <= 2*sqrt(2))

  @NLexpression(mdl, s, alpha * corrp + corrm)
  @NLconstraint(mdl, h(q) + dg(q, alpha, sstar) * (sstar - 2) - g(sstar, q, alpha) == 0)
  @NLconstraint(mdl, s <= sstar)
  @NLobjective(mdl, Max, gbar_opt(q, alpha, s, sstar))

  return mdl
end

# optimisation looking at |alpha| < 1 and s >= sstar
function asym_chsh_nostar_model(corrp, corrm; optim=isres)
  if (isnothing(optim))
    mdl = JuMP.Model()
  else
    mdl = JuMP.Model(optim)
  end

  @variable(mdl, -1 <= alpha <= 1)
  @variable(mdl, 0 <= q <= 0.5)
  @variable(mdl, 0 <= sstar <= 2*sqrt(2))

  @NLexpression(mdl, s, alpha * corrp + corrm)
  @NLconstraint(mdl, h(q) + dg(q, alpha, sstar) * (sstar - 2) - g(sstar, q, alpha) == 0)
  @NLconstraint(mdl, s >= sstar)
  @NLobjective(mdl, Max, g(q, alpha, s))

  return mdl
end

# optimisation looking at |alpha| >= 1
function asym_chsh_bigalpha_model(corrp, corrm; optim=isres)
  if (isnothing(optim))
    mdl = JuMP.Model()
  else
    mdl = JuMP.Model(optim)
  end

  @variable(mdl, -10 <= alpha <= 10)
  @variable(mdl, 0 <= q <= 0.5)

  @NLexpression(mdl, s, alpha * corrp + corrm)
  @NLconstraint(mdl, abs(alpha) >= 1)
  @NLobjective(mdl, Max, g(q, alpha, s))

  return mdl
end

# f = S -> (-0.0625*S^4 + 0.0625*S^2*sqrt(S^2 - 4)*log(0.5 - 0.25*sqrt(S^2 - 4)) - 0.0625*S^2*sqrt(S^2 - 4)*log(0.25*sqrt(S^2 - 4) + 0.5) + 0.25*S^2 - 0.5*sqrt(S^2 - 4)*log(0.5 - 0.25*sqrt(S^2 - 4)) + 0.5*sqrt(S^2 - 4)*log(0.25*sqrt(S^2 - 4) + 0.5))./((0.0625*S^6 - 1.0*S^4 + 5.0*S^2 - 8.0)*log(2))
