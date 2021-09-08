# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:percent
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

# %%
# Basic definitions and imports

using Zygote
using QuantumInformation, LinearAlgebra
using IntervalArithmetic, IntervalRootFinding
using Plots, Printf; pyplot()
using Mux, Interact, WebIO
using LazySets, Polyhedra
using JuMP
import MathOptInterface as MOI
import Juniper, NLopt, Ipopt, Cbc
default(:size, (1200,800))

macro printvals(syms...)
  l = length(syms)
  quote
    $([(i == l) ? :(print(String($(Meta.quot(syms[i]))) * " = "); println($(syms[i]))) :
       :(print(String($(Meta.quot(syms[i]))) * " = "); print($(syms[i])); print(", ")) for i in eachindex(syms)]...)
  end |> esc
end

macro sliceup(arr, info...)
  l = Integer(length(info)/2)
  names = Vector{Any}(undef, l)
  steps = Vector{Any}(undef, l)
  for i in eachindex(names)
    names[i] = info[2*i-1]
    steps[i] = info[2*i]
  end

  quote
    curr = 1
    $([:(
         next = curr + $(steps[j]) - 1;
         $(names[j]) = $arr[curr:next]; 
         curr = next + 1
        ) 
       for j in 1:l]...)
  end |> esc
end

function sliceup(arr, steps...)
  l = length(steps)
  sliced = Vector{Any}(undef, l)
  curr = 1
  for j in 1:l
    next = curr + steps[j] - 1
    sliced[j] = arr[curr:next]
    curr = next + 1
  end
end

# %%
# Pironio rates

h(x) = -x*log2(x) - (1-x)*log2(1-x)
r(Q,S) = 1-h((1+sqrt((S/2)^2 - 1))/2)-h(Q)

function plot_pironio(Qval=nothing, Sval=nothing, samples=100)
  if isnothing(Qval) && isnothing(Sval)
    plot(range(0,stop=0.5,length=samples), range(2,stop=2*sqrt(2),length=samples), r, st=:surface, xlabel="Q",ylabel="S")
  elseif isnothing(Sval)
    drdS = S -> gradient((Q, S) -> r(Q,S), Qval, S)[2]
    plot(range(2,stop=2*sqrt(2),length=samples), drdS, xlabel="S",ylabel="dr/dS", label="Q = $Qval")
  else
    drdQ = Q -> gradient((Q, S) -> r(Q,S), Q, Sval)[1]
    plot(range(0,stop=0.5,length=samples), drdQ, xlabel="Q",ylabel="dr/dQ", label="S = $Sval")
  end
end

# %%
# Asymmetric CHSH

phi(x) = h(0.5 + 0.5*x)
g(q, alpha, s) = 1 + phi(sqrt((1-2*q)^2 +4*q*(1-q)*((s^2/4)-alpha^2))) - phi((s^2/4)-alpha^2)
function dg(q, alpha, s)
  R1 = sqrt(q * (4*alpha^2 - s^2) * (q-1) + (2*q-1)^2)
  R2 = sqrt(-4*alpha + s^2)
  return s*q*(q-1)/(2*R1) * log((1+R1)/(1-R1))/log(2) + s/(4*R2) * log((1+R2)/(1-R2))/log(2)
end

function sstar(q, alpha)
  starfn = s -> h(q) - dg(q, alpha, s) * (s - 2) - g(q, alpha, s)
  starl = 2*sqrt(1+alpha^2-alpha^4)
  staru = 2*sqrt(1+alpha^2)
  roots(starfn, starl, staru)
end

function gbar(q, alpha, s)
  if abs(alpha) < 1 && s < sstar(q, alpha)
    return h(q) + dg(q, alpha, sstar(q, alpha)) * (abs(s) - 2)
  else
    return g(q, alpha, s)
  end
end

gchsh(s) = 1-phi(sqrt(s^2/4 - 1))

nl_solver = optimizer_with_attributes(Ipopt.Optimizer)
mip_solver = optimizer_with_attributes(Cbc.Optimizer)
juniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_solver, "mip_solver" => mip_solver)

isres = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:GN_ISRES)
direct = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:GN_ORIG_DIRECT)
directl = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:GN_ORIG_DIRECT_L)

slsqp = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:LD_SLSQP)
ccsaq = optimizer_with_attributes(NLopt.Optimizer, "algorithm"=>:LD_CCSAQ)

function asym_chsh_star_model(; optim=isres)
  if (isnothing(optim))
    mdl = JuMP.Model()
  else
    mdl = JuMP.Model(optim)
  end

  @variable(mdl, -1 <= alpha <= 1)
  @variable(mdl, 0 <= q <= 1)
  @variable(mdl, 0 <= sstar <= 4)

end

# %%
# Wirings

num_wirings(c, o, i) = o^(i^c * o^c) * prod([i^(i^(j-1) * o^(j-1)) for j in 2:c])
num_wirings_fix(c, o, i, f) = o^((i-f)*i^(c-1) * o^c) * prod([i^((i-f)*i^(j-2) * o^(j-1)) for j in 2:c])
function wiring_prob(CA, CAj, CB, CBj, pax, pby, pabxy)
  oA, oB, iA, iB = size(pabxy)
  ppax, ppby, ppabxy = (zeros(size(p)) for p in [pax, pby, pabxy]) |> collect
  c = Integer((CA |> size |> length) / 2)

  # iterate over every possible combination
  for params in Iterators.product(vcat([[1:n for i in 1:c] for n in [oA, oB, iA, iB]]...)...)
    @sliceup params as c bs c xs c ys c
    pABXY = 1
    for j in 1:c
      if CAj[j][as[1:j-1]..., xs[1:j-1]...] != xs[j] || CBj[j][bs[1:j-1]..., ys[1:j-1]...] != ys[j]
        pABXY = 0
        break
      else
        pABXY *= pabxy[as[j], bs[j], xs[j], ys[j]]
      end
    end

    xp = CA[as..., xs...]
    yp = CB[bs..., ys...]
    ppabxy[as[1], bs[1], xp, yp] += pABXY
  end

  for params in Iterators.product(vcat([[1:n for i in 1:c] for n in [oA, iA]]...)...)
    @sliceup params as c xs c
    pAX = 1
    for j in 1:c
      if CAj[j][as[1:j-1]..., xs[1:j-1]...] != xs[j]
        pAX = 0
        break
      else
        pAX *= pax[as[j], xs[j]]
      end
    end

    xp = CA[as..., xs...]
    ppax[as[1], xp] += pAX
  end

  for params in Iterators.product(vcat([[1:n for i in 1:c] for n in [oB, iB]]...)...)
    @sliceup params bs c ys c
    pBY = 1
    for j in 1:c
      if CBj[j][bs[1:j-1]..., ys[1:j-1]...] != ys[j]
        pBY = 0
        break
      else
        pBY *= pby[bs[j], ys[j]]
      end
    end

    yp = CB[bs..., ys...]
    ppax[bs[1], yp] += pBY
  end

  return ppax, ppby, ppabxy
end

# %%
# Koon Tong's model

kd(i,j) = (i == j) ? 1 : 0
E(M, rho) = tr(M * rho) 
sigmas = [[kd(j, 3) kd(j,1)-im*kd(j,2); kd(j,1)+im*kd(j,2) -kd(j,3)] for j in 1:3]
psi(theta) = cos(theta) * kron(ket(1,2), ket(1,2)) + sin(theta) * kron(ket(2,2), ket(2,2))
rho(theta) = proj(psi(theta))
Mtld(mu) = cos(mu) * sigmas[3] + sin(mu) * sigmas[1]

# let outcome 1 be associated with eigenvalue -1 or logical 0
function probs_from_corrs(Eax, Eby, Eabxy)
  invcorr = 0.25 * [-1.0 -1.0 1.0 1.0; -1.0 1.0 -1.0 1.0; 1.0 -1.0 -1.0 1.0; 1.0 1.0 1.0 1.0]

  oA, oB = 2, 2
  iA, iB = [length(Eax), length(Eby)]
  pax = Array{Float64}(undef, oA, iA)
  pby = Array{Float64}(undef, oB, iB)
  pabxy = Array{Float64}(undef, oA, oB, iA, iB)

  for a in 1:oA, x in 1:iA
    pax[a, x] = sum(pabxy[a, :, x, 1])
  end
  for b in 1:oB, y in 1:iB
    pby[b, y] = sum(pby[:, b, 1, y])
  end
  for x in 1:iA, y in 1:iB
    pabxy[:, :, x, y] = invcorr * [Eax[x]; Eby[y]; Eabxy[x,y]; 1]
  end

  return pax, pby, pabxy
end

function corrs_from_probs(pabxy, pax, pby)
  oA, oB, iA, iB = size(pabxy)
  Eax = [pax[2,x] - pax[1,x] for x in 1:iA]
  Eby = [pby[2,y] - pby[1,y] for y in 1:iB]
  Eabxy = [sum([pabxy[a,b,x,y] * ((a == b) ? 1 : -1) for a in 1:oA, b in 1:iA]) for x in 1:iA, y in 1:iB]

  return Eax, Eby, Eabxy
end

function meas_corrs(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi])
  rhov = rho(theta)
  rhoA = ptrace(rhov, [2,2], 2)
  rhoB = ptrace(rhov, [2,2], 1)

  # mus for A, nus for B
  Atlds = Float64[E(Mtld(mu), rhoA) for mu in mus]
  Btlds = Float64[E(Mtld(nu), rhoB) for nu in nus]
  ABtlds = Float64[E(kron(Mtld(mu), Mtld(nu)), rhov) for mu in mus, nu in nus]

  return Atlds, Btlds, ABtlds
end

function expt_corrs(nc, eta, Atlds, Btlds, ABtlds)
  Eax = Float64[-nc-(1-nc)*((1-eta)-eta*Atlds[x]) for x in 1:iA]
  Eby = Float64[-nc-(1-nc)*((1-eta)-eta*Btlds[y]) for y in 1:iB]
  Eabxy = Float64[nc + (1-nc)*(eta^2 * ABtlds[x,y] - eta*(1-eta)*(Atlds[x] + Btlds[y]) + (1-eta)^2) for x in 1:iA, y in 1:iB]

  gradeta = Float64[ (1-nc)*((1-2*eta)*(Atlds[x] + Btlds[y]) - 2*eta*ABtlds[x,y] + 2 - 2*eta) for x in 1:iA, y in 1:iB ] 
  gradnc = Float64[ eta*((1-eta)*(Atlds[x] + Btlds[y]) - eta*ABtlds[x,y] + 2 - eta) for x in 1:iA, y in 1:iB ]

  return Eax, Eby, Eabxy, gradeta, gradnc
end

function entropy_data(ncs, etas, Atlds, Btlds, ABtlds)
  oA, oB = 2,2
  iA = length(Atlds); iB = length(Btlds) 

  ncl = length(ncs); etal = length(etas)
  pts = Array{Tuple{Float64, Float64}}(undef, ncl, etal)
  grads = Array{Tuple{Float64, Float64}}(undef, ncl, etal)

  for nci in eachindex(ncs), etai in eachindex(etas)
    nc = ncs[nci]; eta = etas[etai]
    Eax, Eby, Eabxy, gradeta, gradnc = expt_corrs(nc, eta, Atlds, Btlds, ABtlds)

    # TODO generalise to other ways to bound H(A|E)
    Q = (1 - Eabxy[1,3]) / 2 # QBER H(A|B)
    S = Eabxy[1,1] + Eabxy[1,2] + Eabxy[2,1] - Eabxy[2,2]
    if abs(S) < 2
      S = sign(S) * 2
    end
    pts[nci, etai] = (Q, gchsh(S))

    Qgradnc = - gradnc[1,3]
    if !isfinite(Qgradnc)
      Qgradnc = 0
    end
    Sgradnc = gradnc[1,1] + gradnc[1,2] + gradnc[2,1] - gradnc[2,2]
    if !isfinite(Sgradnc)
      Sgradnc = 0
    end
    HgradS = S/(4*sqrt(S^2-4)) * log2( (2+sqrt(S^2-4)) / (2-sqrt(S^2-4)) )
    if !isfinite(HgradS)
      HgradS = 0
    end
    Hgradnc = Sgradnc * HgradS
    grads[nci, etai] = (Qgradnc, Hgradnc)
  end

  return pts, grads
end

function entropy_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=20, etasamples=5, etastart=0.8)
  ncs = range(0, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)

  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  qSreg, qSreggrad = entropy_data(ncs, etas, Atlds, Btlds, ABtlds)

  Hmaxs = map(max, qSreg...)
  Hmins = map(min, qSreg...)
  Hranges = [Hmaxs[i] - Hmins[i] for i in 1:2]
  Hlims = [(Hmins[i] - Hranges[i]*0.1, Hmaxs[i] + Hranges[i]*0.1) for i in 1:2]

  etas_const_nc = range(etastart, stop=1, length=50)
  const_nc, const_nc_grad = entropy_data([0.0], etas_const_nc, theta=theta, mus=mus, nus=nus)

  # plot qSs/qQs on plot of H(A|E) against H(A|B)
  plt = plot(range(0,stop=1,length=100), range(0,stop=1,length=100), xlabel="H(A|B)",ylabel="H(A|E)", label="Devetak-Winter bound", xlims=Hlims[1], ylims=Hlims[2])

  # find S region
  for etai in 1:etasamples
    plot!(plt, qSreg[:, etai], label=@sprintf "eta = %.3f" etas[etai])
  end
  plot!(plt, vec(const_nc), label="nc = 0")

  quivlens = 0.1 .* Hranges
  quivpts = vcat(qSreg |> vec, const_nc |> vec)
  rawgrads = vcat(qSreggrad |> vec, const_nc_grad |> vec)
  grads = [Tuple([grad[i] * quivlens[i] for i in 1:2]) for grad in rawgrads] 
  quiver!(plt, quivpts, quiver=grads)

  # TODO find Sa regions

  return plt
end

# %%
function interact_entropy_plot(req)
  # entrplot = @manipulate for theta=0:0.05:2*pi, muA1=0:0.05:2*pi, muA2=0:0.05:2*pi, nuB1=0:0.05:2*pi, nuB2=0:0.05:2*pi, nuB3=0:0.05:2*pi, ncsamples=20, etasamples=10, etastart=0:0.05:1
  entrplot = @manipulate for theta=0.15*pi, muA1=pi, muA2=2.53*pi, nuB1=2.8*pi, nuB2=1.23*pi, nuB3=pi, ncsamples=20, etasamples=10, etastart=0.8
    etastart = etastart % 1
    mus = [muA1, muA2]
    nus = [nuB1, nuB2, nuB3]
    vbox(entropy_plot(theta=theta, mus=mus, nus=nus, ncsamples=ncsamples, etasamples=etasamples, etastart=etastart))
  end

  return entrplot
end

host_entropy_plot(port=8000) = webio_serve(page("/", interact_entropy_plot), port) # this will serve at http://localhost:8000/

# %%
# Wiring exhaustive search

function diqkd_wiring_iters(c = 2)
  iA, oA, iB, oB = 2, 2, 3, 2
  # I love this language lmao
  CAshapes = tuple((iA for j in 1:c)..., (oA for j in 1:c)...)
  CBshapes = tuple((iB for j in 1:c)..., (oB for j in 1:c)...)
  CAjshapes = [tuple((iA for k in 1:j-1)..., (oA for k in 1:j-1)...) for j in 1:c]
  CBjshapes = [tuple((iB for k in 1:j-1)..., (oB for k in 1:j-1)...) for j in 1:c]

  CAiters = fill(1:oA, CAshapes...)
  CBiters = fill(1:oB, CBshapes...)
  CAjiters = [fill(1:iA, CAjshapes[j]...) for j in 1:c]
  CBjiters = [fill(1:iB, CBjshapes[j]...) for j in 1:c]

  # Fix keygen settings x = 1 and y = 3 to be AND-gated
  for params in Iterators.product([1:oA for i in 1:c]..., [1:iA for i in 2:c]...)
    @sliceup params as c xs c-1
    CAiters[1, xs..., as...] = all(as .== 2) ? (2:2) : (1:1)
  end
  for params in Iterators.product([1:oB for i in 1:c]..., [1:iB for i in 2:c]...)
    @sliceup params bs c ys c-1
    CBiters[3, ys..., bs...] = all(bs .== 2) ? (2:2) : (1:1)
  end

  # Fix keygen settings to simply broadcast initial input
  for j in 2:c # j == 1 is a 0-dim array
    for params in Iterators.product([1:oA for i in 1:j-1]..., [1:iA for i in 2:j-1]...)
      @sliceup params as j-1 xs j-2
      CAjiters[j][1, xs..., as...] = 1:1
    end
    for params in Iterators.product([1:oB for i in 1:j-1]..., [1:iB for i in 2:j-1]...)
      @sliceup params bs j-1 ys j-2
      CBjiters[j][3, ys..., bs...] = 3:3
    end
  end

  return CAiters, CBiters, CAjiters, CBjiters
end

function diqkd_wiring_eval(Eabxy, Eax, Eby, HAE, HAB)
  CAiters, CBiters, CAjiters, CBjiters = diqkd_wiring_iters()
  iterarrs = [CAiters, CBiters, CAjiters..., CBjiters...]
  shapes = [size(iterarr) for iterarr in iterarrs]
  lengths = [prod(shape) for shape in shapes]

  for data in Iterators.product(vcat([vec(iterarr) for iterarr in iterarrs]...)...)
    info = sliceup(data, lengths...)
    arrs = [reshape(info[j], shapes[j]) for j in eachindex(info)]
    js = Integer((length(arrs) - 2)/2)
    CA, CB, CAj, CBj = sliceup(arrs, 1, 1, js, js)

    HAEval = HAE(Eax, Eby, Eabxy)
    HABval = HAB(Eax, Eby, Eabxy)
    r = HAEval - HABval

    pax, pby, pabxy = probs_from_corrs(Eax, Eby, Eabxy)
    ppax, ppby, ppabxy = wiring_prob(CA, CB, CAj, CBj, pax, pby, pabxy)
    Epax, Epby, Epabxy = corrs_from_probs(ppabxy, ppax, ppby)

    HAEvalp = HAE(Epax, Epby, Epabxy)
    HABvalp = HAB(Epax, Epby, Epabxy)
    rp = HAEvalp - HABvalp
  end
end
