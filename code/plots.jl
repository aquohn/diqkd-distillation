using Revise
using Plots, LaTeXStrings, ColorSchemes, Printf; # plotlyjs()
using Mux, Interact, WebIO
import Contour: contours, levels, level, lines, coordinates
default(:size, (1200,800))

includet("helpers.jl")
includet("nonlocality.jl")
includet("keyrates.jl")
includet("wiring.jl")
includet("maxcorr.jl")

# %%
# Constants

# Impt states
EaxQ = [0.0, 0.0]
EbyQ = [0.0, 0.0, 0.0]
EabxyQ = [1/sqrt(2) 1/sqrt(2) 1
          1/sqrt(2) -1/sqrt(2) 0]

Eaxbound = [0.0, 0.0]
Ebybound = [0.0, 0.0, 0.0]
Eabxybound = [0.5 0.5 0.5
              0.5 -0.5 0.5]

EaxPR = [0.0, 0.0]
EbyPR = [0.0, 0.0, 0.0]
EabxyPR = [1.0 1.0 1.0;
           1.0 -1.0 1.0]

Eaxmix = [0.0, 0.0]
Ebymix = [0.0, 0.0, 0.0]
Eabxymix = [0.0 0.0 0.0
            0.0 0.0 0.0]

EaxLD = [1.0, 1.0]
EbyLD = [1.0, 1.0, 1.0]
EabxyLD = [1.0 1.0 1.0
           1.0 1.0 1.0]

# %%
# Pironio rates

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

function plot_sstar(q, alpha)
  starfn = s -> h(q) - dg(q, alpha, s) * (s - 2) - g(q, alpha, s)
  starl = 2*sqrt(1+alpha^2-alpha^4)
  staru = 2*sqrt(1+alpha^2)
  starlv = starfn(starl)
  staruv = starfn(staru)
  @printvals(starl, staru, starlv, staruv)
  plot(starfn, xlims=(starl, staru))
end

# %% 3D plot data generators

function maxcorrval_from_probs(pabxy, pax, pby, xysel, modesel)
    Eax, Eby, Eabxy = corrs_from_probs(pabxy, pax, pby)
    maxcorrvals = maxcorrs(pax, pby, pabxy)
    xsel, ysel = xysel
    if modesel == :maxdiff
      maxcorrval = max(maxcorrvals...) - min(maxcorrvals...)
    elseif modesel == :avg
      maxcorrval = sum(maxcorrvals) / length(maxcorrvals)
    elseif modesel == :corr && xsel != 0 && ysel != 0
      maxcorrval = Eabxy[xsel, ysel]
    else
      if xsel == 0 && ysel == 0
        maxcorrval = max(maxcorrvals...)
      elseif xsel == 0
        maxcorrval = max(maxcorrvals[:, ysel]...)
      elseif ysel == 0
        maxcorrval = max(maxcorrvals[xsel, :]...)
      else
        maxcorrval = maxcorrvals[xsel, ysel]
      end
    end
end

# TODO unify data functions

function maxcorr_data(is, js, HAB::Function, HAE::Function, corrf::Function;
    xysel=(0,0), modesel=:reg, type::Type{T} = Float64) where {T <: Real}
  il = length(is); jl = length(js)
  pts = Array{Tuple{T, T}}(undef, il, jl)
  rhos = Array{T}(undef, il, jl)

  for ii in eachindex(is), ji in eachindex(js)
    i = is[ii]; j = js[ji]
    Eax, Eby, Eabxy = corrf(i, j)
    pabxy, pax, pby = probs_from_corrs(Eax, Eby, Eabxy)
    pts[ii, ji] = (HAB(pabxy, pax, pby)[1], HAE(pabxy, pax, pby)[1])
    rhos[ii, ji] = maxcorrval_from_probs(pabxy, pax, pby, xysel, modesel)
  end

  return pts, rhos
end

function grad_data(is, js, HAB::Function, HAE::Function, corrf::Function, gradf::Function;
    type::Type{T} = Float64) where {T <: Real}
  il = length(is); jl = length(js)
  pts = Array{Tuple{T, T}}(undef, il, jl)
  grads = Array{Tuple{T, T}}(undef, il, jl)

  for ii in eachindex(is), ji in eachindex(js)
    i = is[ii]; j = js[ji]
    Eax, Eby, Eabxy = corrf(i, j)
    pabxy, pax, pby = probs_from_corrs(Eax, Eby, Eabxy)
    pts[ii, ji] = (HAB(pabxy, pax, pby)[1], HAE(pabxy, pax, pby)[1])
    grads[ii, ji] = gradf(i, j, Eax, Eby, Eabxy)
  end

  return pts, grads
end

function appwirf_data(is, js, corrf::Function, krf::Function, wirf::Function;
    type::Type{T} = Float64) where {T <: Real}
  il = length(is); jl = length(js)
  rps = Array{T}(undef, il, jl)

  for ii in eachindex(is), ji in eachindex(js)
    i = is[ii]; j = js[ji]
    wircorrs = Correlators(corrf(i, j)...) |> wirf
    rps[ii, ji] = krf(wircorrs)
  end

  return rps
end

function farkas_wiring_data(n, HAE::Function, HAB::Function; c = 2, iterf = nothing, policy = wiring_policy)
  maxrecs = Union{WiringData, Nothing}[]
  for i in 1:n
    Eax = EaxLD * (i-1)/n + EaxQ * (n-i+1)/n
    Eby = EbyLD * (i-1)/n + EbyQ * (n-i+1)/n
    Eabxy = EabxyLD * (i-1)/n + EabxyQ * (n-i+1)/n
    pax, pby, pabxy = probs_from_corrs(Eax, Eby, Eabxy)

    recs = diqkd_wiring_eval(pax, pby, pabxy, HAE, HAB, c=c, iterf=iterf, policy=policy)
    maxrec = nothing
    maxgain = 0
    for rec in recs
      gain = rec.rp - rec.r
      if gain > maxgain
        maxgain = gain
        maxrec = rec
      end
    end
    push!(maxrecs, maxrec)
  end

  return maxrecs
end

# %%
# Compute values given behaviour params

function wiring_plot(is::AbstractVector{T}, js::AbstractVector{T}, iname, jname, corrf::Function; kwargs...) where T <: Real
  kwargs = Dict(kwargs)
  tol = get(kwargs, :tol, 1e-2)
  krf = get(kwargs, :krf, (corrs) -> gchsh(max(2.0, CHSH(corrs...))) - h(QBER(corrs...)))
  wirings = get(kwargs, :wirings, 
                [((corrs) -> and_corrs(N, corrs...), "$N-AND") for N in 2:6])

  rs = T[Correlators(corrf(i, j)...) |> krf for i in is, j in js]
  basezeros = Tuple{T,T}[]
  for ii in eachindex(is)
    jis = filter(ji -> abs(rs[ii, ji]) < tol, eachindex(js))
    if isempty(jis)
      continue
    end
    maxj = map(ji -> js[ji], jis) |> maximum
    push!(basezeros, (is[ii], maxj))
  end
  plt = plot(basezeros, xlabel=iname, ylabel=jname, label="No wiring", xlims=(is[1], is[end]), ylims=(js[1], js[end]), legend=:topleft, size=(800,600))

  for wiring in wirings
    wirf, wirname = wiring
    rps = appwirf_data(is, js, corrf, krf, wirf)
    wirpts = Tuple{T,T}[]
    for ii in eachindex(is)
      jis = filter(ji -> abs(rps[ii, ji]) < tol, eachindex(js))
      if isempty(jis)
        continue
      end
      maxj = map(ji -> js[ji], jis) |> maximum
      push!(wirpts, (is[ii], maxj))
    end
    plot!(plt, wirpts, label = wirname)
  end

  return plt
end

# %%
# Generate points in a space of behaviours

function qset_plot(QLDsamples = 100, boundsamples = 100, kwargs...)
  fIs = range(0,1,length=boundsamples) |> collect
  fLDs = range(0,1,length=QLDsamples) |> collect
  corrf = (fI, fLD) -> (((1-fI) .* (((1-fLD) .* EaxQ) .+ (fLD .* EaxLD))) .+ (fI .* Eaxbound),
                     ((1-fI) .* (((1-fLD) .* EbyQ) .+ (fLD .* EbyLD))) .+ (fI .* Ebybound),
                     ((1-fI) .* (((1-fLD) .* EabxyQ) .+ (fLD .* EabxyLD))) .+ (fI .* Eabxybound),)
  return wiring_plot(fIs, fLDs, "Isotropic fraction", "Deterministic fraction", corrf, kwargs=kwargs)
end

function expt_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=100, etasamples=100, etastart=0.925, ncstart=0.8, kwargs...)
  ncs = range(ncstart, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)

  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  tldcorrs = Correlators(Atlds, Btlds, ABtlds)
  corrf = (nc, eta) -> expt_corrs(nc, eta, tldcorrs...)

  return wiring_plot(ncs, etas, L"n_c", L"\eta", corrf, kwargs=kwargs)
end

# %%
# Compute values given behaviour params TODO make generic

function entropy_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=20, etasamples=5, etastart=0.8, kwargs...)
  ncs = range(0, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)

  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  corrf = (nc, eta) -> expt_corrs(nc, eta, Atlds, Btlds, ABtlds)
  ncgradf = (nc, eta, Eax, Eby, Eabxy) -> expt_chsh_ncgrads(expt_grads(nc, eta, Eax, Eby, Eabxy)..., CHSH(Eax, Eby, Eabxy))
  qSreg, qSreggrad = grad_data(ncs, etas, HAB_oneway, HAE_CHSH, corrf, ncgradf)

  Hmaxs = map(max, qSreg...)
  Hmins = map(min, qSreg...)
  Hranges = [Hmaxs[i] - Hmins[i] for i in 1:2]
  Hlims = [(Hmins[i] - Hranges[i]*0.1, Hmaxs[i] + Hranges[i]*0.1) for i in 1:2]

  etas_const_nc = range(etastart, stop=1, length=50)
  const_nc, const_nc_grad = grad_data([0.0], etas_const_nc, HAB_oneway, HAE_CHSH, corrf, ncgradf)

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

function maxcorr_plot(; theta::T=0.15*pi, mus::Array{T}=[pi, 2.53*pi], nus::Array{T}=[2.8*pi, 1.23*pi, pi], ncsamples=20, etasamples=100, etastart::T=0.65, kwargs...) where T <: Real
  kwargs = Dict(kwargs)
  ncs = range(0, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)
  xysel = get(kwargs, :xysel, (0,0))
  modesel = get(kwargs, :modesel, :reg)
  contoursel = get(kwargs, :contoursel, :corr)
  ncontours = get(kwargs, :ncontours, 10)

  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  corrf = (nc, eta) -> expt_corrs(nc, eta, Atlds, Btlds, ABtlds)
  Hs, rhos = maxcorr_data(ncs, etas, HAB_oneway, HAE_CHSH, corrf; xysel=xysel, modesel=modesel)

  Hmaxs = map(max, Hs...)
  Hmins = map(min, Hs...)
  Hranges = [Hmaxs[i] - Hmins[i] for i in 1:2]
  Hlims = [(Hmins[i] - Hranges[i]*0.1, Hmaxs[i] + Hranges[i]*0.1) for i in 1:2]
  rhomax = maximum(x->isnan(x) ? -Inf : x, rhos)

  etas_const_nc = range(etastart, stop=1, length=50)
  corrf = (nc, eta) -> expt_corrs(nc, eta, Atlds, Btlds, ABtlds)
  const_nc, const_nc_rhos = maxcorr_data([0.0], etas_const_nc, HAB_oneway, HAE_CHSH, corrf; xysel=xysel, modesel=modesel)

  # show Devetak-Winter frontier
  greyscheme = ColorPalette(ColorScheme([colorant"grey", colorant"grey"]))
  plt = plot([(0, 0, 0), (1, 1, rhomax), (-0.01, 0, rhomax), (1.01, 1, 0)], st=:mesh3d, colorbar_entry=false, seriescolor=greyscheme, alpha=0.5, label="Devetak-Winter bound", xlabel=L"H(A|B)",ylabel=L"H(A|E)",zlabel=L"\max_{x,y} \rho(A,B|x,y)", xlims=Hlims[1], ylims=Hlims[2])

  # find S region
  xs, ys, zs = [Array{T}(undef, ncsamples, etasamples) for i in 1:3]
  for nci in 1:ncsamples, etai in 1:etasamples
    xs[nci, etai], ys[nci, etai], zs[nci, etai] = Hs[nci, etai]..., rhos[nci, etai]
  end
  plot!(plt, xs, ys, zs, label="Quantum", st = :surface)

  # boundary
  plot!(plt, vec([(pt..., 0) for pt in const_nc]), primary=false, linecolor=:blue, label="nc = 0")

  # contours
  if contoursel == :etas
    etazs = Array{T}(undef, ncsamples, etasamples)
    for nci in 1:ncsamples, etai in 1:etasamples
      etazs[nci, etai] = etas[etai]
    end
    contourdata = levels(contours(xs, ys, etazs, ncontours))
    contourlabel = "eta"
  else
    contourdata = levels(contours(xs, ys, zs, ncontours))
    contourlabel = "corr"
  end
  for cl in contourdata
    lvl = level(cl) # the z-value of this contour level
    for line in lines(cl)
      _xs, _ys = coordinates(line) # coordinates of this line segment
      _zs = 0 .* _xs
      plot!(plt, _xs, _ys, _zs, linecolor=:black, primary=false, label="$contourlabel = $lvl") # add curve on x-y plane
    end
  end

  return plt
end

# %%
# Widget code
function iplot(plotf)
  plotlambda = (thetapi, ncsamples, etasamples, etastart, args...) -> begin
    mus = [args[1:2]...] * pi
    nus = [args[3:5]...] * pi
    xysel = args[6:7]
    modesel = args[8]
    contoursel = args[9]
    ncontours = args[10]
    theta = thetapi*pi
    etastart = etastart % 1
    vbox(plotf(theta=theta, mus=mus, nus=nus, ncsamples=ncsamples, etasamples=etasamples, etastart=etastart,
               xysel=xysel, modesel=modesel, contoursel=contoursel, ncontours=ncontours))
  end
  return req -> begin
    # iplt = @manipulate for theta=0.15*pi, muA1=pi, muA2=2.53*pi, nuB1=2.8*pi, nuB2=1.23*pi, nuB3=pi, ncsamples=20, etasamples=10, etastart=0.8
    thetapi = widget(0:0.05:2, label="theta/pi", value=0.15) 
    muA1pi = widget(0:0.05:2, label="muA1/pi", value=1.0)
    muA2pi = widget(0:0.05:2, label="muA2/pi", value=0.5)
    xsel = widget(Dict("max"=>0, "1"=>1, "2"=>2), value=0, label="x")
    nuB1pi = widget(0:0.05:2, label="nuB1/pi", value=0.8)
    nuB2pi = widget(0:0.05:2, label="nuB2/pi", value=1.2)
    nuB3pi = widget(0:0.05:2, label="nuB3/pi", value=1.0)
    ysel = widget(Dict("max"=>0, "1"=>1, "2"=>2, "3"=>3), value=0, label="y")

    ncontours = widget(10, label="num contours")
    modesel = widget(Dict("reg"=>:reg, "max diff"=>:maxdiff, "avg"=>:avg, "corr"=>:corr), value=:reg, label="mode")
    contoursel = widget(Dict("etas"=>:etas, "corr"=>:corr), value=:corr, label="contours")

    ncsamples = widget(20, label="nc samples")
    etasamples = widget(100, label="eta samples")
    etastart = widget(0:0.05:1, label="eta start")

    iplt = map(plotlambda, thetapi, ncsamples, etasamples, etastart, muA1pi, muA2pi, nuB1pi, nuB2pi, nuB3pi,
               xsel, ysel, modesel, contoursel, ncontours)
    vbox(hbox(thetapi, muA1pi, muA2pi),
         hbox(nuB1pi, nuB2pi, nuB3pi),
         hbox(ncsamples, etasamples, etastart, ncontours),
         hbox(xsel, ysel, modesel, contoursel),
         iplt)
  end
end

host_plot(iplt, port=8000) = webio_serve(page("/", iplt), port) # this will serve at http://localhost:8000/

