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

# %% Plot data generators

function maxcorrval_from_probs(behav::Behaviour, xysel, modesel)
    Eax, Eby, Eabxy = Correlators(behav)
    maxcorrvals = maxcorrs(behav)
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

function maxcorr_data(is, js, HAB::Function, HAE::Function, corrf::Function;
    type::Type{T} = Float64, kwargs...) where {T <: Real}
  kwd = Dict(kwargs)
  xysel = get(kwd, :xysel, (0,0))
  modesel = get(kwd, :modesel, :reg)
  il = length(is); jl = length(js)
  pts = Array{Tuple{T, T}}(undef, il, jl)
  rhos = Array{T}(undef, il, jl)

  for ii in eachindex(is), ji in eachindex(js)
    i = is[ii]; j = js[ji]
    corrs = corrf(i, j)
    behav = Behaviour(corrs)
    pts[ii, ji] = (HAB(behav)[1], HAE(behav)[1])
    rhos[ii, ji] = maxcorrval_from_probs(behav, xysel, modesel)
  end

  return pts, rhos
end

function maxcorr_3ddata(is, js, HAB::Function, HAE::Function, corrf::Function;
    type::Type{T} = Float64, kwargs...) where {T <: Real}
  pts, rhos = maxcorr_data(is, js, HAB, HAE, corrf, type=type; kwargs...)
  pts3d = Array{Tuple{T, T, T}}(undef, size(pts)...)

  for ii in eachindex(is), ji in eachindex(js)
    pts3d[ii, ji] = (pts[ii, ji]..., rhos[ii, ji])
  end

  return pts3d
end

function grad_data(is, js, HAB::Function, HAE::Function, corrf::Function, gradf::Function;
    type::Type{T} = Float64) where {T <: Real}
  il = length(is); jl = length(js)
  pts = Array{Tuple{T, T}}(undef, il, jl)
  grads = Array{Tuple{T, T}}(undef, il, jl)

  for ii in eachindex(is), ji in eachindex(js)
    i = is[ii]; j = js[ji]
    corrs = corrf(i, j)
    pabxy, pax, pby = Behaviour(corrs)
    pts[ii, ji] = (HAB(behav)[1], HAE(behav)[1])
    grads[ii, ji] = gradf(i, j, corrs)
  end

  return pts, grads
end

function appwirf_data(is, js, corrf::Function, krf::Function, wirf::Function;
    type::Type{T} = Float64) where {T <: Real}
  il = length(is); jl = length(js)
  rps = Array{T}(undef, il, jl)

  for ii in eachindex(is), ji in eachindex(js)
    i = is[ii]; j = js[ji]
    wircorrs = corrf(i, j) |> wirf
    rps[ii, ji] = krf(wircorrs)
  end

  return rps
end

# n points along the line between LD and critical Werner state
# search for optimal wiring
function farkas_wiring_data(n, HAE::Function, HAB::Function; c = 2, iterf = nothing, policy = wiring_policy)
  maxrecs = Union{WiringData, Nothing}[]
  for i in 1:n
    corrs = convexsum([(i-1)/n, (n-i+1)/n], [LD_corrs, werner_corrs(v_crit)])
    behav = Behaviour(corrs)

    recs = diqkd_wiring_eval(behav, HAE, HAB, c=c, iterf=iterf, policy=policy)
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
# Correlation function generators

function qset_corrf()
  corrf = (fI, fLD) -> convexsum([fI, (1-fI)*(1-fLD), (1-fI)*fLD], 
                                 [bound_corrs, singlet_corrs, LD_corrs])
  return corrf
end

function expt_corrf(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi])
  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  tldcorrs = Correlators(Atlds, Btlds, ABtlds)
  corrf = (nc, eta) -> expt_corrs(nc, eta, tldcorrs)
  return corrf
end

# %%
# Given i, minimum j req'd for r > 0
function wiring_plot(is::AbstractVector{T}, js::AbstractVector{T}, iname, jname, corrf::Function; kwargs...) where T <: Real
  kwargs = Dict(kwargs)
  tol = get(kwargs, :tol, 1e-2)
  krf = get(kwargs, :krf, (corrs) -> gchsh(max(2.0, CHSH(corrs))) - h(QBER(corrs)))
  wirings = get(kwargs, :wirings, 
                [((corrs) -> and_corrs(N, corrs), "$N-AND") for N in 2:6])

  rs = T[corrf(i, j) |> krf for i in is, j in js]
  basezeros = Tuple{T,T}[]
  for ii in eachindex(is)
    # find elements close to 0
    jis = filter(ji -> abs(rs[ii, ji]) < tol, eachindex(js))
    if isempty(jis)
      continue
    end
    # find maximum of these elements
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
# Compute values given behaviour params TODO make generic

function expt_grad_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=20, etasamples=5, etastart=0.8, kwargs...)
  ncs = range(0, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)

  tldcorrs = meas_corrs(theta=theta, mus=mus, nus=nus)
  corrf = (nc, eta) -> expt_corrs(nc, eta, tldcorrs)
  ncgradf = (nc, eta, corr) -> expt_chsh_ncgrads(expt_grads(nc, eta, Eax, Eby, Eabxy)..., CHSH(Eax, Eby, Eabxy))
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

# Adds the plane x = y to plt
function DW_frontier_plot(plt, xs, ys, zs)
  # show Devetak-Winter frontier
  uss = [filter(u -> isfinite(u), vec(us)) for us in (xs, ys, zs)]
  ptmins = [minimum(us) for us in uss]
  ptmaxs = [maximum(us) for us in uss]
  ptranges = [ptmaxs[i] - ptmins[i] for i in eachindex(uss)]
  ptlims = [(ptmins[i]-0.1*ptranges[i], ptmaxs[i]+0.1*ptranges[i]) for i in eachindex(uss)]

  greyscheme = ColorPalette(ColorScheme([colorant"grey", colorant"grey"]))
  DW_corners = [(ptmins[1], ptmins[2], ptmins[3]), 
                (ptmaxs[1], ptmaxs[2], ptmaxs[3]),
                (ptmins[1] - 0.01*ptranges[1], ptmins[2], ptmaxs[3]),
                (ptmaxs[1] + 0.01*ptranges[1], ptmaxs[2], ptmins[3])]
  plot!(plt, DW_corners, st=:mesh3d, colorbar_entry=false, seriescolor=greyscheme, alpha=0.5, label="Devetak-Winter bound")
end

function ij_xyz_plot(is, js, dataf, names::Dict{Symbol, S}; type::Type{T} = Float64, kwargs...) where {T <: Real, S <: AbstractString}
  kwd = Dict(kwargs)
  contoursel = get(kwd, :contoursel, :z)
  ncontours = get(kwd, :ncontours, 10)
  addplotf = get(kwd, :addplotf, (plt, xs, ys, zs) -> nothing)
  contourzf = get(kwd, :contourzf, (lvl, _xs, _ys) -> fill(lvl, size(_xs)))
  imax, imin, ilen = maximum(is), minimum(is), length(is)
  jmax, jmin, jlen = maximum(js), minimum(js), length(js)

  pts = dataf(is, js; kwargs...)
  xs, ys, zs = [Array{T}(undef, ilen, jlen) for i in 1:3]
  for ii in 1:ilen, ji in 1:jlen
    xs[ii, ji], ys[ii, ji], zs[ii, ji] = pts[ii, ji]
  end
  plt = plot(xs, ys, zs, xlabel=names[:x], ylabel=names[:y], zlabel=names[:z], label=names[:set], st = :surface)

  # boundary
  boundvals = [(imin, js), (imax, js), (is, jmin), (is, jmax)]
  plot!(plt, vec([(c[1], c[2], 0) for c in dataf([imin], js; kwargs...)]), primary=false, linecolor=:blue, label="Minimal $(names[:i])")
  plot!(plt, vec([(c[1], c[2], 0) for c in dataf([imax], js; kwargs...)]), primary=false, linecolor=:blue, label="Maximal $(names[:i])")
  plot!(plt, vec([(c[1], c[2], 0) for c in dataf(is, [jmin]; kwargs...)]), primary=false, linecolor=:blue, label="Minimal $(names[:j])")
  plot!(plt, vec([(c[1], c[2], 0) for c in dataf(is, [jmax]; kwargs...)]), primary=false, linecolor=:blue, label="Maximal $(names[:j])")

  # contours
  if contoursel == :i
    izs = Array{T}(undef, ilen, jlen)
    for ii in 1:ilen, ji in 1:jlen
      izs[ii, ji] = is[ii]
    end
    contourdata = levels(contours(xs, ys, izs, ncontours))
    contourlabel = names[:i]
  end
  if contoursel == :j
    jzs = Array{T}(undef, ilen, jlen)
    for ii in 1:ilen, ji in 1:jlen
      jzs[ii, ji] = js[ji]
    end
    contourdata = levels(contours(xs, ys, jzs, ncontours))
    contourlabel = names[:j]
  else
    contourdata = levels(contours(xs, ys, zs, ncontours))
    contourlabel = names[:z]
  end
  for cl in contourdata
    lvl = level(cl) # the z-value of this contour level
    for line in lines(cl)
      _xs, _ys = coordinates(line) # coordinates of this line segment
      _zs = contourzf(lvl, _xs, _ys)
      plot!(plt, _xs, _ys, _zs, linecolor=:black, primary=false, label="$contourlabel = $lvl") # add curve on x-y plane
    end
  end

  addplotf(plt, xs, ys, zs)

  return plt
end

# %%
# Top level plotting
function qset_wiring_plot(QLDsamples = 100, boundsamples = 100; kwargs...)
  fIs = range(0,1,length=boundsamples) |> collect
  fLDs = range(0,1,length=QLDsamples) |> collect
  return wiring_plot(fIs, fLDs, "Isotropic fraction", "Deterministic fraction", qset_corrf(); kwargs...)
end

function expt_wiring_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=100, etasamples=100, etastart=0.925, ncstart=0.8, kwargs...)
  ncs = range(ncstart, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)
  corrf = expt_corrf(theta=theta, mus=mus, nus=nus)
  return wiring_plot(ncs, etas, L"n_c", L"\eta", corrf; kwargs...)
end

function qset_kr_plot(QLDsamples = 100, boundsamples = 100; type::Type{T} = Float64,  kwargs...) where {T <: Real}
  fIs = range(0,1,length=boundsamples) |> collect
  fLDs = range(0,1,length=QLDsamples) |> collect
  names = Dict{Symbol, String}(:x => "Isotropic Fraction", :y => "Local Fraction", :z => "Keyrate",
                               :i => "Isotropic Fraction", :j => "Local Fraction", :set => "Polytope Slice"
                              )
  fIs = range(0,1,length=boundsamples) |> collect
  fLDs = range(0,1,length=QLDsamples) |> collect
  corrf = qset_corrf()
  dataf = (is, js; kwargs...) -> begin
    il = length(is); jl = length(js)
    pts = Array{Tuple{T, T, T}}(undef, il, jl)
    for ii in eachindex(is), ji in eachindex(js)
      i = is[ii]; j = js[ji]
      corrs = corrf(i, j)
      behav = Behaviour(corrs)
      hab, hae = HAB_oneway(behav)[1], HAE_CHSH(behav)[1]
      pts[ii, ji] = (i, j, hae - hab)
    end
    return pts
  end
  return ij_xyz_plot(fIs, fLDs, dataf, names; addplotf=DW_frontier_plot, contourzf = (lvl, _xs, _ys) -> 0 .* _xs, kwargs...)
end

function expt_maxcorr_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=100, etasamples=100, etastart=0.65, ncstart=0.0, kwargs...) 
  names = Dict{Symbol, String}(:x => "H(A|B)", :y => "H(A|E)", :z => "Maximal Correlation",
                               :i => L"n_c", :j => L"\eta", :set => "Experimental Model"
                              )
  ncs = range(ncstart, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)
  corrf = expt_corrf(theta=theta, mus=mus, nus=nus)
  dataf = (is, js; kwargs...) -> maxcorr_3ddata(is, js, HAB_oneway, HAE_CHSH, corrf; kwargs...)
  return ij_xyz_plot(ncs, etas, dataf, names; addplotf=DW_frontier_plot, contourzf = (lvl, _xs, _ys) -> 0 .* _xs, kwargs...)
end

function qset_maxcorr_plot(QLDsamples = 100, boundsamples = 100; kwargs...) 
  names = Dict{Symbol, String}(:x => "H(A|B)", :y => "H(A|E)", :z => "Maximal Correlation",
                               :i => L"f_I", :j => L"f_{LD}", :set => "Polytope Slice"
                              )
  fIs = range(0,1,length=boundsamples)
  fLDs = range(0,1,length=QLDsamples)
  corrf = qset_corrf()
  dataf = (is, js; kwargs...) -> maxcorr_3ddata(is, js, HAB_oneway, HAE_CHSH, corrf; kwargs...)
  return ij_xyz_plot(fIs, fLDs, dataf, names; addplotf=DW_frontier_plot, contourzf = (lvl, _xs, _ys) -> 0 .* _xs, kwargs...)
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

