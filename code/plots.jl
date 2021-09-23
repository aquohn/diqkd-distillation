using Revise
using Plots, LaTeXStrings, ColorSchemes, Printf; plotlyjs()
using Mux, Interact, WebIO
import Contour: contours, levels, level, lines, coordinates
default(:size, (1200,800))

includet("helpers.jl")
includet("keyrates.jl")
includet("wiring.jl")
includet("maxcorr.jl")

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

# %%
# Quantum plots

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

function entropy_plot(; theta=0.15*pi, mus=[pi, 2.53*pi], nus=[2.8*pi, 1.23*pi, pi], ncsamples=20, etasamples=5, etastart=0.8, kwargs...)
  ncs = range(0, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)

  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  qSreg, qSreggrad = entropy_data(ncs, etas, Atlds, Btlds, ABtlds)

  Hmaxs = map(max, qSreg...)
  Hmins = map(min, qSreg...)
  Hranges = [Hmaxs[i] - Hmins[i] for i in 1:2]
  Hlims = [(Hmins[i] - Hranges[i]*0.1, Hmaxs[i] + Hranges[i]*0.1) for i in 1:2]

  etas_const_nc = range(etastart, stop=1, length=50)
  const_nc, const_nc_grad = entropy_data([0.0], etas_const_nc, Atlds, Btlds, ABtlds)

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

function maxcorr_data(ncs, etas, Atlds, Btlds, ABtlds; xysel=(0,0), modesel=:reg)
  oA, oB = 2,2
  iA = length(Atlds); iB = length(Btlds) 

  ncl = length(ncs); etal = length(etas)
  pts = Array{Tuple{Float64, Float64, Float64}}(undef, ncl, etal)

  for nci in eachindex(ncs), etai in eachindex(etas)
    nc = ncs[nci]; eta = etas[etai]
    Eax, Eby, Eabxy, gradeta, gradnc = expt_corrs(nc, eta, Atlds, Btlds, ABtlds)

    # TODO generalise to other ways to bound H(A|E)
    Q = (1 - Eabxy[1,3]) / 2 # QBER H(A|B)
    S = Eabxy[1,1] + Eabxy[1,2] + Eabxy[2,1] - Eabxy[2,2]
    if abs(S) < 2
      S = sign(S) * 2
    end
    pax, pby, pabxy = probs_from_corrs(Eax, Eby, Eabxy)
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
    pts[nci, etai] = (Q, gchsh(S), maxcorrval)
  end

  return pts
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
  pts = maxcorr_data(ncs, etas, Atlds, Btlds, ABtlds; xysel=xysel, modesel=modesel)

  Hs = Vector{Tuple{T, T}}(undef, length(pts))
  rhos = Vector{T}(undef, length(pts))
  for i in eachindex(pts)
    Hs[i] = pts[i][1:2]
    rhos[i] = pts[i][3] end
  Hmaxs = map(max, Hs...)
  Hmins = map(min, Hs...)
  Hranges = [Hmaxs[i] - Hmins[i] for i in 1:2]
  Hlims = [(Hmins[i] - Hranges[i]*0.1, Hmaxs[i] + Hranges[i]*0.1) for i in 1:2]
  rhomax = maximum(x->isnan(x) ? -Inf : x, rhos)

  etas_const_nc = range(etastart, stop=1, length=50)
  const_nc = maxcorr_data([0.0], etas_const_nc, Atlds, Btlds, ABtlds; xysel=xysel, modesel=modesel)

  # show Devetak-Winter frontier
  greyscheme = ColorPalette(ColorScheme([colorant"grey", colorant"grey"]))
  plt = plot([(0, 0, 0), (1, 1, rhomax), (-0.01, 0, rhomax), (1.01, 1, 0)], st=:mesh3d, colorbar_entry=false, seriescolor=greyscheme, alpha=0.5, label="Devetak-Winter bound", xlabel=L"H(A|B)",ylabel=L"H(A|E)",zlabel=L"\max_{x,y} \rho(A,B|x,y)", xlims=Hlims[1], ylims=Hlims[2])

  # find S region
  #=
  for etai in 1:etasamples
  plot!(plt, vec(pts[:, etai]), label=@sprintf "eta = %.3f" etas[etai])
  end
  plot!(plt, vec(const_nc), label="nc = 0")
  =#

  xs, ys, zs = [Array{T}(undef, ncsamples, etasamples) for i in 1:3]
  for nci in 1:ncsamples, etai in 1:etasamples
    xs[nci, etai], ys[nci, etai], zs[nci, etai] = pts[nci, etai]
  end
  plot!(plt, xs, ys, zs, label="Quantum", st = :surface)

  # boundary
  plot!(plt, vec([(pt[1:2]..., 0) for pt in const_nc]), primary=false, linecolor=:blue, label="nc = 0")

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
      plot!(plt, _xs, _ys, _zs, linecolor=:black, primary=false, label="$contourlabel = $lvl")        # add curve on x-y plane
    end
  end

  return plt
end

# %%
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

