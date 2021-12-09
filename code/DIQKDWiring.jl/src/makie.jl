using Revise
using GLMakie, Colors, ColorSchemes, Printf, LaTeXStrings

includet("helpers.jl")
includet("keyrates.jl")
includet("wiring.jl")
includet("maxcorr.jl")

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

function maxcorr_data(ncs, etas, Atlds, Btlds, ABtlds)
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
    maxcorrval = max(maxcorrs(pax, pby, pabxy)...)
    pts[nci, etai] = (Q, gchsh(S), maxcorrval)
  end

  return pts
end

function maxcorr_plot(; theta::T=0.15*pi, mus::Array{T}=[pi, 2.53*pi], nus::Array{T}=[2.8*pi, 1.23*pi, pi], ncsamples=20, etasamples=10, etastart::T=0.65) where T <: Real
  ncs = range(0, stop=1, length=ncsamples)
  etas = range(etastart, stop=1, length=etasamples)

  Atlds, Btlds, ABtlds = meas_corrs(theta=theta, mus=mus, nus=nus)
  pts = maxcorr_data(ncs, etas, Atlds, Btlds, ABtlds)

  Hs = Vector{Tuple{T, T}}(undef, length(pts))
  rhos = Vector{T}(undef, length(pts))
  for i in eachindex(pts)
    Hs[i] = pts[i][1:2]
    rhos[i] = pts[i][3]
  end
  Hmaxs = map(max, Hs...)
  Hmins = map(min, Hs...)
  Hranges = [Hmaxs[i] - Hmins[i] for i in 1:2]
  Hlims = [(Hmins[i] - Hranges[i]*0.1, Hmaxs[i] + Hranges[i]*0.1) for i in 1:2]
  rhomax = maximum(x->isnan(x) ? -Inf : x, rhos)

  etas_const_nc = range(etastart, stop=1, length=50)
  const_nc = maxcorr_data([0.0], etas_const_nc, Atlds, Btlds, ABtlds)

  fig = Figure(resolution = (1200, 800))

  # show Devetak-Winter frontier
  # greyscheme = ColorPalette(ColorScheme([colorant"grey", colorant"grey"]))

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
  return surface(xs, ys, zs, axis=(type=Axis3,))
end
