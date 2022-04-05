using DataFrames
using CSV
includet("helpers.jl")

const MaybeMat = Union{AbstractMatrix, Nothing}
const MaybeVec = Union{AbstractVector, Nothing}

# writes all options after the end
function write_lrs_vrep(name, options, verts::AbstractMatrix, rays::MaybeMat, lins::MaybeMat)
  stat("$name.ext").inode == 0 || throw(ArgumentError("$name.ext already exists!"))
  nverts, d = size(verts)
  Ts = [eltype(verts)]
  isnothing(rays) || push!(Ts, eltype(rays))
  isnothing(lins) || push!(Ts, eltype(lins))
  T = promote_type(Ts...)
  if isnothing(rays)
    rays = Matrix{T}(undef, 0, d)
  end
  if isnothing(lins)
    lins = Matrix{T}(undef, 0, d)
  end
  write_lrs_vrep!(name, options, verts, rays, lins)
end
function write_lrs_vrep!(name, options, verts::AbstractMatrix, rays::AbstractMatrix, lins::AbstractMatrix)
  Ts = [eltype(verts), eltype(rays), eltype(lins)]
  T = promote_type(Ts...)
  nverts, _ = size(verts)
  nrays, _ = size(rays)
  nlins, _ = size(lins)
  vertmat = hcat(ones(T, nverts), verts)
  raymat = hcat(zeros(T, nrays), rays)
  linmat = hcat(zeros(T, nlins), lins)
  Vmat = vcat(linmat, raymat, vertmat)
  m, n = size(Vmat)

  ios = open("$name.ext", "w")
  println(ios, name, "\nV-representation")
  if nlins > 0
    print(ios, "linearity ", nlins, " ")
    for j in 1:nlins
      print(ios, j, " ")
    end
    println(ios)
  end
  print(ios, "begin\n", m, " ", n, " rational\n")
  write_lrs_mat(ios, Vmat)
  println(ios, "end")
  for op in options
    println(ios, op)
  end
  close(ios)
end

function write_lrs_hrep(name, options, ineqAs::MaybeMat, ineqbs::MaybeVec, eqAs::MaybeMat, eqbs::MaybeVec)
  stat("$name.ine").inode == 0 || throw(ArgumentError("$name.ine already exists!"))
  d = 0
  Ts = Type[]
  if !isnothing(eqAs)
    meq, deq = size(eqAs)
    meq == length(eqbs) || throw(ArgumentError("eqAs and eqbs dimensions do not match!"))
    push!(Ts, eltype(eqAs), eltype(eqbs))
    d = deq
  end
  if !isnothing(ineqAs)
    mineq, dineq = size(ineqAs)
    mineq == length(ineqbs) || throw(ArgumentError("ineqAs and ineqbs dimensions do not match!"))
    (d != 0 && d != dineq) && throw(ArgumentError("Inconsistent dimensions!"))
    push!(Ts, eltype(ineqAs), eltype(ineqbs))
    d = dineq
  end
  d != 0 || throw(ArgumentError("Empty input!"))
  T = promote_type(Ts...)
  if isnothing(ineqAs)
    ineqAs = Matrix{T}(undef, 0, d)
    ineqbs = Vector{T}(undef, 0)
  end
  if isnothing(eqAs)
    eqAs = Matrix{T}(undef, 0, d)
    eqbs = Vector{T}(undef, 0)
  end
  write_lrs_hrep!(name, options, ineqAs, ineqbs, eqAs, eqbs)
end
# ineqAs[i,:] \dot x \leq ineqbs[i]; eqAs[i,:] \dot x = eqbs[i]
# from LRSLib: As = -A[:, 2:end], bs = -A[:, 1], use linset to split
function write_lrs_hrep!(name, options, ineqAs::AbstractMatrix, ineqbs::AbstractVector, eqAs::AbstractMatrix, eqbs::AbstractVector)
  ineqmat = hcat(ineqbs, -ineqAs)
  eqmat = hcat(eqbs, -eqAs)
  neqs = length(eqbs)
  Hmat = vcat(eqmat, ineqmat)
  m, n = size(Hmat)

  ios = open("$name.ine", "w")
  println(ios, name, "\nH-representation")
  if neqs > 0
    print(ios, "linearity ", neqs, " ")
    for j in 1:neqs
      print(ios, j, " ")
    end
    println(ios)
  end
  print(ios, "begin\n", m, " ", n, " rational\n")
  write_lrs_mat(ios, Hmat)
  println(ios, "end")
  for op in options
    println(ios, op)
  end
  close(ios)
end

function write_lrs_mat(ios::IOStream, mat::AbstractMatrix{Rational{T}}) where T
  permmat = permutedims(mat)
  for i in axes(permmat, 2)  # rows of mat
    for j in axes(permmat, 1)  # cols of mat
      x = permmat[j, i]
      print(ios, x.num, "/", x.den, " ")
    end
    println(ios)
  end
end
function write_lrs_mat(ios::IOStream, mat::AbstractMatrix{T}) where T <: Integer
  permmat = permutedims(mat)
  for i in axes(permmat, 2)  # rows of mat
    for j in axes(permmat, 1)  # cols of mat
      print(ios, permmat[j, i], " ")
    end
    println(ios)
  end
end

# TODO read back in
function read_lrs_ratint_mat(ios::IOStream, n::Integer)
  start = false
  valre = r"(?<sign>-)?(?<num>\d+)(?<den>/\d+)?"
  skipre = r"^[*#$%^&@!]"
  stopre = r"end"
  maxint = typemax(Int)

  mat = Array{Int}(0, n)
  for line in eachline(ios)
    isnothing(match(skipre, line)) || continue
    isnothing(match(stopre, line)) || break
    row = []
    curridx = 1
    while true
      m = match(valre, line, curridx)
      isnothing(currmatch) && break
      sign = isnothing(m[:sign]) ? 1 : -1
      num = sign * parse(BigInt, m[:num])
      if num <= maxint
        num = Int(num)
      end
      den = isnothing(m[:den]) ? 1 : parse(BigInt, m[:den][2:end])

      if den == 1
        push!(row, num)
      else
        if den <= maxint
          den = Int(den)
        end
        push!(row, num // den)
      end
      curridx = length(currmatch.match) + currmatch.offset
    end
    mat = vcat(mat, row)
  end

  return mat
end
