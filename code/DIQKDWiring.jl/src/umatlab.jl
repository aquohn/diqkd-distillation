includet("udirect.jl")
# SymEngine used throughout
const subT = Union{Basic, SymEngine.BasicType}

function lconstrs_to_matrix(lconstrs, vars, coeff=SymEngine.coeff)
  return [coeff(constr, var) for constr in lconstrs, var in vars]
end

struct FminconData
  x
  x0
  lb
  ub
  obj
  grad
  A
  b
  Aeq
  beq
  c
  ceq
  gradc
  gradceq
  ddL
end
function FminconData(obj, leqconstrs, nleqconstrs, lineqconstrs, nlineqconstrs, lb, ub, x0, vars)
  subs = SymEngine.subs
  v = length(vars)
  x = [symbols("x($i)") for i in 1:v]
  submap = [vars[i] => x[i] for i in 1:v]
  grad = [diff(obj, var) for var in vars]
  obj = subs(obj, submap...)
  grad = subs.(grad, submap...)

  A = lconstrs_to_matrix(lineqconstrs, vars, coeff)
  b = lineqconstrs |> length |> zeros
  Aeq = lconstrs_to_matrix(leqconstrs, vars, coeff)
  beq = leqconstrs |> length |> zeros

  clen = length(nlineqconstrs)
  c = subs.(nlineqconstrs, submap...)
  ceqlen = length(nleqconstrs)
  ceq = subs.(nleqconstrs, submap...)
  gradc = [diff(c[j], x[i]) for i in 1:v, j in 1:clen] .|> simplify
  gradceq = [diff(ceq[j], x[i]) for i in 1:v, j in 1:ceqlen] .|> simplify

  leq = Basic[symbols("lambda.eqnonlin($i)") for i in 1:ceqlen]
  l = Basic[symbols("lambda.ineqnonlin($i)") for i in 1:clen]
  L = min_lagrangian(obj, l, nlineqconstrs, leq, nleqconstrs) |> simplify
  dL = lagrangian_grad(L, x, diff) .|> simplify
  ddL = lagrangian_hessian(dL, x, diff)' .|> simplify

  return FminconData(x, x0, lb, ub, obj, grad, A, b, Aeq, beq, c, ceq, gradc, gradceq, ddL)
end
Vector(D::FminconData) = [D.x, D.x0, D.lb, D.ub, D.obj, D.grad, D.A, D.b, D.Aeq, D.beq, D.c, D.ceq, D.gradc, D.gradceq, D.ddL]
function SymEngine.subs(data::FminconData, submaps::Pair...)
  dvec = Vector(data)
  newvec = []
  for v in dvec
    if typeof(v) <: subT
      push!(newvec, SymEngine.subs(v, submaps...))
    elseif eltype(v) <: subT
      push!(newvec, SymEngine.subs.(v, submaps...))
    else
      push!(newvec, v)
    end
  end
  return FminconData(newvec...)
end

function write_mat(io::IOStream, mat::AbstractMatrix, istr)
  (r, c) = size(mat)
  if c == 0
    return
  end
  for i in 1:r
    print(io, istr, join(mat[i, :], " "), ";\n")
  end
end

function write_fmincon(name::AbstractString, F::FminconData, usehessian=false)
  io = open(name * ".m", "w")
  print(io, "function [problem, zeroobjf] = $name()\n",
            "  function [c, ceq, gradc, gradceq] = nonlcon(x)\n",
            "    c = [\n")
  for x in F.c
  print(io, "         ", x, ";\n")
  end
  print(io, "    ];\n",
            "    ceq = [\n")
  for x in F.ceq
  print(io, "         ", x, ";\n")
  end
  print(io, "    ];\n",
            "    if nargout > 2\n",
            "      gradc = [\n")
  istr =    "              "
  write_mat(io, F.gradc, istr)
  print(io, "              ];\n",
            "      gradceq = [\n")
  istr =    "                "
  write_mat(io, F.gradceq, istr)
  print(io, "              ];\n",
            "    end\n",
            "  end\n\n")

  print(io, "  function [obj, grad] = obj(x)\n")
  print(io, "    obj = ", F.obj, ";\n",
            "    if nargout > 1\n",
            "      grad = [\n")
  for x in F.grad
  print(io, "             ", x, ";\n")
  end
  print(io, "      ];\n",
            "    end\n",
            "  end\n\n")

  print(io, "  function [obj, grad] = zeroobj(x)\n")
  print(io, "    obj = 0;\n",
            "    if nargout > 1\n",
            "      grad = [\n")
  print(io, "             ", join(zeros(length(F.x)), ' '), '\n')
  print(io, "      ];\n",
            "    end\n",
            "  end\n\n")

  if usehessian
  print(io, "  function [H] = hessian(x, lambda)\n",
            "    H = [\n")
  istr =    "        "
  write_mat(io, F.ddL, istr)
  print(io, "    ];\n")
  print(io, "  end\n")
  end

  print(io, "  A = [\n")
  istr =    "    "
  write_mat(io, F.A, istr)
  print(io, "];\n")
  print(io, "  b = [", join(F.b, ' '), "];\n")

  print(io, "  Aeq = [\n")
  istr =    "    "
  write_mat(io, F.Aeq, istr)
  print(io, "];\n")
  print(io, "  beq = [", join(F.beq, ' '), "];\n")

  print(io, "  x0 = [", join(F.x0, ' '), "];\n")
  print(io, "  lb = [", join(F.lb, ' '), "];\n")
  print(io, "  ub = [", join(F.ub, ' '), "];\n")

  hessstr = usehessian ? ", 'HessianFcn', @hessian" : ""
  print(io, "opts = optimoptions('fmincon', 'Algorithm', 'interior-point', 'UseParallel', true, 'SpecifyObjectiveGradient', true, 'SpecifyConstraintGradient', true$hessstr);\n")
  # , 'MaxFunctionEvaluations', 1e+6

  print(io, "problem = createOptimProblem('fmincon', 'x0', x0, 'lb', lb, 'ub', ub, 'Aineq', A, 'bineq', b, 'Aeq', Aeq, 'beq', beq, 'objective', @obj, 'nonlcon', @nonlcon, 'options', opts);\n\n")

  print(io, "  if nargout > 1\n",
            "    zeroobjf = @zeroobj;\n",
            "  end\n")


  print(io, "end\n")
  close(io)
end

function generic_map(vpin, vp, pin, p::Behaviour)
  mapv = [vpin[i] => pin[i] for i in eachindex(vpin)]
  return vcat(mapv, [vp[i] => p[i] for i in eachindex(vp)])
end

function generic_HAgE_fmincon(us::UpperSetting; kwargs...)
  @expanduppersett us
  pin = [symbols("pin_{$(join(idxs, ';'))}") for idxs in itprod(1:iA, 1:iB)]
  p = [symbols("p_{$(join(idxs, ';'))}") for idxs in itprod(1:oA, 1:oB, 1:iA, 1:iB)]

  M = [symbols("M_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:oA, 1:iA)]
  N = [symbols("N_{$(join(idxs, ';'))}") for idxs in itprod(1:dB, 1:oB, 1:iB)]
  O = [symbols("O_{$(join(idxs, ';'))}") for idxs in itprod(1:dE, 1:oE)]
  P = [symbols("P_{$(join(idxs, ';'))}") for idxs in itprod(1:dA, 1:dB, 1:dE)]
  vars = [M..., N..., O..., P...]
  obj = (1/log(2)) * sum(p1 * log(p1/p2) for (p1, p2) in probs)

  leqconstrs = generate_leqconstrs(us, M, N, O, P)
  nleqconstrs = eval.(generate_nleqconstr_exprs(us, M, N, O, P, p))
  v = length(vars)
  lb = zeros(Float64, v)
  ub = ones(Float64, v)
  x0 = (lb + ub)/2

  optdata = FminconData(obj, leqconstrs, nleqconstrs, Basic[], Basic[], lb, ub, x0, vars)
  return optdata, pin, p
end
