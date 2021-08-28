using Zygote
using QuantumInformation
using Plots; plotlyjs()

h(x) = -x*log2(x) - (1-x)*log2(1-x)
r(Q,S) = 1-h(Q)-h((1+sqrt((S/2)^2 - 1))/2)
# plot(range(0,stop=0.5,length=100), range(2,stop=2*sqrt(2),length=100), r, st=:surface, xlabel="Q",ylabel="S")

drdS(S) = gradient((Q, S) -> r(Q,S), 0.05, S)[2]
plot(range(2,stop=2*sqrt(2),length=100), drdS, xlabel="S",ylabel="dr/dS", label="Q = 0.05")


