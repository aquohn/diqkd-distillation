# Testing zone
using Symbolics
Symbolics.@variables PA[1:2, 1:2] PB[1:2, 1:2] PC[1:3, 1:2] PD[1:2, 1:3]
PAvec = BehaviourVec(PA); PBvec = BehaviourVec(PB)
PCvec = BehaviourVec(PC); PDvec = BehaviourVec(PD)
PABp = kron(PAvec.p, PBvec.p)
PBAp = kron(PBvec.p, PAvec.p)
permPABp = permutesystems(PABp, [4, 4], [2, 1])
PABAB = kron(PABp, PABp)
permPAABB = permutesystems(PABAB, [4, 4, 4, 4], [1, 3, 2, 4])
PAABB = kron(PAvec.p, PAvec.p, PBvec.p, PBvec.p)
PABCD = kron(PAvec.p, PBvec.p, PCvec.p, PDvec.p)
PDACB = kron(PDvec.p, PAvec.p, PCvec.p, PBvec.p)
permPDACB = permutesystems(PABCD, [4, 4, 6, 6], [4, 1, 3, 2])

Symbolics.@variables PAB1[1:2, 1:2, 1:2, 1:2] PAB2[1:2, 1:2, 1:2, 1:2]
PAB1vec = BehaviourVec([2,2], [2,2], PAB1)
PAB2vec = BehaviourVec([2,2], [2,2], PAB2)
Pand2 = zeros(Num, 2, 2, 2, 2)
P1and2 = zeros(Num, 2, 2, 2, 2)
for (a1, b1, a2, b2, x, y) in itprod(repeat([1:2], 6)...)
   a = (a1 + a2 == 4) ? 2 : 1
   b = (b1 + b2 == 4) ? 2 : 1
   Pand2[a, b, x, y] += PAB1[a1, b1, x, y] * PAB2[a2, b2, x, y]
   P1and2[a, b, x, y] += PAB1[a1, b1, x, y] * PAB1[a2, b2, x, y]
end
margWand222 = MargWiring(2, 2, 2, and_Wmap(2,2))
margWfirst222 = MargWiring(2, 2, 2, first_Wmap(2,2,2))
PABc = kron(PAB1vec.p, PAB2vec.p)
permPABc = permutesystems(PABc, [4,4,4,4], [1, 3, 2, 4])
