from FuncDesigner import *
from numpy import ones, arange
nVar = 50
X = oovar()
nConstr = 2

Delta = 0.00000001
Lambda = 0.02
Sigma = 0.00001

P = 1/sqrt(nVar) * ones(nVar)
T = sqrt(nVar)
PX = sum(P * X)
Norma2 = sum((X - PX * P)**2)
F0 = Lambda * PX - sqrt(Delta**2 * PX**2 - Norma2)

F1 = Norma2 + Sigma**2 - Delta**2 * PX**2
F2 = - PX
F_Opt = Sigma*(Lambda/Delta - 1);

S = oosystem(F0) 
S &= (F1<0,  F2<0)
r = S.minimize(F0, {X:cos(arange(nVar))}, iprint=10, ftol = 1e-10, fTol = 0.001, manage=True, maxFunEvals = 1e9, 
                                 xtol = 1e-8, T='float128', maxIter = 1e5, contol=1e-12, solver='gsubg')
print('Obtained objfunc result: %f   Theoretical: %f' % (F0(r), F_Opt))

