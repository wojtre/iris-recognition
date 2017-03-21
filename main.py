import data.Iris as iris
import log_reg.LogReg as lg
import matplotlib.pyplot as plt




# prepare data
split = 0.67
X, XT, Z, ZT, names = iris.loadDataset(split)

# combine two of the 3 classes for a 2 class problem
Z[Z == 2] = 1
ZT[ZT == 2] = 1

# only look at 2 dimensions of the input data for easy visualisation
X = X[:, :2]
XT = XT[:, :2]

model = lg.LogReg(XT, ZT)
model.plot_decision_boundary(X,Z)
