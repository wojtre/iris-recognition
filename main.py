import data.Iris as iris
import log_reg.LogReg as lg


def test(XT, ZT, classifier):
    errors = 0
    for i in range(len(ZT)):
        if (ZT[i] != classifier.classify(XT[i])):
            errors += 1
    return errors / len(ZT)


# prepare data
split = 0.67
X, XT, Z, ZT, names = iris.loadDataset(split)

# combine two of the 3 classes for a 2 class problem
Z[Z == 2] = 1
ZT[ZT == 2] = 1

# only look at 2 dimensions of the input data for easy visualisation
X = X[:, :2]
XT = XT[:, :2]

model = lg.LogReg(X, Z, epochs=10000)
model.train()
model.plot_decision_boundary(XT, ZT)
print("LogReg err: " + str(test(XT, ZT, model)))
model.save_model("models\\log_reg")
