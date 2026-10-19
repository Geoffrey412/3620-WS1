import numpy as np

def load_data(filename):
    age = []
    hg = []
    cls = []
    with open(filename, "r") as f:
        next(f)
        for line in f:
            a, h, c = line.strip().split(",")
            age.append(float(a))
            hg.append(float(h))
            cls.append(int(c))
    return age, hg, cls


def perceptron(age, hg, cls):
    np.random.seed(0)

    age = np.array(age, dtype=float)
    hg = np.array(hg, dtype=float)
    cls = np.array(cls, dtype=int)

    y = np.where(cls == 1, 1, -1)

    X = np.column_stack((np.ones(len(age)), age, hg))

    w = np.zeros(3, dtype=float)

    for _ in range(20000):
        errors = 0

        idx = np.random.permutation(len(X))
        for i in idx:
            if y[i] * np.dot(w, X[i]) <= 0:
                w = w + y[i] * X[i]
                errors += 1

        if errors == 0:
            break

    return w[0], w[1], w[2]
