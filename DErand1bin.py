import random
import numpy as np

boundary_min = -5.12
boundary_max = 5.12
NP = 25
D = 2
F = 0.8
CR = 0.8
Gmax = 1000


def random_init(size, dimension):
    matrix = np.zeros((size, dimension))
    for i in range(size):
        for j in range(dimension):
            matrix[i][j] = random.uniform(boundary_min, boundary_max)
    return matrix


def f_1_de_jong(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i] ** 2
    return sum


def best_x(*vectors, method=1):
    vector = vectors[0]
    if method == 0:
        best = vector[0]
        best_f = f_1_de_jong(best)
        for i in range(len(vector)):
            if f_1_de_jong(vector[i]) <= best_f:
                best = vector[i]
                best_f = f_1_de_jong(vector[i])
    elif method == 1:
        f = vectors[1]
        best = vector[0]
        best_f = f[0]
        for i in range(len(f)):
            if f[i] <= best_f:
                best = vector[i]
                best_f = f[i]
    else:
        best = None
        print("Exception")
    return best


def mutation(vector, index):
    j = index
    while j == index:
        j = random.randint(0, len(vector) - 1)
    k = index
    while k == index or k == j:
        k = random.randint(0, len(vector) - 1)
    m = index
    while m == index or m == j or m == k:
        m = random.randint(0, len(vector) - 1)

    v = vector[j] + F * vector[k] - vector[m]
    for i in range(len(v)):
        if v[i] <= boundary_min or v[i] >= boundary_max:
            v[i] = random.uniform(boundary_min, boundary_max)
    return v


def crossover(x, v):
    probability = 0
    while probability == 0 or probability == 1:
        probability = random.uniform(0, 1)

    if probability <= CR:
        return v
    else:
        return x


G = 0
xbest = np.zeros((Gmax + 1, D))
P = random_init(NP, D)
P_matrix = np.zeros((Gmax, NP, D))
v_matrix = np.zeros((Gmax, NP, D))
u_matrix = np.zeros((Gmax, NP, D))
f_matrix = np.zeros((Gmax, NP))
p_new = np.zeros((NP, D))

xbest[0] = best_x(P, method=0)

while G < Gmax:
    for i in range(NP):
        x = P[i]

        v_matrix[G][i] = mutation(P, i)
        v = v_matrix[G][i]

        u_matrix[G][i] = crossover(x, v)
        u = u_matrix[G][i]

        f_x = f_1_de_jong(x)
        f_u = f_1_de_jong(u)
        if f_u <= f_x:
            p_new[i] = u
            f_matrix[G][i] = f_u
        else:
            p_new[i] = x
            f_matrix[G][i] = f_x
    P = p_new
    G += 1
    xbest[G] = best_x(P, f_matrix[G - 1])

print(xbest)