import random
import numpy as np
import xlwings

boundary_min_1_de_jong = -5
boundary_max_1_de_jong = 5

boundary_min_2_de_jong = -2
boundary_max_2_de_jong = 2

boundary_min_schwefel = -500
boundary_max_schwefel = 500

boundary_min_rastrigin = -5
boundary_max_rastrigin = 5

boundary_min_ackley_2 = -20
boundary_max_ackley_2 = 20

NP = 50
D = 10
Gmax = 2000
choice = 2

if choice == 1:
    boundary_min = boundary_min_1_de_jong
    boundary_max = boundary_max_1_de_jong
elif choice == 2:
    boundary_min = boundary_min_2_de_jong
    boundary_max = boundary_max_2_de_jong
elif choice == 3:
    boundary_min = boundary_min_schwefel
    boundary_max = boundary_max_schwefel
elif choice == 4:
    boundary_min = boundary_min_rastrigin
    boundary_max = boundary_max_rastrigin
elif choice == 5:
    boundary_min = boundary_min_ackley_2
    boundary_max = boundary_max_ackley_2
else:
    boundary_min = None
    boundary_max = None


def random_init(size, dimension):
    matrix = np.zeros((size, dimension))
    for i in range(size):
        for j in range(dimension):
            matrix[i][j] = random.uniform(boundary_min, boundary_max)
    return matrix


def cr_f_init():
    cr = np.zeros((Gmax + 1, NP))
    f = np.zeros((Gmax + 1, NP))
    for i in range(NP):
        cr[0][i] = 0.5
        f[0][i] = 0.9
    return cr, f


def f_1_de_jong(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i] ** 2
    return sum


def f_2_de_jong(x):
    sum = 0
    for i in range(len(x) - 1):
        sum += 100 * ((x[i] ** 2 - x[i + 1]) ** 2) + ((1 - x[i]) ** 2)
    return sum


def schwefel_func(x):
    sum = 0
    for i in range(len(x)):
        sum += - x[i] * np.sin(np.sqrt(np.abs(x[i])))
    return sum


def rastrigin_func(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
    return 2 * D * sum


def ackley_2_func(x):
    sum = 0
    for i in range(len(x) - 1):
        sum += 20 + np.e - (20 / (np.power(np.e, 0.2 * np.sqrt((x[i] ** 2 + x[i + 1] ** 2) / 2)))) - np.power(np.e, 0.5
                                                                                                              * (np.cos(
            2 * np.pi * x[i]) + np.cos(2 * np.pi * x[i + 1])))
    return sum


def choose_func(x):
    if choice == 1:
        return f_1_de_jong(x)
    elif choice == 2:
        return f_2_de_jong(x)
    elif choice == 3:
        return schwefel_func(x)
    elif choice == 4:
        return rastrigin_func(x)
    elif choice == 5:
        return ackley_2_func(x)
    else:
        return None


def best_x(*vectors, method=1):
    vector = vectors[0]
    if method == 0:
        best = vector[0]
        best_f = choose_func(best)
        for i in range(len(vector)):
            current_best = choose_func(vector[i])
            if current_best <= best_f:
                best = vector[i]
                best_f = choose_func(vector[i])
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
        best_f = None
        print("Exception")
    return best, best_f


def mutation(vector, index, f):
    probability = 0
    while probability == 0 or probability == 1:
        probability = random.uniform(0, 1)
    if probability <= 0.1:
        new_f = random.uniform(0.1, 0.9)
    else:
        new_f = f

    j = index
    while j == index:
        j = random.randint(0, len(vector) - 1)
    k = index
    while k == index or k == j:
        k = random.randint(0, len(vector) - 1)
    m = index
    while m == index or m == j or m == k:
        m = random.randint(0, len(vector) - 1)

    v = vector[j] + new_f * (vector[k] - vector[m])
    for i in range(len(v)):
        if v[i] <= boundary_min or v[i] >= boundary_max:
            v[i] = random.uniform(boundary_min, boundary_max)
    return v, new_f


def crossover(x, v, cr):
    probability = 0
    while probability == 0 or probability == 1:
        probability = random.uniform(0, 1)
    if probability <= 0.1:
        new_cr = random.uniform(0, 1)
    else:
        new_cr = cr
    crossovered_vector = np.zeros(D)

    for i in range(len(x)):
        probability = 0
        while probability == 0 or probability == 1:
            probability = random.uniform(0, 1)
        if probability <= new_cr:
            crossovered_vector[i] = v[i]
        else:
            crossovered_vector[i] = x[i]

    return crossovered_vector, new_cr

G = 0
xbest = np.zeros((Gmax + 1, D))
f_best = np.zeros((Gmax + 1, 1))
P = random_init(NP, D)
P_matrix = np.zeros((Gmax, NP, D))
v_matrix = np.zeros((Gmax, NP, D))
u_matrix = np.zeros((Gmax, NP, D))
f_matrix = np.zeros((Gmax, NP))
cr_matrix, f_mutation_matrix = cr_f_init()
p_new = np.zeros((NP, D))

xbest[0], f_best[0] = best_x(P, method=0)

while G < Gmax:
    for i in range(NP):
        x = P[i]

        v_matrix[G][i], f_new = mutation(P, i, f_mutation_matrix[G][i])
        v = v_matrix[G][i]

        u_matrix[G][i], cr_new = crossover(x, v, cr_matrix[G][i])
        u = u_matrix[G][i]

        f_x = choose_func(x)
        f_u = choose_func(u)
        if f_u <= f_x:
            p_new[i] = u
            f_matrix[G][i] = f_u
            f_mutation_matrix[G + 1][i] = f_new
            cr_matrix[G + 1][i] = cr_new
        else:
            p_new[i] = x
            f_matrix[G][i] = f_x
            f_mutation_matrix[G + 1][i] = f_mutation_matrix[G][i]
            cr_matrix[G + 1][i] = cr_matrix[G][i]
    P = p_new
    G += 1
    xbest[G], f_best[G] = best_x(P, f_matrix[G - 1])

for i in range(len(xbest)):
    print(f_best[i])
print(xbest[-1])


# wb_statistics = xlwings.Book('Statistics.xlsx')
# ws_statistics = wb_statistics.sheets[9]
# wb_convergence = xlwings.Book('Convergence (1) (1).xlsx')
# ws_convergence = wb_convergence.sheets[9]
#
# row = 30
# for i in range(len(xbest[-1])):
#     ws_statistics.range(row, i + 1).value = xbest[-1][i]
# wb_statistics.save()
# wb_statistics.close()
#
# for i in range(len(xbest)):
#     ws_convergence.range(row, i + 1).value = f_best[i]
# wb_convergence.save()
# wb_convergence.close()