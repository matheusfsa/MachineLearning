import numpy as np

def one_point_crossover(sol1, sol2):
    point = np.random.randint(1, sol1.shape[0])
    c1 = np.copy(sol1)
    c2 = np.copy(sol2)
    for i in range(point):
        c1[i] = sol2[i]
        c2[i] = sol1[i]
    return  c1, c2

def mutation(sol, probability=0.2, intervalo=0.2):
    new_sol = np.copy(sol)
    for i in range(new_sol.shape[0]):
        if np.random.rand() < probability:
            new_sol[i] += (-1 ** np.random.randint(0, 1)) * (np.random.rand()*intervalo)
    return new_sol


def sbx_crossover(sol1, sol2, probability=0.9, eps=np.exp(-14), distribution_index=20):
    x1 = sol1.copy()
    x2 = sol2.copy()

    if np.random.rand() < probability:
        for i in range(x1.shape[0]):
            vx1 = x1[i].copy()
            vx2 = x2[i].copy()
            if np.random.rand() < 0.5:
                if np.abs((vx1 - vx2) > eps):
                    if vx1 < vx2:
                        y1, y2 = vx1, vx2
                    else:
                        y1, y2 = vx2, vx1
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (y1 / (y2 - y1)))
                    alpha = 2.0 - beta ** -(distribution_index + 1.0)
                    if rand <= (1.0 / alpha):
                        betaq = np.power(rand * alpha, (1.0 / (distribution_index + 1.0)))
                    else:
                        betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (distribution_index + 1.0))
                    c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                    beta = 1.0 + (2.0 * (y2) / (y2 - y1))
                    alpha = 2.0 - np.power(beta, -(distribution_index + 1.0))
                    if rand <= (1 / alpha):
                        betaq = np.power((rand * alpha), (1.0 / (distribution_index + 1.0)))
                    else:
                        betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (distribution_index + 1.0))
                    c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
                    if np.random.rand() <= 0.5:
                        x1[i] = c2
                        x2[i] = c1
                    else:
                        x1[i] = c1
                        x2[i] = c2
    return x1, x2


def polynomial_mutation(solucao, lower, upper,distribution_index=20):
    mutation_probability = 1 / solucao.shape[0]
    p = solucao.copy()
    for i in range(p.shape[0]):
        if np.random.rand() <= mutation_probability:
            y = p[i]
            rnd = np.random.rand()
            mut_pow = 1.0 / (distribution_index + 1.0)
            yl = lower[i]
            yu = upper[i]
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (distribution_index + 1.0))
                deltaq = val ** (mut_pow - 1.0)
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (distribution_index + 1.0))
                deltaq = 1.0 - (val ** mut_pow)

            y += deltaq * (yu - yl)


            p[i] = y
    return p