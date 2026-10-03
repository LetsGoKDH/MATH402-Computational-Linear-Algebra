from SVD import SVD


A = [
    [2, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 2]
]

U, Sigma, V = SVD(A)

a = ((5 - 5 ** 0.5) / 20) ** 0.5
b = ((5 + 5 ** 0.5) / 20) ** 0.5

Q = [
    [a, b, b, a],
    [-b, -a, a, b],
    [b, -a, -a, b],
    [-a, b, -b, a]
]

Sigma_exact = [
    [(5 + 5 ** 0.5) / 2, 0, 0, 0],
    [0, (3 + 5 ** 0.5) / 2, 0, 0],
    [0, 0, (5 - 5 ** 0.5) / 2, 0],
    [0, 0, 0, (3 - 5 ** 0.5) / 2]
]

print('\nExact singular values:')
print('(5 + sqrt(5))/2, (3 + sqrt(5))/2, (5 - sqrt(5))/2, (3 - sqrt(5))/2')
print('a = sqrt((5 - sqrt(5))/20), b = sqrt((5 + sqrt(5))/20)')

print('\nTheoretical U = V = Q from (a):')
for row in Q:
    print(row)

print('\nTheoretical Sigma from (a):')
for row in Sigma_exact:
    print(row)

for j in range(4):
    dot = 0

    for i in range(4):
        dot += V[i][j] * Q[i][j]

    if dot.real < 0:
        for i in range(4):
            U[i][j] = -U[i][j]
            V[i][j] = -V[i][j]

print('\nComputed U after aligning column signs:')
for row in U:
    print(row)

print('\nComputed V after aligning column signs:')
for row in V:
    print(row)
