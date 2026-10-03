import numpy as np
from numpy import linalg as LA


def fn(A):
  eig_vals, eiv_vecs = LA.eig(A)
  return eig_vals, eiv_vecs


def transpose(A):

    AT = [[0] * len(A) for c in range(len(A[0]))]

    for i in range(len(A)):
        for j in range(len(A[i])):
            AT[j][i] = A[i][j]

    return AT


def mat_mul(A, B):

    if len(A[0]) == len(B):

        C = [[0] * len(B[0]) for c in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(A[0])):
                    C[i][j] += A[i][k] * B[k][j]

        return C

    else:
        print('Matrix multiplication is not possible')
        return None


def mat_add(A, B):

    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print('Matrix addition is not possible')
        return None

    return [
        [A[i][j] + B[i][j] for j in range(len(A[0]))]
        for i in range(len(A))
    ]


def SVD(A):

    eig_vals, eig_vecs = fn(mat_mul(transpose(A), A))

    sing_vals = []

    U = [[0] * len(A) for i in range(len(A))]
    Sigma = [[0] * len(A[0]) for i in range(len(A))]
    V = []

    for val in eig_vals:
        sing_vals.append(val ** 0.5)

    order = sorted(
        range(len(sing_vals)),
        key=lambda i: sing_vals[i],
        reverse=True
    )

    for i in range(min(len(A), len(A[0]))):
        Sigma[i][i] = sing_vals[order[i]]

    for i in range(len(sing_vals)):
        V.append(eig_vecs[:, order[i]])

    V = transpose(V)

    AV = mat_mul(A, V)

    for i in range(len(AV)):
        for j in range(min(len(Sigma), len(Sigma[0]))):
            if Sigma[j][j] > 0:
                U[i][j] = AV[i][j] / Sigma[j][j]

    num_known = 0

    for j in range(min(len(A), len(A[0]))):
        if Sigma[j][j] > 0:
            num_known += 1

    current_col = num_known

    for t in range(len(A)):
        if current_col == len(A):
            break

        x = [0] * len(A)
        x[t] = 1

        w = x[:]

        for j in range(current_col):

            dot = 0

            for k in range(len(A)):
                dot += U[k][j] * w[k]

            for k in range(len(A)):
                w[k] -= dot * U[k][j]

        norm = 0

        for k in range(len(A)):
            norm += w[k] ** 2

        norm = norm ** 0.5

        if norm > 0:

            for k in range(len(A)):
                U[k][current_col] = w[k] / norm

            current_col += 1

        if current_col == len(A):
            break
    print(f"A=U*Sigma*V^T \n U = {U} \n Sigma = {Sigma} \n V = {V}")
    return U, Sigma, V


def main():

    text = input('Enter matrix: ').strip()

    if text[:1] != '[' or text[-1:] != ']':
        print('Enter a matrix such as [[2, -1], [-1, 2]].')
        return

    rows = text[1:-1].split('],')
    A = []

    for i in range(len(rows)):
        row_text = rows[i].strip()

        if i == len(rows) - 1:
            if row_text[-1:] != ']':
                print('Each row must be enclosed in brackets.')
                return
            row_text = row_text[:-1].strip()

        if row_text[:1] != '[' or row_text[1:].strip() == '':
            print('Each row must be a nonempty list.')
            return

        values = row_text[1:].split(',')
        row = []

        for value in values:
            row.append(float(value))

        A.append(row)

    m = len(A)
    n = len(A[0])

    for row in A:
        if len(row) != n:
            print('All rows must have the same number of entries.')
            return

    U, Sigma, V = SVD(A)

    report = 'SVD result: A = U * Sigma * V^T\n'
    report += f'Matrix size: {m} x {n}\n'
    report += 'Columns of U: left singular vectors\n'
    report += 'Columns of V: right singular vectors\n'
    report += '\nSingular values (descending order):\n'

    for i in range(min(m, n)):
        value = Sigma[i][i]
        if value.imag == 0:
            value = value.real
        report += f'sigma_{i + 1} = {value:.12f}\n'

    matrices = [A, U, Sigma, V]
    names = ['A', 'U', 'Sigma', 'V']

    for k in range(4):
        report += f'\n{names[k]}:\n'
        for i in range(len(matrices[k])):
            for j in range(len(matrices[k][i])):
                value = matrices[k][i][j]
                if value.imag == 0:
                    value = value.real
                report += f'{value: .10f} '
            report += '\n'

    print('\n' + report)

    filename = input('Enter a txt filename to save (Enter: skip saving): ').strip()

    if filename != '':
        if not filename.endswith('.txt'):
            filename += '.txt'

        file = open(filename, 'x', encoding='utf-8')
        file.write(report)
        file.close()
        print(f'Results saved to {filename}.')


if __name__ == '__main__':
    main()
