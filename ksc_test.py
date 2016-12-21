# coding = 'utf-8'

from numpy import *
from numpy.linalg import norm, linalg
import matplotlib.pyplot as plt


# dist of time series
def dist_ts(x, y):
    alpha = dot(transpose(x), y) / sum(y ** 2)
    return norm(x - alpha * y) / norm(x)


def dist_ts_shift(x, y):
    dist_min = dist_ts(x, y)
    opt_shift = 0
    opt_y = y
    for i in range(-5, 6):
        if i != 0:
            if i < 0:
                y_shift = append(y[-i:], zeros((1, -i)))
            elif i > 0:
                y_shift = append(zeros((1, i)), y[:-i])
            cur_dist = dist_ts(x, y_shift)
            if cur_dist < dist_min:
                dist_min = cur_dist
                opt_shift = i
                opt_y = y_shift
    return dist_min, opt_y, opt_shift


def ksc_center(X, C, k, center):
    [m, d] = X.shape
    clusters = empty((0, d))
    for j in range(0, m):
        if int(C[j]) == k:
            if sum(center) == 0:
                opt_xi = X[j]
            else:
                [tmp, opt_xi, tmps] = dist_ts_shift(center, X[j])
            opt_xi = opt_xi.reshape((1, d))
            clusters = append(clusters, opt_xi, axis=0)
        cluster_member_num = clusters.shape[0]
    if cluster_member_num == 0:
        return zeros((1, d))
    else:
        cluster_sample_norms = sum(clusters ** 2, axis=-1) ** 1./2
        cluster_sample_norms = cluster_sample_norms.reshape([cluster_member_num, 1])  # transpose doesn't work (one dimension)
        cluster_sample_norms = tile(cluster_sample_norms, d)
        clusters = divide(clusters, cluster_sample_norms)

        M = cluster_member_num * identity(d) - dot(transpose(clusters), clusters)

        eig_val, eig_vec = linalg.eig(M)
        idx = eig_val.argsort()
        eig_vec = eig_vec[:, idx]
        eig_vec = eig_vec[:, 0].reshape([1, d])
        if sum(eig_vec) < 0:
            eig_vec = -eig_vec
        return eig_vec


def ksc_toy(X, K):
    iter_num = 100
    [m, d] = X.shape
    C = floor(random.rand(m, 1) * K)
    centroid = zeros((K, d))
    D = zeros((m, K))
    for i in range(iter_num):
        if i % 10 == 0:
            print 'iter:', i
        prev_C = C
        for k in range(K):
            centroid[k, :] = ksc_center(X, C, k, centroid[k, :])

        for j in range(m):
            xj = X[j]
            for k in range(K):
                center = centroid[k, :]
                [D[j, k], tmp, tmps] = dist_ts_shift(center, xj)
        C = argmin(D, axis=1)

        if sum(abs(C - prev_C)) == 0:
            break
    return C, centroid


if __name__ == '__main__':
    X = genfromtxt('time_series_data.tsv', delimiter='\t')
    K = 5
    [C, centroid] = ksc_toy(X, K)

    f, fig = plt.subplots(1, 5)
    for i in range(K):
        c = centroid[i]
        fig[i].plot(c)
    plt.show()
