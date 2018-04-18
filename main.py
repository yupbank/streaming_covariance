def _online_covariance(data1, data2):
    meanx = meany = c = 0.0
    for n, (x, y) in enumerate(zip(data1, data2)):
        dx = x - meanx
        dy = y - meany
        meanx = meanx + dx/n
        meany = meany + dy/n
        c = c + dx*dy
    return c, n, meanx, meany


def online_covariance(data1, data2):
    c, n, _, _ = _online_covariance(data1, data2)
    return c/n


def parallel_online_covariance(data1_1, data2_1, data1_2, data2_2):
    ca, na, mean_x_a, mean_y_a = _online_covariance(data1_1, data2_1)
    cb, nb, mean_x_b, mean_y_b = _online_covariance(data1_2, data2_2)
    c = ca+cb+(mean_x_a - mean_x_b)(mean_y_a - mean_y_b)*(na*nb)/(na+nb)
    n = na+nb
    mean_x = (mean_x_a*na+mean_x_b*nb)/n
    mean_y = (mean_y_a*na+mean_y_b*nb)/n
    return c, n, mean_x, mean_y


def _vectorized_online_covariance(data):
    mean = np.zeros(data.shape[1])
    c = np.zeros((data.shape[1], data.shape[1]))
    for n, d in enumerate(data):
        diff = d - mean
        mean = mean + diff/n
        c = c + diff.T.dot(diff)
    return c, n, mean


def vectorized_online_covariance(data):
    c, n, mean = _vectorized_online_covariance(data)
    return c/n


def vectorized_parallel_online_covariance(data_a, data_b):
    c_a, n_a, mean_a = _vectorized_online_covariance(data_a)
    c_b, n_b, mean_b = _vectorized_online_covariance(data_b)
    c = c_a + c_b + (mean_a-mean_b).T.dot(mean_a-mean_b) * n_a*n_b/(n_a+n_b)
    n = n_a+n_b
    mean = (mean_a*n_a + mean_b*n_b)/(n_a+n_b)
    return c, n, mean
