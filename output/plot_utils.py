import numpy as np


def smoothen(data, bucket_size):
    smoothening_factor = bucket_size
    dim = data.shape[0]
    num_buckets = dim//smoothening_factor

    smoothed_data = data
    nb = dim
    for sf_t in range(1, smoothening_factor+1, 10):
        d = nb
        sf =  sf_t//(dim//d)
        if sf > 1:
            nb = d//sf
            smoothening_matrix = []
            for i in range(nb):
                smoothening_matrix.append([])
                for j in range(d):
                    if j >= i*sf and j < (i+1)*sf:
                        smoothening_matrix[-1].append(1)
                    else:
                        smoothening_matrix[-1].append(0)
            smoothening_matrix = np.array(smoothening_matrix)/sf
            smoothed_data = smoothening_matrix.dot(smoothed_data)
    
    buckets = (dim//len(smoothed_data))*np.array(list(range(len(smoothed_data))))

    return smoothed_data, buckets
