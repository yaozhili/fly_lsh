from fly_hash import * 
import matplotlib.pyplot as plt

data = preprocess(load_mnist())
# d: dimension of vector, k: hash length, p: probability of 1 in fly hash matrix
n, d = data.shape[0], data.shape[1]
query_size = 1000
p = 0.1

ks = np.array([2, 4, 8, 12, 16, 20, 24, 28, 32, 40])
k_num = ks.shape[0]

gaussian = np.zeros(k_num)
fly = np.zeros(k_num)
fly_expansion = np.zeros(k_num)
fly_WTA = np.zeros(k_num)
fly_binary = np.zeros(k_num)

repeat_times = 10

for i in range(k_num):
    for t in range(repeat_times):
        k = ks[i]
        queries = np.random.choice(n, query_size, False)
        gaussian_matrix = createGaussianMatrix(k, d)
        fly_matrix = createFlyMatrix(k, d, p)
        expan_mat = createFlyMatrix(20*k, d, p)
        
        # mAP(data, queries, proj_matrix, hash_length, WTA = 'all', knn_size = 200)
        gaussian[i] += mAP(data, queries, gaussian_matrix, k)
        fly[i] += mAP(data, queries, fly_matrix, k)
        fly_expansion[i] += mAP(data, queries, expan_mat, k, 'random')
        fly_WTA[i] += mAP(data, queries, expan_mat, k, 'top')
        fly_binary[i] += mAP(data, queries, expan_mat, k, 'binary')
        
    gaussian[i] /= repeat_times
    fly[i] /= repeat_times
    fly_expansion[i] /= repeat_times
    fly_WTA[i] /= repeat_times
    fly_binary[i] /= repeat_times
    
#     print('k = ', k)
#     print('gaussian : ', gaussian[i])
#     print('fly : ', fly[i])
#     print('fly_expansion : ', fly_expansion[i])
#     print('fly_WTA : ', fly_WTA[i])
#     print('fly_binary : ', fly_binary[i])

with open('result1.txt', 'w') as f:
    f.write('k gaussian fly fly_expansion fly_WTA fly_binary\n')
    for i in range(k_num):
        f.write('{} {} {} {} {} {}\n'.format(
            ks[i], gaussian[i], fly[i], fly_expansion[i], fly_WTA[i], fly_binary[i]))


plt.figure(figsize=(16,9)) 
plt.title('mAP')
gaussian_plot, = plt.plot(ks, gaussian, label='gaussian')
fly_plot, = plt.plot(ks, fly, label='fly')
expansion_plot, = plt.plot(ks, fly_expansion, label='expansion')
WTA_plot, = plt.plot(ks, fly_WTA, label='WTA')
binary_plot, = plt.plot(ks, fly_binary, label='binary')
plt.legend(handles=[gaussian_plot, fly_plot, expansion_plot, WTA_plot, binary_plot])
plt.savefig('result1.png')