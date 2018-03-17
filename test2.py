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

repeat_times = 10

for i in range(k_num):
    for t in range(repeat_times):
        k = ks[i]
        queries = np.random.choice(n, query_size, False)
        gaussian_matrix = createGaussianMatrix(k, d)
        fly_matrix = createFlyMatrix(10*d, d, p)
        
        gaussian[i] += mAP(data, queries, gaussian_matrix, k)
        fly[i] += mAP(data, queries, fly_matrix, k, 'top')
        
    gaussian[i] /= repeat_times
    fly[i] /= repeat_times
    
    # print('k = ', k)
    # print('gaussian : ', gaussian[i])
    # print('fly : ', fly[i])

with open('result2.txt', 'w') as f:
    f.write('k gaussian fly\n')
    for i in range(k_num):
        f.write('{} {}\n'.format(
            ks[i], gaussian[i], fly[i]))

index = np.arange(k_num)
bar_width = 0.35
gaussian_bar = plt.bar(index, gaussian, width = bar_width, 
							label='gaussian',tick_label = ks, fc = 'y')
fly_bar = plt.bar(index+bar_width, fly, width = bar_width, label='fly',fc = 'r')  
plt.legend()
plt.savefig('result2.png')