import numpy as np
import matplotlib.pyplot as plt

# K-Means clustering function
def kmeans(itr, k, data):
    for t in range(itr):
        #randomly initialize cluster centers
        centers = data[np.random.choice(data.shape[0], k, replace=False)]  
        old_centers = np.zeros_like(centers)
        clusters = np.zeros(len(data))
        
        #run K-means for a set number of iterations
        for iteration in range(1, 11):  
            old_centers = np.copy(centers)

            #calculate the distance from the cluster center and assign to a cluster
            for i, point in enumerate(data):
                clusters[i] = np.argmin(np.linalg.norm(centers - point, axis=1))

            #update the cluster centroids
            for j in range(k):
                if len(data[clusters == j]) > 0:
                    centers[j] = np.mean(data[clusters == j], axis=0)

            if np.array_equal(old_centers, centers):
                break

        #plot the clusters
        plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', label='Cluster Centers')
        plt.title(f'K-Means Clustering (K={k}) - Run {t + 1}')
        plt.legend()
        plt.show()

        #calculate the sum square errors
        sse = np.sum([np.sum((data[clusters == i] - centers[i]) ** 2) for i in range(k)])
        print(f"Sum Squares Error: {sse}")

#load the dataset
dataset = np.loadtxt('/content/545_cluster_dataset programming 3.txt')  

#initalize the number of iterations and number of clusters
iterations = 10
k = 7 
kmeans(iterations, k, dataset)