import numpy as np
import matplotlib.pyplot as plt

def initialize_cluster_centers(data, num_clusters):
    #randomly initialize cluster centers
    indices = np.random.choice(len(data), num_clusters, replace=False)
    return data[indices]

def update_membership_matrix(data, centers, fuzziness):
    num_data_points = len(data)
    num_clusters = len(centers)
    membership_matrix = np.zeros((num_data_points, num_clusters))

    for i in range(num_data_points):
        #update the membership value for the closest cluster
        closest_cluster = np.argmin(np.linalg.norm(centers - data[i], axis=1))
        membership_matrix[i, closest_cluster] = 1.0

    return membership_matrix

def update_cluster_centers(data, membership_matrix, fuzziness):
    num_clusters = membership_matrix.shape[1]
    cluster_centers = np.zeros((num_clusters, data.shape[1]))

    for j in range(num_clusters):
        #updtae the cluster values based on the mebership value
        u_jm = membership_matrix[:, j] ** fuzziness
        cluster_centers[j] = np.sum(u_jm[:, np.newaxis] * data, axis=0) / np.sum(u_jm)

    return cluster_centers

def fuzzy_c_means(data, num_clusters, fuzziness, num_random_initializations=1, max_iterations=100, tolerance=1e-4):
    best_centers = None
    best_membership_matrix = None
    best_error = float('inf')

    for _ in range(num_random_initializations):
        #initalize the cluster centers randomly
        centers = initialize_cluster_centers(data, num_clusters)

        for _ in range(max_iterations):
            old_centers = centers.copy()

            #For each iteration updtae the cluster centers and membership matix
            membership_matrix = update_membership_matrix(data, centers, fuzziness)
            centers = update_cluster_centers(data, membership_matrix, fuzziness)

            if np.linalg.norm(centers - old_centers) < tolerance:
                break

        #calculate the error
        error = np.sum((membership_matrix * fuzziness) * np.linalg.norm(data[:, np.newaxis] - centers, axis=2) * 2)

        if error < best_error:
            best_error = error
            best_centers = centers
            best_membership_matrix = membership_matrix

    return best_centers, best_membership_matrix, best_error

def plot_clusters(data, centers, membership_matrix):
    #plot the clusters with the cluster centers
    clusters = np.argmax(membership_matrix, axis=1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', c='red', label='Cluster centers')
    plt.show()

#load the dataset
def load_gaussian_dataset(file_path):
    dataset_list = []

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            values = line.split()
            dataset_list.append(list(map(float, values)))

    return np.array(dataset_list)

dataset_file_path = '/content/545_cluster_dataset programming 3.txt'
data = load_gaussian_dataset(dataset_file_path)
#initialize the number of clusters and fuzziness value
num_clusters_value = 3 
fuzziness_value = 2  
random_initializations_value = 1  

centers, membership_matrix, error = fuzzy_c_means(data, num_clusters_value, fuzziness_value, num_random_initializations=random_initializations_value)

plot_clusters(data, centers, membership_matrix)
print(f'Sum Squares Error: {error}')
