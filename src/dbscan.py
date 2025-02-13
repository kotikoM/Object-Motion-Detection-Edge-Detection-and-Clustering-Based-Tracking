import numpy as np
from collections import deque


def dbscan(X, eps, min_samples):
    n_points = X.shape[0]
    labels = -2 * np.ones(n_points, dtype=int)  # -2 means unvisited
    cluster_id = 0

    # Function to find neighbors within 'eps' distance
    def region_query(point_idx):
        point = X[point_idx]
        return np.where(np.linalg.norm(X - point, axis=1) < eps)[0]

    # Function to expand the cluster
    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()
            if labels[neighbor_idx] == -2:  # Point has not been visited
                labels[neighbor_idx] = cluster_id
                neighbor_neighbors = region_query(neighbor_idx)
                if len(neighbor_neighbors) >= min_samples:
                    queue.extend(neighbor_neighbors)
            elif labels[neighbor_idx] == -1:  # Point was marked as noise
                labels[neighbor_idx] = cluster_id

    # DBSCAN algorithm
    for point_idx in range(n_points):
        if labels[point_idx] != -2:
            continue  # Skip if already processed or marked as noise

        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Mark as noise
        else:
            expand_cluster(point_idx, neighbors)
            cluster_id += 1

    return labels
