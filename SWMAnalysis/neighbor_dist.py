from scipy.spatial.distance import euclidean
import numpy as np

def neighbor_distances(coords, neighbor_list):
    """Finds the nearest, farthest, and average distance of the points
    in "neighbor_list" to the coordinates given.
    """

    lowest_dist = float("inf")
    highest_dist = 0.0
    dist_sum = 0.0

    for n in neighbor_list:
        d = euclidean(coords, n)
        if d > highest_dist: highest_dist = d
        if d < lowest_dist: lowest_dist = d
        dist_sum += d

    average_dist = dist_sum / len(neighbor_list)

    return lowest_dist, highest_dist, average_dist


def neighbor_distances(coord_list):
    """Gets the average-to-nearest, average-to-furthest, and overall average distance to neighbors"""

    distances = [ [0.0 for i in range(len(coord_list))] for i in range(len(coord_list)) ] 

    #get all the distances (other than on the diagonal, which will stay as zeros)
    for i in range(len(coord_list)):
        for j in range(i+1, len(coord_list)):
            d = euclidean(coord_list[i], coord_list[j])
            distances[i][j] = d
            distances[j][i] = d

    #get the max, min, and average for each point
    max_dists = [ max(distances[i]) for i in range(len(coord_list))]
    #in the following two steps, i'm skipping the diagonals
    min_dists = [ min(distances[i][:i] + distances[i][i+1:]) for i in range(len(coord_list))]
    mean_dists = [ np.mean(distances[i][:i] + distances[i][i+1:]) for i in range(len(coord_list))]

    #get the highest and lowest and average max
    highest_max = max(max_dists)
    lowest_max = min(max_dists)
    ave_max = np.mean(max_dists)

    #get the highest and lowest and average min
    highest_min = max(min_dists)
    lowest_min = min(min_dists)
    ave_min = np.mean(min_dists)

    #get the overal average
    average_dist = np.mean(mean_dists)

    summary={}
    summary["Average Distance"] = average_dist
    summary["Average Furthest"] = ave_max
    summary["Average Nearest"] = ave_min
    summary["Absolute Furthest"] = highest_max
    summary["Absolute Nearest"] = lowest_min
    summary["Shortest Furthest"] = lowest_max
    summary["Furthest Nearest"] = highest_min

    return summary
    

            