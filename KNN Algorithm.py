# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:49:13 2023

@author: EGabriele
"""
from collections import Counter
import math

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    for index, example in enumerate(data):
        distance = distance_fn(example[:-1], query)
        neighbor_distances_and_indices.append((distance, index))
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

def mean(labels):
    return sum(labels) / len(labels)

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def main():
      
    spread_gen_dem = [5494,4942,5017,4601,5530,4163,4551,5423,5099,5023]
    wind_gen = [2526,3478,7140,5661,8066,4970,4730,9942,10836,7433]
    gas_DA_price = [64,68,74.275,72,72.1,73,92.5,85]
    N2EX_Price = [73.91,45.53,69.43,59.11,79.26,77.62,62.4,71.63,72.65,77.44,80.63]

    reg_data = []
    for x in spread_gen_dem:
        for y in wind_gen:
            for z in gas_DA_price:
                for p in N2EX_Price:       
                    reg_data.append((x,y,z,p))
 
    # Question:
    # Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?
    num_neighbours = int(input("Enter number K: "))
    
    for k in range(1,num_neighbours):
        reg_query = [5000,8000,72]
        reg_k_nearest_neighbors, reg_prediction = knn(
            reg_data, reg_query, k, distance_fn=euclidean_distance, choice_fn=mean)
        print(reg_prediction)
   
if __name__ == '__main__':
    main()