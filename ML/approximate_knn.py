from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:

    distances = np.linalg.norm(documents - pointA, axis=1)
    return distances.reshape(-1, 1)



def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:

    N = data.shape[0]

    graph = dict(zip(np.arange(N), [{'far_edges':[], 'close_edges':[]} for _ in range(N)]))

    for i in range(N):
        current_point = data[i]
        distances = dist_f(current_point, data)
        dists_for_short = distances.copy()
        dists_for_short[i] = np.inf
        far_points = np.argsort(distances.flatten())[-num_candidates_for_choice_long:].flatten()
        far_edges = np.random.choice(far_points, max(0, num_edges_long - len(graph[i]['far_edges'])), replace=False).tolist()

        close_points = np.argsort(dists_for_short.flatten())[:num_candidates_for_choice_short].flatten()
        close_edges = np.random.choice(close_points, max(0, num_edges_short - len(graph[i]['close_edges'])),replace=False).tolist()

        graph[i]['far_edges'].extend(far_edges)
        graph[i]['close_edges'].extend(close_edges)


        for edge in far_edges:
            graph[edge]['far_edges'].append(i)

        
        for edge in close_edges:
            graph[edge]['close_edges'].append(i)
            
    for i in range(N):
        graph[i] = graph[i]['far_edges'] + graph[i]['close_edges']

    return graph


def nsw(query_point: np.ndarray,
        all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, 
        num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    
    points = [] # в ответ
    points_dists = np.array([np.inf for _ in range(len(graph_edges))]).flatten() # array с расстояними, если меньше np.inf => мы там были
    start_points = np.random.choice(np.arange(len(graph_edges)), size=num_start_points, replace=False)
    current_points = start_points
    
    
    while len(points) < search_k and len(points) < len(graph_edges) and len(current_points) != 0:

        temp_points = []
        for point in current_points:
            
            neighbours = np.array(graph_edges[point])
            neighbours = np.array([n for n in neighbours if points_dists[n]==np.inf]).flatten()
            if len(neighbours) == 0:
                continue
            neighbours_dists = dist_f(query_point.reshape(1, -1), all_documents[neighbours]).flatten()
            
            
            for i in range(len(neighbours)):
                points_dists[neighbours[i]] = neighbours_dists[i]
            
            self_dist = dist_f(query_point.reshape(1, -1), all_documents[point]).flatten()

            if np.all(self_dist < neighbours_dists):
                points.append(point)
            
            else: 
                
                to_add = [neighbours[i] for i in range(len(neighbours_dists)) if neighbours_dists[i] < self_dist] # добавляем соседей для обхода в следующей итерации 
                temp_points.extend(to_add)
        
        current_points = temp_points.copy()
    return np.argsort(points_dists)[:search_k]
