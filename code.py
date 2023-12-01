import networkx as nx
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def construct_graph(graph_data):
    """
    To construct the graph with the input graph data about nodes

    Parameters:
        graph_data: the file path of the provided graph data

    Returns:
        Graph constructed with provided graph_data
    """
    g = nx.Graph()
    graph_data = open(graph_data, "r")
    for edge in graph_data.readlines():
        if edge:
            edge = edge.strip()
            nodes = edge.split(" ")
            node1 = int(nodes[0])
            node2 = int(nodes[1])
            g.add_edge(node1, node2)
    graph_data.close()
    return g


def shortest_path(graph, source):
    """
    To calculate all nodes' sigma value 
        -- the number of shortest paths pass through that node
        from the source node

    Parameters:
        graph: The graph
        source: The source node implementing BFS on

    Returns:
        sigma: The dict of sigma value of each node
        stack: The list of single source shortest path with visited vertices
        predecessors: The dict of lists of 
            predecessor (previous neighbor) of each node

    """

    stack = []

    # A dict to document the predecessors of each node for later check
    predecessors = {}

    # Initialize the predecessor of nodes
    for vertice in graph:
        predecessors[vertice] = []

    # Initialize sigma value for each node
    sigma = {vertice: 0 for vertice in graph}

    # A dict to document the distance from source to each node
    distance = {}

    # start from source node
    distance[source] = 0

    sigma[source] = 1

    # The queue of nodes waiting for being visited
    # visit start from source node
    queue = [source]

    # use BFS to find shortest paths of each node
    while queue:
        # first-in-first-out strategy in BFS
        vertice = queue.pop(0)
        # document the visit path in stack
        stack.append(vertice)

        # visit each node's neighbor
        for neighbor in nx.to_dict_of_lists(graph)[vertice]:

            # neighbor is not visted
            if neighbor not in distance:
                queue.append(neighbor)
                # the distance from source node to 'neighbor' node
                distance[neighbor] = distance[vertice] + 1

            # vertice is a predecessor of neighbor on the shortest path 
            #   from source node to neighbor
            if distance[neighbor] == distance[vertice] + 1:
                # sigma of a node is the sum of its predecessors' sigma
                sigma[neighbor] += sigma[vertice]
                predecessors[neighbor].append(vertice)

    return sigma, stack, predecessors


def betweenness_centrality(graph):
    """
    To calculate the betweenness centrality of each node in the graph 
    based on Brandes Algorithm

    To calculate the delta
    (dependence of the starting node source to each node on the shortest path)
    based on Brandes Algorithm's backward step

    Parameters:
        graph: The graph

    Returns:
        centrality: The dict of betweenness centrality of each node
    """

    # Initialize the betweenness centrality result
    centrality = {node: 0 for node in graph}

    # Iterate over all vertices in graph
    # to calculate the sigma and delta
    # and use those sigmas and deltas to calculate the betweenness centrality
    for source in graph:
        sigma, stack, predecessors = shortest_path(graph, source)

        # Initialize delta value for each node on shortest path
        delta = {vertice: 0 for vertice in stack}

        while stack:
            # Backward step accumulation
            neighbor = stack.pop()

            for vertice in predecessors[neighbor]:
                # According to Brandes Algorithm
                delta[vertice] += ((sigma[vertice] / sigma[neighbor]) * (1 + delta[neighbor]))
            # According to Brandes Algorithm
            if neighbor != source:
                centrality[neighbor] += delta[neighbor]

    return centrality


def pagerank_centrality(graph, alpha, beta):
    """
    Use Power Iteration to calculate the Pagerank Centrality

    1) Create adjacency matrix with zeros
    2) Populate the matrix with entries = 1
    3) Power Iteration about new Pagerank Centrality matrix calculated based on
    the previous c and neighbors' outgoing links, including alpha and beta with
    formula provided in lecture notes
    4) When the absolute value of two continuously c larger than eps tolerance
    do iteration

    Parameters:
        graph: The graph
        alpha: damping factor
        beta: teleportation probability

    Returns:
        pagerank_centrality

    """
    # Construct adjacency matrix
    n = len(graph)
    nodes = sorted(list(graph.keys()))

    # Initialize the adjacency matrix with zeros
    A = np.zeros((n, n))

    # A=A^T in undirected graph
    A = nx.to_dict_of_dicts(graph)

    # Construct degree matrix by adjacency matrix
    D = np.diag(A.sum(axis= 1))

    # Construct inverse of degree matrix
    Di = inv(D)

    # Use Power Iteration to calculate pagerank
    # tolerance
    eps = 1e-4
    # Initialize previous pagerank
    pc_previous = np.zeros(n)

    pc = np.ones(n) /n

    while np.sum(np.abs(pc_previous - pc)) > eps:
        pc_previous = pc
        pc = alpha * A.T @ Di @ pc + beta * np.ones(n)
        pc = pc/np.sum(np.abs(pc))




    pass


def main():
    """
    main function
    """
    # Construct graph
    g = construct_graph("3. data.txt")
    # g = construct_graph("test_data.txt")

    # print top 10 betweenness centrality node
    sorted_bc = sorted(betweenness_centrality(g), key=betweenness_centrality(g).get, reverse=True)
    print(sorted_bc[:10])


    # # -------- nx
    # print("nx:", nx.betweenness_centrality(g, normalized=False))
    # sortnx = sorted(nx.betweenness_centrality(g, normalized=False), key=nx.betweenness_centrality(g).get, reverse=True)
    # print(sortnx)


if __name__ == "__main__":
    main()

