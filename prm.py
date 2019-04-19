import numpy as np
import pylab as pl
import sys
import scipy.spatial
import environment_2d
import networkx as nx


class ProbablisticRoadmap(object):
    def __init__(self, env):
        """
        This class generates a probabilistic roadmap sampled from the environment.
        It maintains a kd-tree of sampled points for finding k-nearest neighbors
        and an undirected graph for searching.
        :param env: reference to environment object for collision checking of points.
        """
        self.N_SAMPLE = 500
        self.N_KNN = 10
        self.N_LOCAL = 30

        self.env = env
        self.kdtree = None
        self.graph = None

        self._generate_roadmap()

    def query(self, source, dest):
        """
        This method uses Dijkstra algorithm to find the shortest path between source and dest
        based on the L2 distance between each node.
        :param source: (x_start, y_start) Starting coordinate
        :param dest: (x_goal, y_goal) Goal coordinate
        :return: path: array of tuple. Returns array of points of the path. Return None if no path was found.
        """
        nearest_node_to_source = self._nearest_node(source)
        nearest_node_to_dest = self._nearest_node(dest)
        path = None
        if nearest_node_to_source is not None and nearest_node_to_dest is not None:
            try:
                path_in_graph = nx.shortest_path(self.graph, source=nearest_node_to_source, target=nearest_node_to_dest)
                path = [source] + path_in_graph + [dest]
            except:
                print("No path was found")
        return path

    def _nearest_node(self, point):
        distances, indexes = self.kdtree.query(np.array(point), k=self.N_KNN)
        for distance, index in zip(distances, indexes):
            neighbor = self.kdtree.data[index]
            if tuple(neighbor) in self.graph.nodes and self._local_planner(point, neighbor):
                return tuple(neighbor)
        return None

    def _local_planner(self, a, b, n=30):
        for i in range(n):
            x_i = a[0] + i * (b[0] - a[0]) / n
            y_i = a[1] + i * (b[1] - a[1]) / n
            if self.env.check_collision(x_i, y_i):
                return False
        return True

    def _generate_roadmap(self):
        nodes = []
        while len(nodes) < self.N_SAMPLE:
            x, y = np.random.rand() * 10, np.random.rand() * 6
            if not env.check_collision(x, y):
                nodes.append([x, y])
        nodes = np.array(nodes)

        self.kdtree = scipy.spatial.cKDTree(nodes)
        self.graph = nx.Graph()
        for node in nodes:
            distances, indexes = self.kdtree.query(node, k=self.N_KNN)
            for distance, index in zip(distances, indexes):
                neighbor = self.kdtree.data[index]
                if self._local_planner(node, neighbor):
                    self.graph.add_edge(tuple(node), tuple(neighbor), weight=distance)


if __name__ == '__main__':
    pl.ion()
    pl.clf()
    # np.random.seed(4)

    # Initialize environment
    env = environment_2d.Environment(10, 6, 5)
    env.plot()

    # Build probabilistic road map
    prm = ProbablisticRoadmap(env)

    # Plot the nodes in graph
    nodes = prm.graph.nodes
    pl.scatter([node[0] for node in nodes], [node[1] for node in nodes], s=3)

    # Run multiple queries
    for i in range(2):
        q = env.random_query()
        env.plot_query(*q)
        if q is None: exit(0)
        x_start, y_start, x_goal, y_goal = q
        path = prm.query((x_start, y_start), (x_goal, y_goal))

        # Plot the paths
        if path is not None:
            pl.plot([node[0] for node in path], [node[1] for node in path], "k", linewidth=1.2)
            c = input("PRESS ENTER TO CONTINUE.")