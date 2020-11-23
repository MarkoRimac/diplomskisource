import networkx as nx
from random import randint
import random as rand
from math import floor

class Uc1_graph(object):

    # CONSTANTS ( Shortcuts -> DC = Data Center, NR = router (Network Center), GW = gateway, MM = measuring module )
    # NODES - use case1
    DC_NODE_IPT = 10000
    NR_NODE_IPT = 500
    GW_NODE_IPT = 20
    MM_NODE_IPT = 1

    DC_NODE_RAM = 1000
    NR_NODE_RAM = 60 #TODO this parameter is weird
    GW_NODE_RAM = 100
    MM_NODE_RAM = 1

    # EDGES - use case1
    #mBits
    GW_BW = 10
    NR_BW = 1000
    DC_BW = 100000

    GW_PR = 2
    NR_PR = 2
    DC_PR = 2
    MM_PR = 2

    #TYPES
    DC = "DC"
    NR = "NR"
    GW = "GW"
    MM = "MM"

    #LoRaWAN Data Rate to BW and Propagation Time attributes table
    #Spreading factor is used to determin the package loss.
    # TODO: BW is Mb/s and this is in b/s
    LoRaWAN_databit_translation = {0: (250, 12), 1: (440, 11), 2: (980, 10), 3: (1760, 9),
                                   4: (3125, 8), 5: (5470, 7), 6: (11000, 7)}

    def __init__(self, core_node_count, gw_node_count, method, star_node_count, variance, seed, shared_star_nodes_percentage=0):
        self.core_node_count = core_node_count
        self.gw_node_count = gw_node_count
        self.method = method
        self.start_node_count = star_node_count
        self.variance = variance
        self.shared_star_nodes_percentage = shared_star_nodes_percentage
        self.seed = seed
        self.gateways_index = set()
        rand.seed(seed)

        self.__UC1_graph_generation(core_node_count, gw_node_count, method, star_node_count, variance, seed)

    def next(self):
        None,

    def __UC1_graph_generation(self, core_node_count, gw_node_count, method, star_node_count, variance, seed):

        if method == "newman-watts-strogatz":
            self.G = nx.newman_watts_strogatz_graph(core_node_count, randint(0, core_node_count-1), rand.random(), seed)

        if method == "barabasi-albert":
            self.G = nx.barabasi_albert_graph(core_node_count, randint(1, core_node_count-1), seed)

        if method == "erdos-renyi":
            self.G = nx.erdos_renyi_graph(core_node_count, rand.random(), seed)

        if method == "euclidean":
            dimensions = 2
            self.G = nx.random_geometric_graph(core_node_count, dimensions * rand.random(), dim=dimensions, seed=seed)

        # TODO: check if attributes for "IP" edges should be added here.

        self.__pick_random_GW_nodes(core_node_count, gw_node_count)
        self.__add_star_nodes(star_node_count, variance)

    def __add_star_nodes(self, star_node_count, variance, shared_star_nodes_percentage=0, datarate=5):

        for gw in self.gateways_index:
            nb_star_nodes_rand = abs(int(floor(rand.normalvariate(star_node_count, variance))))
            G_help = nx.generators.star_graph(nb_star_nodes_rand)

            # TODO: check if attributes should be added here.
            for edge in G_help.edges:
                G_help.add_edge(edge[0], edge[1], BW=self.LoRaWAN_databit_translation[datarate][0], PR=self.MM_PR)

            for node in G_help.nodes:
                if node == 0:
                    continue # Index 0 will be concatenated to GW
                G_help.add_node(node, IPT=self.MM_NODE_IPT, RAM=self.MM_NODE_RAM, role_affinity=self.MM)

            self.G = nx.union(self.G, G_help, (None, str(gw) + 's'))
            self.G = nx.algorithms.contracted_nodes(self.G, gw, str(gw) + 's0')

    def __pick_random_GW_nodes(self, core_node_count, gw_node_count):
        used_gw = set()
        while gw_node_count:
            gw_index = randint(0, core_node_count-1)
            while gw_index in used_gw:
                gw_index = randint(0, core_node_count-1)
            used_gw.add(gw_index)
            self.G.add_node(gw_index, role_affinity=self.GW)
            self.gateways_index.add(gw_index)
            gw_node_count -= 1