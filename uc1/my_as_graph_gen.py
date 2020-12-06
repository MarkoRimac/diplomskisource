"""Generates graphs resembling the Internet Autonomous System network"""

import networkx as nx
import random as rand
import math
from networkx.utils import py_random_state

__all__ = ["my_as_graph_gen"]


def uniform_int_from_avg(a, m, seed):
    """ Pick a random integer with uniform probability.

    Returns a random integer uniformly taken from a distribution with
    minimum value 'a' and average value 'm', X~U(a,b), E[X]=m, X in N where
    b = 2*m - a.

    Notes
    -----
    p = (b-floor(b))/2
    X = X1 + X2; X1~U(a,floor(b)), X2~B(p)
    E[X] = E[X1] + E[X2] = (floor(b)+a)/2 + (b-floor(b))/2 = (b+a)/2 = m
    """

    from math import floor

    assert m >= a
    b = 2 * m - a
    p = (b - floor(b)) / 2
    X1 = int(round(rand.random() * (floor(b) - a) + a))
    if rand.random() < p:
        X2 = 1
    else:
        X2 = 0
    return X1 + X2


def choose_pref_attach(degs, seed):
    """ Pick a random value, with a probability given by its weight.

    Returns a random choice among degs keys, each of which has a
    probability proportional to the corresponding dictionary value.

    Parameters
    ----------
    degs: dictionary
        It contains the possible values (keys) and the corresponding
        probabilities (values)
    seed: random state

    Returns
    -------
    v: object
        A key of degs or None if degs is empty
    """

    if len(degs) == 0:
        return None
    s = sum(degs.values())
    if s == 0:
        return rand.choice(list(degs.keys()))
    v = rand.random() * s

    nodes = list(degs.keys())
    i = 0
    acc = degs[nodes[i]]
    while v > acc:
        i += 1
        acc += degs[nodes[i]]
    return nodes[i]


class MY_as_graph_gen:
    """ Generates random internet AS graphs.
    """

    def __init__(self, nb_regions, nb_core_nodes, nb_gw_per_region, nb_gw_per_region_variance, avg_deg_core_node,
                nb_mm, nb_mm_variance, t_connection_probability, seed):
        """ Initializes variables. Immediate numbers are taken from [1].

        Parameters
        ----------
        n: integer
            Number of graph nodes
        seed: random state
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        GG: AS_graph_generator object

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        """

        self.t_m = t_connection_probability / 100  #  probability M's provider is T, ovako osiguravamo da veza izmedu regija ne bude uvijek
                          #  jednog linka (u slucaju da su nam NR u zasebnim regijama. Za sada smo stavili da su NR u
                          #  svim regijama

        # Moji parametri:
        rand.seed(seed)  # has to be an object.
        self.seed = seed
        self.nb_regions = nb_regions

        self.nb_gw_per_region = nb_gw_per_region
        self.nb_gw_per_region_variance = nb_gw_per_region_variance

        # TODO: Check da broj regija uvijek bude manji ili jednak broju cvorova u core mrezi - Sanity checks..
        self.n_t = int(rand.randint(1, self.nb_regions))  # num of T nodes
        self.nb_core_nodes = nb_core_nodes
        self.n_m = nb_core_nodes - self.n_t  # number of M nodes
        self.avg_deg_core_node = avg_deg_core_node

        self.nb_mm = nb_mm
        self.nb_mm_variance = nb_mm_variance

        # Code static help variables.
        self.reg_count = 0
        self.LoRaWAN_databit_translation = {0: (250, 12), 1: (440, 11), 2: (980, 10), 3: (1760, 9),
                                       4: (3125, 8), 5: (5470, 7), 6: (11000, 7)}

    def t_graph(self):
        """ Generates the core mesh network of tier one nodes of a AS graph.

        Returns
        -------
        G: Networkx Graph
            Core network
        """

        self.G = nx.Graph()
        for i in range(self.n_t):
            self.G.add_node(i, type="T")
            for r in self.regions:
                self.regions[r].add(i)
            for j in self.G.nodes():
                if i != j:
                    self.add_edge(i, j, "peer")
            self.customers[i] = set()
            self.providers[i] = set()
        return self.G

    def add_edge(self, i, j, kind):
        if kind == "transit":
            customer = str(i)
        else:
            customer = "none"
        self.G.add_edge(i, j, type=kind, customer=customer)

    def choose_peer_pref_attach(self, node_list):
        """ Pick a node with a probability weighted by its peer degree.

        Pick a node from node_list with preferential attachment
        computed only on their peer degree
        """

        d = {}
        for n in node_list:
            d[n] = self.G.nodes[n]["peers"]
        return choose_pref_attach(d, self.seed)

    def choose_node_pref_attach(self, node_list):
        """ Pick a node with a probability weighted by its degree.

        Pick a node from node_list with preferential attachment
        computed on their degree
        """

        degs = dict(self.G.degree(node_list))
        return choose_pref_attach(degs, self.seed)

    def add_customer(self, i, j):
        """ Keep the dictionaries 'customers' and 'providers' consistent.
        """

        self.customers[j].add(i)
        self.providers[i].add(j)
        for z in self.providers[j]:
            self.customers[z].add(i)
            self.providers[i].add(z)

    def add_node(self, i, kind, avg_deg, t_edge_prob):
        """ Add a node and its customer transit edges to the graph.

        Parameters
        ----------
        i: object
            Identifier of the new node
        kind: string
            Type of the new node. Options are: 'M' for middle node, 'CP' for
            content provider and 'C' for customer.
        reg2prob: float
            Probability the new node can be in two different regions.
        avg_deg: float
            Average number of transit nodes of which node i is customer.
        t_edge_prob: float
            Probability node i establish a customer transit edge with a tier
            one (T) node

        Returns
        -------
        i: object
            Identifier of the new node
        """

        if kind == "GW":
            regs = 1
        else:
            regs = self.nb_regions  # regions in which node resides will always be max
            """"
            if rand.random() < reg2prob:  # node is in two regions
                regs = 5
            """
        node_options = set()

        self.G.add_node(i, type=kind)

        self.customers[i] = set()
        self.providers[i] = set()
        self.nodes[kind].add(i)
        if kind == "GW":
            r = self.reg_count
            node_options = node_options.union(self.regions["REG" + str(r)])  # Union of regions
            self.regions["REG" + str(r)].add(i)
            self.reg_count = (self.reg_count + 1) % self.nb_regions # u svakoj regiji po reg_count
            self.G.add_node(i, regions="REG" + str(r)) # for debugging add region attribute
        else:
            region_union = ""
            for r in rand.sample(list(self.regions), regs):  # Choose random regs
                region_union = region_union + "REG" + str(r)                                        # For debugging
                node_options = node_options.union(self.regions[r])  # Union of regions
                self.regions[r].add(i)
            #self.G.add_node(i, regions=region_union) # for debugging add region attribute

        edge_num = uniform_int_from_avg(1, avg_deg, self.seed)

        t_options = node_options.intersection(self.nodes["T"])  # Presjek skupova
        m_options = node_options.intersection(self.nodes["M"])
        if i in m_options:
            m_options.remove(i)
        d = 0
        while d < edge_num and (len(t_options) > 0 or len(m_options) > 0):
            if len(m_options) == 0 or (
                len(t_options) > 0 and rand.random() < t_edge_prob
            ):  # add edge to a T node
                j = self.choose_node_pref_attach(t_options)
                t_options.remove(j)
            else:
                j = self.choose_node_pref_attach(m_options)
                m_options.remove(j)
            self.add_edge(i, j, "transit")
            self.add_customer(i, j)
            d += 1

        return i

    def add_m_peering_link(self, m, to_kind):
        """ Add a peering link between two middle tier (M) nodes.

        Target node j is drawn considering a preferential attachment based on
        other M node peering degree.

        Parameters
        ----------
        m: object
            Node identifier
        to_kind: string
            type for target node j (must be always M)

        Returns
        -------
        success: boolean
        """

        # candidates are of type 'M' and are not customers of m
        node_options = self.nodes["M"].difference(self.customers[m])
        # candidates are not providers of m
        node_options = node_options.difference(self.providers[m])
        # remove self
        if m in node_options:
            node_options.remove(m)

        # remove candidates we are already connected to
        for j in self.G.neighbors(m):
            if j in node_options:
                node_options.remove(j)

        if len(node_options) > 0:
            j = self.choose_peer_pref_attach(node_options)
            self.add_edge(m, j, "peer")
            self.G.nodes[m]["peers"] += 1
            self.G.nodes[j]["peers"] += 1
            return True
        else:
            return False

    def add_cp_peering_link(self, cp, to_kind):
        """ Add a peering link to a content provider (CP) node.

        Target node j can be CP or M and it is drawn uniformely among the nodes
        belonging to the same region as cp.

        Parameters
        ----------
        cp: object
            Node identifier
        to_kind: string
            type for target node j (must be M or CP)

        Returns
        -------
        success: boolean
        """

        node_options = set()
        for r in self.regions:  # options include nodes in the same region(s)
            if cp in self.regions[r]:
                node_options = node_options.union(self.regions[r])

        # options are restricted to the indicated kind ('M' or "GW")
        node_options = self.nodes[to_kind].intersection(node_options)

        # remove self
        if cp in node_options:
            node_options.remove(cp)

        # remove nodes that are cp's providers
        node_options = node_options.difference(self.providers[cp])

        # remove nodes we are already connected to
        for j in self.G.neighbors(cp):
            if j in node_options:
                node_options.remove(j)

        if len(node_options) > 0:
            j = rand.sample(node_options, 1)[0]
            self.add_edge(cp, j, "peer")
            self.G.nodes[cp]["peers"] += 1
            self.G.nodes[j]["peers"] += 1
            return True
        else:
            return False

    def graph_regions(self, rn):
        """ Initializes AS network regions.

        Parameters
        ----------
        rn: integer
            Number of regions
        """

        self.regions = {}
        for i in range(rn):
            self.regions["REG" + str(i)] = set()

    def add_peering_links(self, from_kind, to_kind):
        """ Utility function to add peering links among node groups.
        """
        peer_link_method = None
        if from_kind == "M":
            peer_link_method = self.add_m_peering_link
            m = self.p_m_m
        if from_kind == "GW":
            peer_link_method = self.add_cp_peering_link
            if to_kind == "M":
                m = self.p_cp_m
            else:
                m = self.p_cp_cp

        for i in self.nodes[from_kind]:
            num = uniform_int_from_avg(0, m, self.seed)
            for _ in range(num):
                peer_link_method(i, to_kind)

    def add_star_nodes(self, start_index, datarate=5):

        for gw in self.nodes["GW"]:
            nb_star_nodes_rand = abs(int(math.floor(rand.normalvariate(self.nb_mm, self.nb_mm_variance))))
            G_help = nx.generators.star_graph(nb_star_nodes_rand)

            # TODO: check if attributes should be added here.
            for edge in G_help.edges:
                G_help.add_edge(edge[0], edge[1], BW=self.LoRaWAN_databit_translation[datarate][0])

            for node in G_help.nodes:
                if node == 0:
                    continue  # Index 0 will be concatenated to GW
                G_help.add_node(node, type="MM")

            self.G = nx.union(self.G, G_help, (None, gw[2] + 's'))
            self.G = nx.algorithms.contracted_nodes(self.G, gw, gw[2] + 's0')


    def generate(self):
        """ Generates a random AS network graph as described in [1].

        Returns
        -------
        G: Graph object

        Notes
        -----
        The process steps are the following: first we create the core network
        of tier one nodes, then we add the middle tier (M), the content
        provider (CP) and the customer (C) nodes along with their transit edges
        (link i,j means i is customer of j). Finally we add peering links
        between M nodes, between M and CP nodes and between CP node couples.
        For a detailed description of the algorithm, please refer to [1].

        References
        ----------
        [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
        BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
        in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
        """

        self.graph_regions(self.nb_regions)
        self.customers = {}
        self.providers = {}
        self.nodes = {"T": set(), "M": set(), "GW": set(), "C": set(), "MM": set()}

        self.t_graph()
        self.nodes["T"] = set(list(self.G.nodes()))

        i = len(self.nodes["T"])
        for _ in range(self.n_m):
            self.nodes["M"].add(self.add_node(i, "M", self.avg_deg_core_node, self.t_m))
            i += 1
        for l in range(self.nb_regions):
            x = abs(int(math.floor(rand.normalvariate(self.nb_gw_per_region, self.nb_gw_per_region_variance))))
            for _ in range(x):
                self.nodes["GW"].add(self.add_node(i, "GW", self.avg_deg_core_node, self.t_m))
                i += 1

        """
        for _ in range(self.n_c):
            self.nodes["C"].add(self.add_node(i, "C", 0, self.d_c, self.t_c))
            i += 1
        """

        # currently all core nodes (M, T and CP) are created as tho they re all in all 5 groups, no need for peering
        # links. If you do add them later, makes sure to add the necessary probablity values
        """
        self.add_peering_links("M", "M")
        self.add_peering_links("GW", "M")
        self.add_peering_links("GW", "GW")
        """

        # Calculate the shortest distance between CP nodes in the same region.
        shortest_paths_len = dict()
        for region in self.regions:
            for node_src in self.nodes["GW"].intersection(self.regions[region]):
                helps = dict()
                for node_des in self.nodes["GW"].intersection(self.regions[region]):
                    if node_des != node_src:
                        length = nx.algorithms.shortest_path_length(self.G, node_src, node_des)
                        helps.update({node_des: length})
                shortest_paths_len.update({node_src: helps})

        # Add MM devices to gateways
        for node in self.nodes["GW"]:
            x = abs(int(math.floor(rand.normalvariate(self.nb_mm, self.nb_mm_variance))))
            for _ in range(x):
                self.nodes['MM'].add(i)
                self.G.add_node(i, type="MM")
                self.G.add_edge(i, node)

                # Add a connection to a nearby gateway based on the distance provided in shortest_paths_len dictionary.
                for m, n in shortest_paths_len[node].items():
                    connection_probability = 1/pow(2, n)
                    if rand.random() < connection_probability:
                        self.G.add_edge(m, i)
                i += 1

        return self.G



def my_random_internet_as_graph(nb_regions, nb_core_nodes, nb_gw_per_region, nb_gw_per_region_variance, avg_deg_core_node,
                nb_mm, nb_mm_variance, t_connection_probability, seed=None):
    """ Generates a random undirected graph resembling the Internet AS network

    Parameters
    ----------
    n: integer in [1000, 10000]
        Number of graph nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G: Networkx Graph object
        A randomly generated undirected graph

    Notes
    -----
    This algorithm returns an undirected graph resembling the Internet
    Autonomous System (AS) network, it uses the approach by Elmokashfi et al.
    [1] and it grants the properties described in the related paper [1].

    Each node models an autonomous system, with an attribute 'type' specifying
    its kind; tier-1 (T), mid-level (M), customer (C) or content-provider (CP).
    Each edge models an ADV communication link (hence, bidirectional) with
    attributes:
        - type: transit|peer, the kind of commercial agreement between nodes;
        - customer: <node id>, the identifier of the node acting as customer
            ('none' if type is peer).

    References
    ----------
    [1] A. Elmokashfi, A. Kvalbein and C. Dovrolis, "On the Scalability of
    BGP: The Role of Topology Growth," in IEEE Journal on Selected Areas
    in Communications, vol. 28, no. 8, pp. 1250-1261, October 2010.
    """

    GG = MY_as_graph_gen(nb_regions, nb_core_nodes, nb_gw_per_region, nb_gw_per_region_variance, avg_deg_core_node,
                nb_mm, nb_mm_variance, t_connection_probability, seed)
    G = GG.generate()
    return G
