import networkx as nx
import argparse
import json
from my_as_graph_gen import my_random_internet_as_graph

def main(data):
    """
    TOPOLOGY creation
    """
    G = my_random_internet_as_graph(data.nb_regions, data.nb_core_nodes, data.nb_gw_per_region, data.nb_gw_per_region_variance, data.avg_deg_core_node, data.nb_mm, data.nb_mm_variance, data.t_connection_probability, data.seed)
    #graph = Uc1_graph(args.core_node_count, args.gw_node_count, args.method, args.star_node_count, args.variance, args.seed)
    nx.write_gexf(G, data.outFILE + '.gexf')


def check_bigger_than_zero(value):
    ival = int(value)
    if ival < 1:
        raise argparse.ArgumentTypeError("Has to be greater than 0")
    return ival

def check_positive_and_zero(value):
    ival = int(value)
    if ival < 0:
        raise argparse.ArgumentTypeError("Has to be positive!")
    return ival

def get_and_check_args(data):
    parser.add_argument("outFILE",
                        type=str,
                        help="output file name for generated graph")
    parser.add_argument("nb_regions", type=check_bigger_than_zero,
                        help="Number of regions")
    parser.add_argument("nb_core_nodes", type=check_bigger_than_zero, help="Number of core nodes")
    parser.add_argument("nb_gw_per_region", type=check_positive_and_zero, help="Number of gateways per region")
    parser.add_argument("nb_gw_per_region_variance", type=check_positive_and_zero, help="Number of gateways per region variance")
    parser.add_argument("avg_deg_core_node", type=check_bigger_than_zero, help="average degree of a core node, Pick a random integer with uniform probability.")
    parser.add_argument("nb_mm", type=check_positive_and_zero, help="Number of measuring modules")
    parser.add_argument("nb_mm_variance", type=check_positive_and_zero, help="variance for normal distribution for number of measuring modules")
    parser.add_argument("t_connection_probability", type=check_positive_and_zero, help="probability of m connections to T nodes")
    parser.add_argument("seed", type=check_positive_and_zero, help="set seed for random values")

    x = list(data.values())
    arguments = parser.parse_args(x)

    # Ovisnosti medu argumentima
    if (arguments.nb_gw_per_region + arguments.nb_gw_per_region_variance * 3) * arguments.nb_regions > arguments.nb_core_nodes - arguments.nb_regions:
        raise argparse.ArgumentTypeError("Please make sure that max possible number of gateways (considering 3sigma normal distribution with a variance) be less than a total nb_core_nodes! ")
    return arguments


parser = argparse.ArgumentParser()

if __name__ == '__main__':
    data = json.load(open('topology_input.json'))
    args = get_and_check_args(data)
    main(args)


