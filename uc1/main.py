from uc1_graph import Uc1_graph
import networkx as nx
import argparse

def main(args):
    """
    TOPOLOGY creation
    """
    graph = Uc1_graph(args.core_node_count, args.gw_node_count, args.method, args.star_node_count, args.variance, args.seed)
    nx.write_gexf(graph.G, args.outFILE + '.gexf')


def check_bigger_than_three(value):
    ival = int(value)
    if ival < 3:
        raise argparse.ArgumentTypeError("Number of nodes in core part has to be at least 3." +
                                         "That is, it has to be one DC one NR and one GW")
    return ival

def check_bigger_than_one(value):
    ival = int(value)
    if ival < 1:
        raise argparse.ArgumentTypeError("Number of GW nodes has to be greater than 1")
    return ival

def check_possible_methods(method):
    if method not in core_graph_methods:
        raise argparse.ArgumentTypeError("Not valid method! Valid methods are: " + str(core_graph_methods))
    return method

def get_and_check_args():
    parser.add_argument("outFILE",
                        type=str,
                        help="output file name for generated graph")
    parser.add_argument("core_node_count", type=check_bigger_than_three,
                              help="Number of nodes in core part of topology")
    parser.add_argument("gw_node_count", type=check_bigger_than_one, help="Number of GW nodes")
    parser.add_argument("method", type=check_possible_methods, help="Name of method used to generate core function, possible:" + str(core_graph_methods))
    parser.add_argument("star_node_count", type=check_bigger_than_one, help="middle val for normal distribution")
    parser.add_argument("variance", type=int, help="variance for normal distribution")
    parser.add_argument("seed", type=int, help="set seed for random values")

    arguments = parser.parse_args()

    # Post parser conditions. Conditions between args checked.
    if arguments.gw_node_count > arguments.core_node_count - 2:
        raise argparse.ArgumentTypeError("Please make sure that: nb_gw <= nb_core_nodes - 2"
                                         "always have to exist ")
    return arguments


parser = argparse.ArgumentParser()
core_graph_methods = ("newman-watts-strogatz", "barabasi-albert", "erdos-renyi", "euclidean")

if __name__ == '__main__':
    args = get_and_check_args()
    main(args)


