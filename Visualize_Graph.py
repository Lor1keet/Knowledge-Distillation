import networkx as nx
import matplotlib.pyplot as plt
from CARProblemDef import generate_graph_degree

def visualize_graph(G):
    pos = nx.spring_layout(G)  
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)    
    # 显示图
    plt.show()

if __name__ == "__main__":
    
    distribution = {
        'data_type': 'cluster',
        'n_cluster': 3,
        'n_cluster_mix': 1,
        'lower': 0.2,
        'upper': 0.8,
        'std': 0.07,
    }
    
    G, depot, total_cost, total_demand = generate_graph_degree(
        vertex_size=50, 
        edge_size=100, 
        distribution=distribution, 
        max_dhcost=10, 
        max_demand=5, 
        min_degree=2, 
        max_degree=4
    )
    
    # 可视化生成的图
    visualize_graph(G)