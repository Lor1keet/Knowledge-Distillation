import networkx as nx
import matplotlib.pyplot as plt
from CARProblemDef import generate_graph_degree,adjust_edge_count

def visualize_graph(G, depot=None):
    """
    可视化图 G，包括节点和边。可以选择高亮显示仓库节点 (depot)。
    
    参数:
    ----------
    G : networkx.Graph
        需要可视化的图
    depot : 节点, optional
        仓库节点。如果提供，则会高亮显示该节点
    """
    
    # 获取节点的坐标信息，如果存在 'pos' 属性
    # 将torch.Tensor转换为普通的列表或元组
    pos = nx.spring_layout(G)  # 节点布局
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"dem: {d['demand']}, cost: {d['dhcost']}" for u, v, d in
    #                                                   G.edges(data=True)})

    # 显示图
    plt.show()
if __name__ == "__main__":
    # 例如，加载生成的图
    # 你可以从其他模块或生成函数中导入生成好的图 G 和 depot
    from CARProblemDef import generate_graph_degree

    
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
    visualize_graph(G, depot)