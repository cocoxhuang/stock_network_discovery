import numpy as np
import networkx as nx
import torch
import plotly.graph_objects as go
from sklearn import covariance
from sklearn.covariance import EmpiricalCovariance
import pandas as pd
import requests
import yfinance as yf
from sklearn.decomposition  import PCA
import networkx as nx

def get_data(top,start,end,symbols = None):

    # download the symbols of stocks
    df = pd.read_html(requests.get('https://www.slickcharts.com/sp500',
                            headers={'User-agent': 'Mozilla/5.0'}).text)[0]
    if symbols == None: # if user does not specify stocks 
        symbols = list(df['Symbol'].values[:top])
        names = list(df['Company'].values[:top])
        print(f"The top {top} symbols in S&P 500 by weights order is: {symbols}")
    else:   # if user specify stocks
        names = list(df[df['Symbol'].isin(symbols)]['Company'].values)

    # downloading data
    data = yf.download(symbols, start=start, end=end)
    open = data['Open']
    close = data['Close']
    variation = close - open

    # remove NA stocks
    symbols_remove = variation.isna().sum()[variation.isna().sum() != 0].keys()
    names_remove = df[df['Symbol'].isin(symbols_remove)]['Company'].values
    variation = variation.drop(columns=symbols_remove)
    symbols = [x for x in symbols if x not in symbols_remove]
    names = [x for x in names if x not in names_remove]

    # standardize the time series: using correlations rather than covariance
    # former is more efficient for structure recovery
    X = variation.copy()
    X /= X.std(axis=0)
    
    # get embedding for graphing using PCA
    node_position_model = PCA(n_components=2)
    embedding = node_position_model.fit_transform(X.T).T

    return X, embedding, names

def cov_adj(X,threshold=0.5):
    cov = EmpiricalCovariance().fit(X)
    emp_cov = cov.covariance_

    det_emp_cov = np.linalg.det(emp_cov)
    print(f"determinant of the MLE Empirical Covariance Matrix is : {det_emp_cov:2f}.")
    if det_emp_cov < 0.0001:
        print('The MLE Empirical Covariance Matrix is singular.')
    else:
        print('The MLE Empirical Covariance Matrix is NOT singular.')

    d = 1 / np.sqrt(np.diag(emp_cov))
    emp_corr = emp_cov*d*d[:, np.newaxis]
    binary_adj_cov = np.abs(np.triu(emp_corr, k=1)) > threshold

    return binary_adj_cov


def glasso_adj(X):
    alphas = np.logspace(-1.5, 1, num=10)
    GLasso_model = covariance.GraphicalLassoCV(alphas=alphas)
    GLasso_model.fit(X)

    # Plot the graph of partial correlations
    precision = GLasso_model.precision_.copy()
    # Let the partial correlations matrix be partial_correlations
    # Denote the (i,j)-th entry of partial_correlations be rho_ij
    # Denote the (i,j)-th entry of precision matrix be p_ij
    # rho_ij = - (p_ij) / sqrt(p_ii * p_ij)
    d = 1 / np.sqrt(np.diag(precision))
    partial_correlations = precision*d*d[:, np.newaxis]
    binary_adj = np.abs(np.triu(partial_correlations, k=1)) > 0.02
    binary_adj.astype(int)
    return binary_adj


def create_G(binary_adj):
    np.fill_diagonal(binary_adj,False)

    edge_index = torch.tensor(binary_adj).to_sparse().indices()
    edges = list(zip(edge_index[0].tolist(),edge_index[1].tolist()))

    G = nx.Graph()
    pos = nx.get_node_attributes(G, "pos")
    G.add_nodes_from(np.arange(len(binary_adj)))
    G.add_edges_from(edges)
    
    return G

def plot_network(G,embedding,names,title):
    edge_x = []
    edge_y = []
    pos = {i:embedding.T[i] for i in G.nodes}
    nx.set_node_attributes(G, pos, "pos")

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = list(names)
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        # node_text.append('# of connections: '+str(len(adjacencies[1])))
        node_text[node] = node_text[node] + ', # of connections: '+str(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>' + title,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        # text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        text='',
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    return fig