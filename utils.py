import networkx as nx

from itertools import combinations


def build_graph(
    data,
    threshold=10,
    use_edge_attrs=True,
    use_beer_attrs=True,
    add_beer_similarity_edges=True,
    debug=False
):
    df_ratings = data.rename(columns={'review_overall': 'weight'})

    df_ratings = df_ratings.groupby(['beer_beerid']).filter(lambda x: len(x) >= threshold)

    graph = nx.from_pandas_edgelist(
        df_ratings,
        source='review_profilename',
        target='beer_beerid',
        edge_attr='weight',
        create_using=nx.DiGraph
    )

    if use_edge_attrs:
        index_data = data.set_index(['review_profilename', 'beer_beerid'])

        before = len(index_data)
        index_data = index_data[~index_data.index.duplicated(keep='last')]
        after = len(index_data)
        if debug:
            print(f'Found {before - after} duplicates.')

        # Drop fields related to the beer, not to the review
        index_data = index_data.drop(['brewery_name'], axis=1)
        index_data = index_data.drop(['beer_name'], axis=1)
        index_data = index_data.drop(['brewery_id'], axis=1)
        index_data = index_data.drop(['beer_style'], axis=1)
        index_data = index_data.drop(['beer_abv'], axis=1)
        # We already use this as weight
        index_data = index_data.drop(['review_overall'], axis=1)
        # We don't need the timestamp either
        index_data = index_data.drop(['review_time'], axis=1)

        edge_attrs = index_data.to_dict(orient='index')
        nx.set_edge_attributes(graph, edge_attrs)

        if debug:
            print('Attrs for edge "fodeeoz", 436:')
            print(graph.edges()['fodeeoz', 436])

    if use_beer_attrs:
        beer_attrs = data[['beer_beerid', 'beer_style', 'brewery_id', 'beer_abv']]
        beer_attrs = beer_attrs.set_index(['beer_beerid'])
        beer_attrs = beer_attrs[~beer_attrs.index.duplicated(keep='last')]

        beer_attrs = beer_attrs.to_dict(orient='index')
        nx.set_node_attributes(graph, beer_attrs)

        if debug:
            print('Example attrs for node 436:')
            print(graph.nodes()[436])

    if add_beer_similarity_edges:
        styles_dict = {}
        for node, style in graph.nodes(data='beer_style'):
            if not isinstance(node, int):
                # It's a user
                continue
            styles_dict.setdefault(style, []).append(node)

        brewery_dict = {}
        for node, brewery in graph.nodes(data='brewery_id'):
            if not isinstance(node, int):
                # It's a user
                continue
            brewery_dict.setdefault(brewery, []).append(node)

        edges = []
        for styles, nodes in styles_dict.items():
            edges.extend(list(combinations(nodes, 2)))
            styles_edges_len = len(edges)
        if debug:
            print(
                f'Adding {styles_edges_len} edges from beer similarity based on styles.'
            )

        for brewery, nodes in brewery_dict.items():
            edges.extend(list(combinations(nodes, 2)))
            brewery_edges_len = len(edges) - styles_edges_len
        if debug:
            print(
                f'Adding {brewery_edges_len} edges from beer similarity based on breweries.'
            )

        graph.add_edges_from(edges)

    return graph


def get_beers_mask(graph):
    return [isinstance(node, int) for node in list(graph.nodes())]


def get_user_idxs(graph):
    return [isinstance(node, str) for node in list(graph.nodes())]


def get_beer_list(graph):
    return [node for node in list(graph.nodes()) if isinstance(node, int)]
