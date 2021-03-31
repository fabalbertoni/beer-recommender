import networkx as nx
import numpy as np

from itertools import permutations

from sklearn.preprocessing import LabelEncoder


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
            edges.extend(list(permutations(nodes, 2)))
            styles_edges_len = len(edges)
        if debug:
            print(
                f'Adding {styles_edges_len} edges from beer similarity based on styles.'
            )

        for brewery, nodes in brewery_dict.items():
            edges.extend(list(permutations(nodes, 2)))
            brewery_edges_len = len(edges) - styles_edges_len
        if debug:
            print(
                f'Adding {brewery_edges_len} edges from beer similarity based on breweries.'
            )

        graph.add_edges_from(edges)

    return graph


def decompose_graph(graph, split_ratio=.8):
    # TODO Instead of thecking str or int, build the initial mask for the graph and use it.
    # Transform user ids in the graph, then avoid calling transform a lot of times.

    print('1')
    le_users = LabelEncoder()
    le_items = LabelEncoder()

    le_users.fit([user for user in graph.nodes() if isinstance(user, str)])
    le_items.fit([item for item in graph.nodes() if isinstance(item, int)])

    print('2')

    history_u_lists = {
        user:
        [
            item for item in graph.successors(user) if isinstance(item, int)
        ]
        for user in graph.nodes() if isinstance(user, str)
    }

    history_ur_lists = {
        user:
        [
            list(graph.get_edge_data(user, item).values()) for item in graph.successors(user)
            if isinstance(item, int)
        ]
        for user in graph.nodes() if isinstance(user, str)
    }

    # assert len(history_v_lists['fodeeoz']) == len(history_vr_lists['fodeeoz'])

    history_v_lists = {
        item:
        [
            user for user in graph.predecessors(item) if isinstance(user, str)
        ]
        for item in graph.nodes() if isinstance(item, int)
    }

    history_vr_lists = {
        item:
        [
            list(graph.get_edge_data(user, item).values()) for user in graph.predecessors(item)
            if isinstance(user, str)
        ]
        for item in graph.nodes() if isinstance(item, int)
    }

    # assert len(history_u_lists[681]) == len(history_ur_lists[681])

    split_data = np.array(
        [
            [user, item, list(graph.get_edge_data(user, item).values())]
            for user, item in graph.edges()
            if isinstance(user, str) and isinstance(item, int)
        ]
    )

    u, v, r = np.split(split_data, 3, axis=1)
    assert len(u) == len(v) == len(r)

    print('6')

    u, v, r = list(v), list(u), list(r)

    u = le_users.transform(u)
    v = le_items.transform(v)

    history_u_lists = {
        le_users.transform([user])[0]: le_items.transform(items)
        for user, items in history_u_lists.items()
    }

    history_ur_lists = {
        le_users.transform([user])[0]: ratings
        for user, ratings in history_ur_lists.items()
    }

    history_v_lists = {
        le_items.transform([item])[0]: le_users.transform(users)
        for item, users in history_v_lists.items()
    }

    history_vr_lists = {
        le_items.transform([item])[0]: ratings
        for item, ratings in history_vr_lists.items()
    }

    print('Done transforming. ')

    N = len(u)

    train_u = u[:int(N * split_ratio)]
    train_v = v[:int(N * split_ratio)]
    train_r = r[:int(N * split_ratio)]

    test_u = u[int(N * split_ratio):]
    test_v = v[int(N * split_ratio):]
    test_r = r[int(N * split_ratio):]

    print('7')

    item_adj_lists = {le_items.transform([item])[0]: le_items.transform(list(set(graph.neighbors(item)))) for item in graph.nodes() if isinstance(item, int)}

    assert len(history_u_lists) == len(history_ur_lists)
    assert len(history_v_lists) == len(history_vr_lists)

    return (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v,
            train_r, test_u, test_v, test_r, item_adj_lists)


def get_beers_mask(graph):
    return [isinstance(node, int) for node in list(graph.nodes())]


def get_user_mask(graph):
    return [isinstance(node, str) for node in list(graph.nodes())]


def get_beer_list(graph):
    return [node for node in list(graph.nodes()) if isinstance(node, int)]
