def construct_tree_dict(estimator):
    tree_dict = {}
    single_tree = estimator.tree_
    for ind, (feat, thresh, imp, left, right) in enumerate(zip(single_tree.feature, single_tree.threshold, single_tree.impurity, single_tree.children_left, single_tree.children_right)):
        if feat != -2:
            tree_dict[ind] = {'feature': data_df.columns[feat],
                              'threshold': thresh,
                              'impurity': imp,
                              'left_node': left,
                              'right_node':right}
        else:
            tree_dict[ind] = {'feature': 'terminal_leaf',
                              'threshold': np.nan,
                              'impurity': imp,
                              'left_node': np.nan,
                              'right_node':np.nan}
    return tree_dict
    
def find_path(tree_dict, point_series):
    path = []
    node_id = 0
    
    node_data = tree_dict[node_id]
    node_data.update({'node_id': node_id, 'value': point_series[node_data['feature']]}) # Need to add node_id for visualization purposes
    
    feat = node_data['feature']

    while feat != 'terminal_leaf':
        if node_data['value'] <= node_data['threshold']:
            node_data['comparison'] = '<='
            path.append(copy.deepcopy(node_data))
            node_id = node_data['left_node']
        elif node_data['value'] > node_data['threshold']:
            node_data['comparison'] = '>'
            path.append(copy.deepcopy(node_data))
            node_id = node_data['right_node']
        else:
            print('something went wrong')
            break
        node_data = tree_dict[node_id]
        node_data['node_id'] = node_id
        try:
            node_data.update({'node_id': node_id, 'value': point_series[node_data['feature']]})
        except KeyError:
            node_data.update({'node_id': node_id, 'value': np.nan})
        feat = node_data['feature']
    path.append(copy.deepcopy(node_data))
    return path
    
def find_divergence_features(combos):
    divergence_feats = []
    for ind, c in enumerate(combos):
        c = list(zip(*c)) # switch the orientation of the combinations to compare each step through the tree
        for ind, (f, t) in enumerate(c):
            if f['feature'] != t['feature']: # find where the steps are different
                divergence_feats.append(c[ind-1][0]['feature']) # add the previous step as the divergence point
    return divergence_feats
    
def calc_divergence_difference(combos):
    divergence_feat_dif = {}
    for ind, c in enumerate(combos):
        c = list(zip(*c))
        for ind, (f, t) in enumerate(c):
            if f['feature'] != t['feature']:
                # Calculate the difference in value at the divergence point
                try:
                    divergence_feat_dif[c[ind-1][0]['feature']].append(c[ind-1][0]['value'] - c[ind-1][1]['value'])
                except:
                    divergence_feat_dif[c[ind-1][0]['feature']] = [c[ind-1][0]['value'] - c[ind-1][1]['value']]
    return divergence_feat_dif
    
def calc_divergence_value_difference(combos):
    divergence_feat_dif = {}
    for _, c in enumerate(combos):
        c = list(zip(*c))
        for ind, (f, t) in enumerate(c):
            if f['feature'] != t['feature']:
                # Calculate the difference in value at the divergence point
                try:
                    divergence_feat_dif[c[ind-1][0]['feature']+'_diff'].append(c[ind-1][0]['value'] - c[ind-1][1]['value'])
                    divergence_feat_dif[c[ind-1][0]['feature']+'_val'].append(c[ind-1][0]['value'])
                except:
                    divergence_feat_dif[c[ind-1][0]['feature']+'_diff'] = [c[ind-1][0]['value'] - c[ind-1][1]['value']]
                    divergence_feat_dif[c[ind-1][0]['feature']+'_val'] = [c[ind-1][0]['value']]
    return divergence_feat_dif

def find_all_tree_divergence_points(model, test_df):
    
    fns = test_df[test_df['FN']==1].reset_index(drop=True)
    tps = test_df[test_df['TP']==1].reset_index(drop=True)
    tns = test_df[test_df['TN']==1].reset_index(drop=True)
    
    divergence_feats_fn = {}
    divergence_feats_tp = {}
    divergence_feats_tn = {}
    
    all_divergence_feat_fn_diff = {}
    all_divergence_feat_tp_diff = {}
    all_divergence_feat_tn_diff = {}
    
    for estimator in model.estimators_:
        tree_dict = construct_tree_dict(estimator)
        
        fn_paths = [find_path(tree_dict, row) for ind, row in fns.iterrows()]
        tp_paths = [find_path(tree_dict, row) for ind, row in tps.iterrows()]
        tn_paths = [find_path(tree_dict, row) for ind, row in tns.iterrows()]
        
        # Make pairwise combinations of each FN and TP
        combos_fn = [[f, t] for f in fn_paths for t in tp_paths]
        
        # Do the same for each TP and TN for comparison
        combos_tp = [[t_1, t_2] for t_1 in tp_paths for t_2 in tp_paths]
        combos_tn = [[n, p] for n in tn_paths for p in tp_paths]
        
        tree_diverge_fn = find_divergence_features(combos_fn)
        tree_diverge_tp = find_divergence_features(combos_tp)
        tree_diverge_tn = find_divergence_features(combos_tn)
        
        for k, v in Counter(tree_diverge_fn).items():
            try:
                divergence_feats_fn[k] += v
            except KeyError:
                divergence_feats_fn[k] = v
        for k, v in Counter(tree_diverge_tp).items():
            try:
                divergence_feats_tp[k] += v
            except KeyError:
                divergence_feats_tp[k] = v
        for k, v in Counter(tree_diverge_tn).items():
            try:
                divergence_feats_tn[k] += v
            except KeyError:
                divergence_feats_tn[k] = v
        
        divergence_feat_fn_diff = calc_divergence_value_difference(combos_fn)
        for k, v in divergence_feat_fn_diff.items():
            try:
                all_divergence_feat_fn_diff[k].extend(v)
            except KeyError:
                all_divergence_feat_fn_diff[k] = v
        
        divergence_feat_tp_diff = calc_divergence_value_difference(combos_tp)
        for k, v in divergence_feat_tp_diff.items():
            try:
                all_divergence_feat_tp_diff[k].extend(v)
            except KeyError:
                all_divergence_feat_tp_diff[k] = v
            
        divergence_feat_tn_diff = calc_divergence_value_difference(combos_tn)
        for k, v in divergence_feat_tn_diff.items():
            try:
                all_divergence_feat_tn_diff[k].extend(v)
            except KeyError:
                all_divergence_feat_tn_diff[k] = v

    all_divergence_feat_fn_diff = {k: np.mean(v) for k, v in all_divergence_feat_fn_diff.items()}
    all_divergence_feat_tp_diff = {k: np.mean(v) for k, v in all_divergence_feat_tp_diff.items()}
    all_divergence_feat_tn_diff = {k: np.mean(v) for k, v in all_divergence_feat_tn_diff.items()}
    
    tp_div_df = pd.DataFrame(all_divergence_feat_tp_diff, index = ["tp"]).T
#     print(tp_div_df.index)
    fn_div_df = pd.DataFrame(all_divergence_feat_fn_diff, index = ["fn"]).T
#     print(fn_div_df.index)
    compare_tp_div_df = pd.merge(fn_div_df, tp_div_df, how = 'left', left_index = True, right_index = True)
    tn_div_df = pd.DataFrame(all_divergence_feat_tn_diff, index = ["tn"]).T
    compare_tp_tn_div_df = pd.merge(compare_tp_div_df, tn_div_df, how = 'left', left_index = True, right_index = True)
    return (dict(sorted(divergence_feats_fn.items(), key=lambda x: x[1], reverse=True)),
            dict(sorted(divergence_feats_tp.items(), key=lambda x: x[1], reverse=True)),
            dict(sorted(divergence_feats_tn.items(), key=lambda x: x[1], reverse=True)),
            divergence_feat_fn_diff, divergence_feat_tp_diff, divergence_feat_tn_diff, compare_tp_tn_div_df)

def find_left_path(node, tree):
    path = [copy.deepcopy(node)]
    feat = tree[node]['feature']
    while feat != 'terminal_leaf':
        node = tree[node]['left_node']
        path.append(copy.deepcopy(node))
        feat = tree[node]['feature']
    return path
    
def backtrack(tree, path):
    position = -1
    try:
        while tree[path[position-1]]['right_node'] == path[position]:
            position -= 1
        return path[position-1]
    except IndexError:
        return None

def find_all_paths_in_tree(tree):
    paths = []
    path = find_left_path(0, tree)
    paths.append(path)
    backtrack_node = backtrack(tree, path)
    backtrack_path = find_left_path(tree[backtrack_node]['right_node'], tree)
    new_path = copy.deepcopy(paths[-1][:paths[-1].index(backtrack_node)+1]+backtrack_path)
    while new_path != paths[-1]:
        paths.append(new_path)
        backtrack_node = backtrack(tree, new_path)
        if backtrack_node is not None:
            backtrack_path = find_left_path(tree[backtrack_node]['right_node'], tree)
            new_path = copy.deepcopy(paths[-1][:paths[-1].index(backtrack_node)+1]+backtrack_path)
    return paths
    
def find_tree_width_depth(tree_paths):
    depth = max([len(l) for l in tree_paths])
    width = max(pd.DataFrame(tree_paths).nunique())
    return depth, width