import networkx as nx

# Import your physics functions from your pipeline
from mtmc_pipeline import get_time_bounds, get_absolute_time

def generate_global_ids(all_camera_matches, all_tracklets, start_times):
    G = nx.Graph()
    
    # 1. Build the graph
    for (cam1, cam2), match_list in all_camera_matches.items():
        for id1, id2 in match_list:
            node1 = f"{cam1}_{id1}"
            node2 = f"{cam2}_{id2}"
            G.add_edge(node1, node2)
            
    global_id_mapping = {}
    valid_global_id = 1
    
    nodes_sorted = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    assigned_nodes = set()
    
    def check_conflict(node_a, node_b):
        cam_a, id_a = node_a.split('_')
        cam_b, id_b = node_b.split('_')
        
        start_a = all_tracklets[cam_a][int(id_a)]['start_frame']
        end_a   = all_tracklets[cam_a][int(id_a)]['end_frame']
        start_b = all_tracklets[cam_b][int(id_b)]['start_frame']
        end_b   = all_tracklets[cam_b][int(id_b)]['end_frame']
        
        if cam_a == cam_b:
            # SAME CAMERA: Cars physically cannot overlap in time.
            if max(start_a, start_b) <= min(end_a, end_b):
                return True
            return False
            
        else:
            # DIFFERENT CAMERAS: Enforce the physical transition matrix!
            t_mid_a = (start_a + end_a) / 2
            t_mid_b = (start_b + end_b) / 2
            
            abs_time_a = get_absolute_time(cam_a, t_mid_a, start_times)
            abs_time_b = get_absolute_time(cam_b, t_mid_b, start_times)
            
            time_gap = abs(abs_time_a - abs_time_b)
            min_gap, max_gap = get_time_bounds(cam_a, cam_b)
            
            # If the time gap is physically impossible, they conflict!
            if time_gap < min_gap or time_gap > max_gap:
                return True
                
            return False

    # 2. Extract clusters greedily
    for start_node in nodes_sorted:
        if start_node in assigned_nodes:
            continue
            
        current_cluster = [start_node]
        assigned_nodes.add(start_node)
        
        lengths = nx.single_source_shortest_path_length(G, start_node)
        candidates = sorted(lengths.keys(), key=lambda n: (lengths[n], -G.degree(n)))
        
        for candidate in candidates:
            if candidate in assigned_nodes:
                continue
                
            has_conflict = False
            for cluster_node in current_cluster:
                if check_conflict(candidate, cluster_node):
                    has_conflict = True
                    break
                    
            if not has_conflict:
                current_cluster.append(candidate)
                assigned_nodes.add(candidate)
        
        # 3. Assign the Global ID mapping
        for node in current_cluster:
            cam_id, local_id = node.split('_')
            if cam_id not in global_id_mapping:
                global_id_mapping[cam_id] = {}
            global_id_mapping[cam_id][int(local_id)] = valid_global_id
            
        valid_global_id += 1
            
    return global_id_mapping