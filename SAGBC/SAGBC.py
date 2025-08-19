
def getAnchor(fea, k):
    n = fea.shape[0]
    data1 = fea[np.random.choice(n, k, replace=False)]
    data_cell = [data1]
    while True:
        old_len = len(data_cell)
        data_cell = division(data_cell)
        if len(data_cell) == old_len:
            break

    last_center = []
    for cluster in data_cell:
        data = np.vstack(cluster)
        if data.shape[0] == 1:
            last_center.append(data[0])
        else:
            last_center.append(np.mean(data, axis=0))
    return np.vstack(last_center)


def division(hb_cell):
    gb_newcell = []
    for cluster in hb_cell:
        data = np.vstack(cluster)
        if data.shape[0] > 8:
            ball_1, ball_2 = spilt_ball(data)
            gb_newcell.append(ball_1)
            gb_newcell.append(ball_2)
        else:
            gb_newcell.append([data])
    return gb_newcell


def spilt_ball(data):
    D = squareform(pdist(data))
    r, c = np.unravel_index(np.argmax(D), D.shape)
    r1, c1 = r, c

    ball_1 = []
    ball_2 = []
    for row in data:
        if np.linalg.norm(row - data[r1]) < np.linalg.norm(row - data[c1]):
            ball_1.append(row)
        else:
            ball_2.append(row)
    return [np.array(ball_1)], [np.array(ball_2)]
path = 'C:/Users/'
mat = io.loadmat(path)
keys = [k for k in mat if not k.startswith('__')]
data = mat[keys[0]].astype(float)

start_time = time.perf_counter()
anchors = getAnchor(data, k=5000)
K = 5
N = data.shape[0]
g = len(anchors)

nn = NearestNeighbors(n_neighbors=K)
nn.fit(anchors)
knnDist, knnIdx = nn.kneighbors(data)

knnMeanDiff = np.mean(knnDist)
Gsdx = np.exp(-(knnDist ** 2) / (2 * knnMeanDiff ** 2))
Gsdx[Gsdx == 0] = np.finfo(float).eps


data_ids = np.repeat(np.arange(N), K)
anchor_ids = knnIdx.flatten()
anchor_to_data_map = defaultdict(list)
for anchor, point_id in zip(anchor_ids, data_ids):
    anchor_to_data_map[anchor].append(point_id)

particles = []
for idx, anchor in enumerate(anchors):
    covered_points = anchor_to_data_map.get(idx, [])
    if covered_points:
        distances = np.linalg.norm(data[covered_points] - anchor, axis=1)
        anchor_weight_for_this = []
        for pid in covered_points:
            anchor_list = knnIdx[pid]
            w_list = Gsdx[pid]
            try:
                local_idx = np.where(anchor_list == idx)[0][0]
                anchor_weight_for_this.append(w_list[local_idx])
            except IndexError:
                anchor_weight_for_this.append(0.0)
        weights = np.array(anchor_weight_for_this)
        radius = np.average(distances, weights=weights)
    else:
        radius = 0.0

    particles.append({
        'id': idx,
        'coord': anchor,
        'radius': radius,
        'type': 'anchor'
    })

particles_with_id = [(particle['id'], particle) for particle in particles]

partitioned_particles_rdd = sc.parallelize(particles_with_id) \
                              .partitionBy(1, lambda key: hash(key))

all_particles_broadcast = sc.broadcast(particles)


def build_edges(kv_pair):
    cid, particle = kv_pair
    all_particles = all_particles_broadcast.value
    coords = [p['coord'] for p in all_particles]
    ids = [p['id'] for p in all_particles]
    tree = KDTree(coords)
    neighbor_indices = tree.query_ball_point(particle['coord'], r=particle['radius'] * 2)
    local_edges = set()
    for idx in neighbor_indices:
        nid = ids[idx]
        if nid == cid or cid > nid:
            continue
        points_c = set(anchor_to_data_map.get(cid, []))
        points_n = set(anchor_to_data_map.get(nid, []))
        shared_points = points_c & points_n
        if not shared_points:
            continue
        size_c = len(points_c)
        size_n = len(points_n)
        size_shared = len(shared_points)
        if size_c == 0 or size_n == 0:
            continue
        sim_weight = size_shared / min(size_c, size_n)
        edge = Edge(cid, nid, weight=1 - sim_weight)
        local_edges.add(edge)
    return list(local_edges)

edges_rdd = partitioned_particles_rdd.flatMap(build_edges).cache()
edges = edges_rdd.collect()

class UnionFind:
    def __init__(self, particle_ids):
        self.id_map = {pid: idx for idx, pid in enumerate(particle_ids)}
        self.parent = list(range(len(particle_ids)))
        self.rank = [1] * len(particle_ids)
        self.size = [1] * len(particle_ids)

    def find(self, u):
        u_idx = self.id_map.get(u, -1)
        if u_idx == -1:
            print(f"Warning: Particle ID {u} not found in id_map.")
            return -1
        path = []
        while self.parent[u_idx] != u_idx:
            path.append(u_idx)
            u_idx = self.parent[u_idx]
        for node in path:
            self.parent[node] = u_idx
        return u_idx

    def union(self, u, v):
        u_idx = self.id_map.get(u, -1)
        v_idx = self.id_map.get(v, -1)
        if u_idx == -1 or v_idx == -1:
            print(f"Warning: Particle IDs {u}, {v} not found.")
            return

        root_u = self.find(u)
        root_v = self.find(v)
        if root_u == root_v:
            return

        if self.rank[root_u] > self.rank[root_v]:
            self.parent[root_v] = root_u
            self.size[root_u] += self.size[root_v]
        elif self.rank[root_u] < self.rank[root_v]:
            self.parent[root_u] = root_v
            self.size[root_v] += self.size[root_u]
        else:
            self.parent[root_v] = root_u
            self.rank[root_u] += 1
            self.size[root_u] += self.size[root_v]

    def get_size(self, u):
        root_u = self.find(u)
        return self.size[root_u]

class Cluster:
    def __init__(self, cid):
        self.cluster_id = cid
        self.cell_ids = set()
    def add_cell(self, pid):
        self.cell_ids.add(pid)
    def get_all_point_indices(self, particle_index_map):
        indices = set()
        print("particle_index_map keys:", sorted(particle_index_map.keys()))
        for cid in self.cell_ids:
            indices.update(particle_index_map[cid])
        return indices

class MinimumSpanningForest:
    def build_clusters(self, edges):
        involved_ids = set()
        for edge in edges:
            involved_ids.add(edge.u)
            involved_ids.add(edge.v)
        uf = UnionFind(list(involved_ids))
        edges = sorted(edges, key=lambda x: x.weight)
        for edge in edges:
            u, v = edge.u, edge.v
            uf.union(u, v)
        cluster_map = defaultdict(set)
        for cid in involved_ids:
            cluster_map[uf.find(cid)].add(cid)
        clusters = []
        for idx, group in enumerate(cluster_map.values()):
            cluster = Cluster(idx)
            for pid in group:
                cluster.add_cell(pid)
            clusters.append(cluster)
        return clusters


def summarize_clusters(clusters, anchor_to_data_map, n, knnIdx, Gsdx):
    point_cluster = -np.ones(n, dtype=int)
    point_weights = np.zeros(n)
    for cluster in clusters:
        for anchor_id in cluster.cell_ids:
            data_point_ids = anchor_to_data_map.get(anchor_id, [])
            for pid in data_point_ids:
                anchor_list = knnIdx[pid]
                w_list = Gsdx[pid]
                if anchor_id in anchor_list:
                    local_idx = np.where(anchor_list == anchor_id)[0][0]
                    weight = w_list[local_idx]
                    if weight > point_weights[pid]:
                        point_weights[pid] = weight
                        point_cluster[pid] = cluster.cluster_id
    return point_cluster

mst = MinimumSpanningForest()
clusters = mst.build_clusters(edges)
anchor_to_data_map = dict(anchor_to_data_map)
point_cluster = summarize_clusters(clusters, anchor_to_data_map, N, knnIdx, Gsdx)













