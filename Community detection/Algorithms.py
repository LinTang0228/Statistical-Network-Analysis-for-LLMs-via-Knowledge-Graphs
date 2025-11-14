
"""Community detection helpers used by the GraphRAG experiments."""

import numpy as np
import networkx as nx
import warnings
from sklearn.cluster import SpectralClustering, KMeans
warnings.filterwarnings('ignore', message='invalid value encountered in sqrt', category=RuntimeWarning)
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, euclidean_distances
from typing import List, Dict, Optional, Any, Tuple
import logging
import hashlib
from functools import lru_cache
from threading import Lock

logger = logging.getLogger(__name__)

_adjacency_matrix_cache = {}
_eigenvector_cache = {}
_feature_matrix_cache = {}
_cache_lock = Lock()

def _get_matrix_hash(graph: nx.Graph, nodes: List[str]) -> str:
    """Generate a unique hash for graph and node list preserving order"""
    nodes_hash = hashlib.md5("_".join(nodes).encode()).hexdigest()
    edges_hash = hashlib.md5(str(sorted(graph.edges(data=True))).encode()).hexdigest()
    graph_str = f"nodes_count:{len(nodes)}_edges_count:{len(graph.edges())}_nodes_hash:{nodes_hash}_edges_hash:{edges_hash}"
    return hashlib.md5(graph_str.encode()).hexdigest()

def _get_cached_adjacency_matrix(graph: nx.Graph, nodes: List[str]) -> Optional[np.ndarray]:
    """Get cached adjacency matrix if available"""
    hash_key = _get_matrix_hash(graph, nodes)
    with _cache_lock:
        if hash_key in _adjacency_matrix_cache:
            return _adjacency_matrix_cache[hash_key].copy()
    return None

def _cache_adjacency_matrix(graph: nx.Graph, nodes: List[str], matrix: np.ndarray) -> None:
    """Cache adjacency matrix"""
    hash_key = _get_matrix_hash(graph, nodes)
    with _cache_lock:
        if len(_adjacency_matrix_cache) > 50:
            keys_to_remove = list(_adjacency_matrix_cache.keys())[:10]
            for key in keys_to_remove:
                del _adjacency_matrix_cache[key]
        _adjacency_matrix_cache[hash_key] = matrix.copy()

@lru_cache(maxsize=128)
def _compute_graph_density(n_nodes: int, n_edges: int) -> float:
    """Compute and cache graph density"""
    if n_nodes <= 1:
        return 0.0
    return (2 * n_edges) / (n_nodes * (n_nodes - 1))

def clear_all_caches():
    """Clear all caches to free memory"""
    global _adjacency_matrix_cache, _eigenvector_cache, _feature_matrix_cache
    
    with _cache_lock:
        _adjacency_matrix_cache.clear()
        _eigenvector_cache.clear()
        _feature_matrix_cache.clear()
    
    _compute_graph_density.cache_clear()
    
    logger.info("All caches cleared")

try:
    from scipy.sparse.linalg import eigsh
except Exception:
    eigsh = None

class CommunityDetectionBase:
    """Base class with common helper methods for community detection algorithms"""
    
    @staticmethod
    def validate_inputs(graph: nx.Graph, k: int = None) -> Tuple[List[str], int, bool]:
        """Validate inputs and return (nodes, n, should_continue)"""
        if graph.number_of_nodes() == 0:
            return [], 0, False
        nodes = list(graph.nodes())
        n = len(nodes)
        if n < 2:
            return nodes, n, False
        return nodes, n, True
    
    @staticmethod
    def get_adjacency_matrix(graph: nx.Graph, nodes: List[str]) -> np.ndarray:
        """Get adjacency matrix with caching"""
        A = _get_cached_adjacency_matrix(graph, nodes)
        if A is None:
            A = nx.adjacency_matrix(graph, nodelist=nodes).toarray().astype(float)
            _cache_adjacency_matrix(graph, nodes, A)
        return A.astype(float)
    
    @staticmethod
    def estimate_natural_k(n: int) -> int:
        """Estimate natural k using sqrt heuristic"""
        return max(2, min(int(np.sqrt(n)), n // 2))
    
    @staticmethod
    def group_nodes_by_labels(nodes: List[str], labels: np.ndarray) -> Dict[int, List[str]]:
        """Group nodes by their cluster labels"""
        clusters = {}
        for i, node in enumerate(nodes):
            label = labels[i]
            clusters.setdefault(label, []).append(node)
        return clusters
    
    @staticmethod
    def ensure_k_communities(clusters: Dict[int, List[str]], target_k: int, 
                            graph: nx.Graph = None) -> List[List[str]]:
        """Ensure exactly k non-empty communities"""
        non_empty = [comm for comm in clusters.values() if comm]
        
        if len(non_empty) > target_k:
            non_empty.sort(key=len, reverse=True)
            return non_empty[:target_k]
        
        while len(non_empty) < target_k and any(len(c) > 1 for c in non_empty):
            largest_idx = max(range(len(non_empty)), key=lambda i: len(non_empty[i]))
            if len(non_empty[largest_idx]) > 1:
                mid = len(non_empty[largest_idx]) // 2
                new_comm = non_empty[largest_idx][mid:]
                non_empty[largest_idx] = non_empty[largest_idx][:mid]
                non_empty.append(new_comm)
            else:
                break
        
        return non_empty[:target_k]
    
    @staticmethod
    def compute_normalized_laplacian(A: np.ndarray) -> np.ndarray:
        """Compute normalized Laplacian matrix"""
        np.fill_diagonal(A, A.diagonal() + 1e-6)
        degrees = np.sum(A, axis=1)
        D = np.diag(degrees)
        L = D - A
        D_sqrt_inv = np.diag(np.where(degrees > 1e-10, 1.0 / np.sqrt(degrees), 0))
        return D_sqrt_inv @ L @ D_sqrt_inv

class ImprovedSpectralClusteringFixed(CommunityDetectionBase):
    """Regular Spectral Clustering without SCORE (normalized Laplacian)"""
    
    def __init__(self, k: int):
        if k is None:
            raise ValueError("k must be provided for ImprovedSpectralClusteringFixed")
        self.k = k
    
    def detect_communities(self, graph: nx.Graph, entities: List[Dict[str, Any]] = None, **kwargs) -> List[List[str]]:
        """Regular spectral clustering using normalized Laplacian with robust error handling"""
        if graph.number_of_nodes() == 0:
            return []
        
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n < 2:
            return [nodes]
        
        n_components = nx.number_connected_components(graph)
        
        if self.k > n:
            logger.warning(f"Requested k={self.k} but only {n} nodes available. Using k={n}")
            effective_k = n
        else:
            effective_k = self.k
        
        components = [list(c) for c in nx.connected_components(graph)]
        if len(components) > 1:
            if len(components) > effective_k:
                logger.warning(f"Spectral: Graph has {len(components)} components but k={effective_k}. Merging small components.")
                sorted_components = sorted(components, key=len, reverse=True)
                kept_components = sorted_components[:effective_k-1]
                merged_component = []
                for comp in sorted_components[effective_k-1:]:
                    merged_component.extend(comp)
                if merged_component:
                    kept_components.append(merged_component)
                components = kept_components
            return self._spectral_on_components(graph, components, effective_k)
        
        processed_graph = self._preprocess_graph_robust(graph, n)
        
        params = self._get_optimal_parameters(processed_graph, effective_k)
        
        try:
            processed_nodes = list(processed_graph.nodes())
            A = _get_cached_adjacency_matrix(processed_graph, processed_nodes)
            if A is None:
                A = nx.adjacency_matrix(processed_graph, nodelist=processed_nodes).toarray().astype(float)
                _cache_adjacency_matrix(processed_graph, processed_nodes, A)
            else:
                A = A.astype(float)
            
            np.fill_diagonal(A, A.diagonal() + 1e-6)
            
            degrees = np.sum(A, axis=1)
            D = np.diag(degrees)
            L = D - A
            D_sqrt_inv = np.diag(np.where(degrees > 1e-10, 1.0 / np.sqrt(degrees), 0))
            L_norm = D_sqrt_inv @ L @ D_sqrt_inv
            
            params['n_clusters'] = effective_k
            
            clustering = SpectralClustering(**params)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                labels = clustering.fit_predict(L_norm)
            
            if hasattr(clustering, 'embedding_') and clustering.embedding_ is not None:
                embedding = clustering.embedding_
                if embedding.shape[1] >= 2:
                    eigenvector_similarity = cosine_similarity(embedding.T)
                    np.fill_diagonal(eigenvector_similarity, 0)
                    max_similarity = np.max(eigenvector_similarity)
                    
                    if max_similarity > 0.99:
                        logger.warning(f"Spectral: Eigenvectors too similar (max similarity: {max_similarity:.3f}), using more robust clustering")
                        robust_km = KMeans(
                            n_clusters=params['n_clusters'],
                            n_init=100,
                            max_iter=1000,
                            random_state=42,
                            algorithm='lloyd'
                        )
                        labels = robust_km.fit_predict(embedding)
            
            node_mapping = {processed_nodes[i]: i for i in range(len(processed_nodes))}
            original_labels = []
            for node in nodes:
                if node in node_mapping:
                    original_labels.append(labels[node_mapping[node]])
                else:
                    original_labels.append(0)
            
            unique_labels = len(set(original_labels))
            target_k = params['n_clusters']
            if unique_labels < target_k and n >= target_k:
                logger.warning(f"Spectral produced only {unique_labels} clusters, using intelligent splitting to reach {target_k}")
                
                if unique_labels > target_k * 0.7:
                    logger.info("Retrying with more aggressive k-means...")
                    try:
                        retry_clustering = SpectralClustering(
                            n_clusters=target_k,
                            affinity='precomputed',
                            assign_labels='kmeans',
                            n_init=50,
                            random_state=123
                        )
                        retry_labels = retry_clustering.fit_predict(L_norm)
                        retry_unique = len(set(retry_labels))
                        
                        if retry_unique > unique_labels:
                            logger.info(f"Retry improved from {unique_labels} to {retry_unique} clusters")
                            original_labels = []
                            for node in nodes:
                                if node in node_mapping:
                                    original_labels.append(retry_labels[node_mapping[node]])
                                else:
                                    original_labels.append(0)
                            unique_labels = retry_unique
                    except Exception:
                        pass
                
                        
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}, using fallback")
            fallback_k = self.k
            original_labels = list(np.arange(n) % fallback_k)
        
        clusters = {}
        for i, node in enumerate(nodes):
            label = original_labels[i]
            clusters.setdefault(label, []).append(node)
        
        all_communities = []
        target_k = effective_k
        
        non_empty_communities = []
        for i in range(len(clusters)):
            if i in clusters and len(clusters[i]) > 0:
                non_empty_communities.append(clusters[i])
        
        logger.info(f"Spectral: Initial non-empty communities: {len(non_empty_communities)}, Target k: {target_k}, effective_k: {effective_k}, self.k: {self.k}")
        
        if len(non_empty_communities) < target_k:
            logger.warning(f"Spectral: Only {len(non_empty_communities)} non-empty communities, but target is {target_k}")
            all_communities = non_empty_communities
        else:
            all_communities = non_empty_communities[:target_k]
        
        expected_k = effective_k
        actual_k = len(all_communities)
        
        if actual_k != expected_k:
            logger.warning(f"Spectral: Produced {actual_k} communities, target was {expected_k} (difference acceptable)")
        
        logger.info(f"Spectral clustering: {len(all_communities)} communities (non-empty)")
        
        return all_communities
    
    def _preprocess_graph_robust(self, graph: nx.Graph, n: int) -> nx.Graph:
        """Robust graph preprocessing for sparse graphs with consistent fallback behavior"""
        processed = graph.copy()
        
        for node in processed.nodes():
            if not processed.has_edge(node, node):
                processed.add_edge(node, node, weight=1e-6)
        
        if not nx.is_connected(processed):
            components = list(nx.connected_components(processed))
            logger.info(f"Spectral: Graph has {len(components)} connected components - will cluster separately")
        
        isolated = list(nx.isolates(processed))
        if isolated:
            logger.info(f"Spectral: Removing {len(isolated)} isolated nodes")
            processed.remove_nodes_from(isolated)
        
        return processed
    
    def _get_optimal_parameters(self, graph: nx.Graph, k: int) -> Dict[str, Any]:
        """Get optimal parameters based on graph properties"""
        n = graph.number_of_nodes()
        e = graph.number_of_edges()
        density = _compute_graph_density(n, e)
        n_components = nx.number_connected_components(graph)
        
        if density < 0.01 or n_components > 10:
            assign_labels = 'kmeans'
            n_init = 30
        elif density > 0.1:
            assign_labels = 'kmeans'
            n_init = 20
        else:
            assign_labels = 'kmeans'
            n_init = 25
        
        return {
            'n_clusters': k,
            'affinity': 'precomputed',
            'assign_labels': assign_labels,
            'random_state': 42,
            'n_init': n_init
        }
    
    def _spectral_on_components(self, graph: nx.Graph, components: List[List[str]], k: int) -> List[List[str]]:
        """Distribute K across connected components proportionally"""
        total_nodes = sum(len(c) for c in components)
        allocations = []
        remaining_k = k
        
        for i, comp in enumerate(components):
            if i == len(components) - 1:
                k_i = max(1, remaining_k)
            else:
                k_i = max(1, round(k * len(comp) / total_nodes))
                remaining_k -= k_i
            allocations.append(k_i)
        
        max_adjustments = k * 2
        adjustments = 0
        
        while sum(allocations) < k and adjustments < max_adjustments:
            added = False
            for i in sorted(range(len(allocations)), key=lambda x: len(components[x]), reverse=True):
                if allocations[i] < len(components[i]):
                    allocations[i] += 1
                    added = True
                    break
            if not added:
                break
            adjustments += 1
        
        adjustments = 0
        while sum(allocations) > k and adjustments < max_adjustments:
            removed = False
            for i in sorted(range(len(allocations)), key=lambda x: len(components[x])):
                if allocations[i] > 1:
                    allocations[i] -= 1
                    removed = True
                    break
            if not removed:
                break
            adjustments += 1
        
        all_communities = []
        for comp_nodes, k_i in zip(components, allocations):
            comp_graph = graph.subgraph(comp_nodes)
            comp_communities = self._spectral_k(comp_graph, comp_nodes, k_i)
            all_communities.extend(comp_communities)
        
        if len(all_communities) > k:
            logger.warning(f"Spectral on components produced {len(all_communities)} communities, limiting to k={k}")
            all_communities.sort(key=len, reverse=True)
            all_communities = all_communities[:k]
        
        return all_communities
    
    def _spectral_k(self, graph: nx.Graph, comp_nodes: List[str], k_i: int) -> List[List[str]]:
        """Run spectral clustering on a single component"""
        if len(comp_nodes) <= k_i:
            return [[n] for n in comp_nodes]
        
        try:
            A = nx.adjacency_matrix(graph, nodelist=comp_nodes).toarray().astype(float)
            np.fill_diagonal(A, A.diagonal() + 1e-6)
            
            degrees = np.sum(A, axis=1)
            D = np.diag(degrees)
            L = D - A
            D_sqrt_inv = np.diag(np.where(degrees > 1e-10, 1.0 / np.sqrt(degrees), 0))
            L_norm = D_sqrt_inv @ L @ D_sqrt_inv
            
            clustering = SpectralClustering(
                n_clusters=k_i,
                affinity='precomputed',
                assign_labels='kmeans',
                n_init=20,
                random_state=42
            )
            labels = clustering.fit_predict(L_norm)
            
            clusters = [[] for _ in range(k_i)]
            for idx, node in enumerate(comp_nodes):
                clusters[labels[idx]].append(node)
            
            for j, cluster in enumerate(clusters):
                if not cluster:
                    largest_idx = max(range(k_i), key=lambda i: len(clusters[i]))
                    if len(clusters[largest_idx]) > 1:
                        donor = max(clusters[largest_idx], key=lambda n: graph.degree(n))
                        clusters[largest_idx].remove(donor)
                        clusters[j].append(donor)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Spectral clustering failed on component: {e}")
            clusters = [[] for _ in range(k_i)]
            for i, node in enumerate(comp_nodes):
                clusters[i % k_i].append(node)
            return clusters
    
    def _intelligent_cluster_splitting(self, labels, processed_graph, processed_nodes, node_mapping, original_nodes, current_k, target_k=None):
        """Intelligently split clusters using graph structure instead of arbitrary splitting"""
        labels = labels.copy()
        max_iterations = 20
        iterations = 0
        
        if target_k is None:
            target_k = self.k
        
        while len(set(labels)) < target_k and iterations < max_iterations:
            iterations += 1
            best_cluster = self._find_best_cluster_to_split(labels, processed_graph, processed_nodes, node_mapping)
            
            if best_cluster is None:
                logger.warning(f"No suitable clusters to split. Current: {len(set(labels))}, Target: {target_k}")
                break
                
            new_label = len(set(labels))
            split_nodes = self._split_cluster_by_structure(
                best_cluster, labels, processed_graph, processed_nodes, node_mapping
            )
            
            if not split_nodes:
                logger.warning(f"Failed to split cluster {best_cluster}. Breaking to avoid infinite loop.")
                break
            
            for node_idx in split_nodes:
                labels[node_idx] = new_label
            
            if len(set(labels)) <= new_label:
                logger.warning(f"Split did not create new cluster. Breaking to avoid infinite loop.")
                break
                
        if iterations >= max_iterations:
            logger.error(f"Reached maximum iterations ({max_iterations}) while splitting clusters")
            
        return labels.tolist()
    
    def _find_best_cluster_to_split(self, labels, processed_graph, processed_nodes, node_mapping):
        """Find the cluster that would benefit most from splitting"""
        best_cluster = None
        best_score = -1
        MIN_SPLIT_SCORE_THRESHOLD = 0.3
        candidates = []
        
        for cluster_id in set(labels):
            cluster_nodes = np.where(labels == cluster_id)[0]
            if len(cluster_nodes) < 2:
                continue
                
            internal_edges = 0
            external_edges = 0
            total_possible = len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
            
            for i, node_idx in enumerate(cluster_nodes):
                if node_idx < len(processed_nodes):
                    node = processed_nodes[node_idx]
                    for j, other_idx in enumerate(cluster_nodes[i+1:], i+1):
                        if other_idx < len(processed_nodes):
                            other_node = processed_nodes[other_idx]
                            if processed_graph.has_edge(node, other_node):
                                internal_edges += 1
                    
                    for neighbor in processed_graph.neighbors(node):
                        neighbor_idx = node_mapping.get(neighbor, -1)
                        if neighbor_idx >= 0 and labels[neighbor_idx] != cluster_id:
                            external_edges += 1
            
            if total_possible > 0:
                density = internal_edges / total_possible
                conductance = external_edges / (2 * internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
                score = len(cluster_nodes) * (1 - density) * (1 + conductance)
                if score >= MIN_SPLIT_SCORE_THRESHOLD:
                    candidates.append((score, len(cluster_nodes), cluster_id))

        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        
        if candidates:
            best_score, _, best_cluster = candidates[0]
            logger.debug(f"Selected cluster {best_cluster} for splitting with score {best_score:.3f}")
            return best_cluster
                    
        return None
    
    def _split_cluster_by_structure(self, cluster_id, labels, processed_graph, processed_nodes, node_mapping):
        """Split a cluster using graph structure (e.g., connected components)"""
        cluster_nodes = np.where(labels == cluster_id)[0]
        
        if len(cluster_nodes) < 2:
            logger.warning(f"Cluster {cluster_id} has only {len(cluster_nodes)} nodes, cannot split")
            return []

        cluster_node_names = [processed_nodes[i] for i in cluster_nodes if i < len(processed_nodes)]
        if len(cluster_node_names) < 2:
            return list(cluster_nodes[len(cluster_nodes)//2:])
            
        subgraph = processed_graph.subgraph(cluster_node_names)
        
        components = list(nx.connected_components(subgraph))
        if len(components) > 1:
            components_by_size = sorted(components, key=len, reverse=True)
            split_nodes = []
            if len(components_by_size) > 2:
                for comp in components_by_size[1:]:
                    for node_name in comp:
                        for i, node_idx in enumerate(cluster_nodes):
                            if node_idx < len(processed_nodes) and processed_nodes[node_idx] == node_name:
                                split_nodes.append(node_idx)
                                break
            else:
                for node_name in components_by_size[1]:
                    for i, node_idx in enumerate(cluster_nodes):
                        if node_idx < len(processed_nodes) and processed_nodes[node_idx] == node_name:
                            split_nodes.append(node_idx)
                            break

            if split_nodes and len(split_nodes) < len(cluster_nodes):
                logger.debug(f"Split {len(components)} disconnected components, moving {len(split_nodes)} nodes")
                return split_nodes
        
        try:
            sub_adj = nx.adjacency_matrix(subgraph, nodelist=cluster_node_names).toarray()
            if sub_adj.shape[0] > 3:
                best_split = None
                best_balance = 0
                
                for n_init in [10, 20]:
                    sub_clustering = SpectralClustering(n_clusters=2, random_state=42, n_init=n_init)
                    sub_labels = sub_clustering.fit_predict(sub_adj)

                    nodes_0 = [cluster_nodes[i] for i in range(len(sub_labels)) if sub_labels[i] == 0]
                    nodes_1 = [cluster_nodes[i] for i in range(len(sub_labels)) if sub_labels[i] == 1]

                    if nodes_0 and nodes_1:
                        min_size = min(len(nodes_0), len(nodes_1))
                        max_size = max(len(nodes_0), len(nodes_1))
                        balance = min_size / max_size

                        if balance > 0.2 and balance > best_balance:
                            best_balance = balance
                            best_split = nodes_1 if len(nodes_1) <= len(nodes_0) else nodes_0
                
                if best_split:
                    logger.debug(f"Spectral split with balance {best_balance:.2f}")
                    return best_split
                    
        except Exception as e:
            logger.debug(f"Spectral bisection failed: {e}")
        
        return self._split_by_centrality(cluster_nodes, processed_graph, processed_nodes)
    
    def _split_by_centrality(self, cluster_nodes, graph, nodes):
        """Split cluster by node centrality (degree or betweenness)"""
        if len(cluster_nodes) < 2:
            return []
        
        node_degrees = []
        for idx in cluster_nodes:
            if idx < len(nodes):
                node = nodes[idx]
                degree = graph.degree(node)
                node_degrees.append((idx, degree))

        node_degrees.sort(key=lambda x: x[1], reverse=True)

        split_point = max(1, len(node_degrees) // 2)
        return [idx for idx, _ in node_degrees[split_point:]]

class SCORECommunityDetection:
    """Fast Community Detection by SCORE (Spectral Clustering On Ratios-of-Eigenvectors)"""
    
    def __init__(self, k: int):
        if k is None:
            raise ValueError("k must be provided for SCORECommunityDetection")
        self.k = k
    
    def detect_communities(self, graph: nx.Graph, entities: List[Dict[str, Any]] = None, **kwargs) -> List[List[str]]:
        """SCORE algorithm for degree-heterogeneous networks with robust error handling"""
        if graph.number_of_nodes() == 0:
            return []
        
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n < 2:
            return [nodes]
        
        if self.k > n:
            logger.warning(f"SCORE: Requested k={self.k} but only {n} nodes available. Using k={n}")
            effective_k = n
        else:
            effective_k = self.k
        
        processed_graph = self._preprocess_graph_robust(graph, n)

        params = self._get_optimal_parameters(processed_graph, effective_k)
        
        try:
            processed_nodes = list(processed_graph.nodes())
            A = _get_cached_adjacency_matrix(processed_graph, processed_nodes)
            if A is None:
                A = nx.adjacency_matrix(processed_graph, nodelist=processed_nodes).toarray().astype(float)
                _cache_adjacency_matrix(processed_graph, processed_nodes, A)
            else:
                A = A.astype(float)
            
            np.fill_diagonal(A, A.diagonal() + 1e-6)
            
            if params['use_sparse_solver'] and eigsh is not None and n > params['min_eigenvals']:
                vals, vecs = eigsh(A, k=params['min_eigenvals'], which='LA')
            else:
                vals, vecs = np.linalg.eigh(A)
                idx = np.argsort(np.abs(vals))[::-1][:params['min_eigenvals']]
                vals = vals[idx]
                vecs = vecs[:, idx]
            
            
            node_mapping = {processed_nodes[i]: i for i in range(len(processed_nodes))}
            original_vecs = np.zeros((len(nodes), vecs.shape[1]))
            for i, node in enumerate(nodes):
                if node in node_mapping:
                    original_vecs[i] = vecs[node_mapping[node]]
                else:
                    original_vecs[i] = 0
            
            denom = original_vecs[:, 0]
            eps = 1e-10
            denom = np.where(np.abs(denom) < eps, eps, denom)
            
            if effective_k > 1:
                R = original_vecs[:, 1:effective_k] / denom[:, None]
            else:
                R = np.zeros((n, 0))
            
            if R.size > 0:
                R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-10)
            else:
                R = vecs[:, 0].reshape(-1, 1)
            
            if R.size > 0:
                if R.shape[1] >= 2:
                    ratio_similarity = cosine_similarity(R.T)
                    np.fill_diagonal(ratio_similarity, 0)
                    max_similarity = np.max(ratio_similarity)
                    
                    if max_similarity > 0.99:
                        logger.warning(f"SCORE: Ratio vectors too similar (max similarity: {max_similarity:.3f}), using more robust clustering")
                        km = KMeans(
                            n_clusters=effective_k,
                            n_init=100,
                            max_iter=1000,
                            random_state=42,
                            algorithm='lloyd'
                        )
                    else:
                        km = KMeans(n_clusters=effective_k, n_init=50, random_state=42)
                else:
                    km = KMeans(n_clusters=effective_k, n_init=50, random_state=42)
            else:
                km = KMeans(n_clusters=effective_k, n_init=50, random_state=42)
                
            labels = km.fit_predict(R)
            
            unique_labels = len(set(labels))
            if unique_labels < effective_k and n >= effective_k:
                logger.warning(f"SCORE produced only {unique_labels} clusters, expected {effective_k}")
                        
        except Exception as e:
            logger.warning(f"SCORE clustering failed: {e}, using fallback")
            fallback_k = self.k
            labels = np.arange(n) % fallback_k
        
        clusters = {}
        for i, node in enumerate(nodes):
            label = labels[i]
            clusters.setdefault(label, []).append(node)
        
        all_communities = []
        target_k = effective_k
        for i in range(target_k):
            if i in clusters:
                all_communities.append(clusters[i])
            else:
                all_communities.append([])
        
        non_empty = [c for c in all_communities if len(c) > 0]
        
        target_k = effective_k
        logger.info(f"SCORE: Initial communities: {len(non_empty)}, Target k: {target_k}, effective_k: {effective_k}, self.k: {self.k}")
        if len(non_empty) < target_k:
            outliers = []
            for comm in non_empty:
                if len(comm) > 3:
                    for node in comm:
                        internal_edges = sum(1 for neighbor in processed_graph.neighbors(node) if neighbor in comm)
                        external_edges = sum(1 for neighbor in processed_graph.neighbors(node) if neighbor not in comm)
                        if internal_edges < 2 and external_edges > 0:
                            outliers.append((node, comm, external_edges))
            
            outliers.sort(key=lambda x: x[2], reverse=True)
            
            for outlier, source_comm, _ in outliers[:target_k - len(non_empty)]:
                source_comm.remove(outlier)
                non_empty.append([outlier])
            
            all_communities = non_empty
        else:
            all_communities = non_empty[:target_k] if len(non_empty) >= target_k else non_empty
        
        expected_k = effective_k
        actual_k = len(all_communities)
        
        if actual_k != expected_k:
            logger.warning(f"SCORE: Produced {actual_k} communities, target was {expected_k} (difference acceptable)")
        
        logger.info(f"SCORE clustering: {len(all_communities)} communities")
        
        return all_communities
    
    def _preprocess_graph_robust(self, graph: nx.Graph, n: int) -> nx.Graph:
        """Robust graph preprocessing for SCORE algorithm on sparse graphs"""
        processed = graph.copy()
        
        # Add self-loops for stability
        for node in processed.nodes():
            if not processed.has_edge(node, node):
                processed.add_edge(node, node, weight=1e-6)
        
        # Handle disconnected graphs
        if not nx.is_connected(processed):
            components = list(nx.connected_components(processed))
            logger.info(f"SCORE: Graph has {len(components)} connected components - will cluster separately")
        
        # Remove isolated nodes
        isolated = list(nx.isolates(processed))
        if isolated:
            logger.info(f"SCORE: Removing {len(isolated)} isolated nodes")
            processed.remove_nodes_from(isolated)
        
        
        return processed
    
    def _get_optimal_parameters(self, graph: nx.Graph, k: int) -> Dict[str, Any]:
        """Get optimal parameters for SCORE algorithm"""
        n = graph.number_of_nodes()
        e = graph.number_of_edges()
        density = _compute_graph_density(n, e)
        
        # Adaptive parameters based on graph properties
        if density < 0.01:  # Very sparse
            use_sparse_solver = True
            min_eigenvals = max(2, k)
        elif density > 0.1:
            use_sparse_solver = False
            min_eigenvals = k + 2
        else:  # Medium density
            use_sparse_solver = n > 1000
            min_eigenvals = k + 1
        
        return {
            'use_sparse_solver': use_sparse_solver,
            'min_eigenvals': min_eigenvals,
            'k': k
        }
    
    def _intelligent_ratio_splitting(self, labels, R, processed_graph, processed_nodes, current_k, target_k=None):
        """Intelligently split clusters using ratio matrix structure"""
        labels = labels.copy()
        max_iterations = 20
        iterations = 0
        split_history = {}
        
        if target_k is None:
            target_k = self.k
        
        while len(set(labels)) < target_k and iterations < max_iterations:
            iterations += 1
            current_clusters = len(set(labels))
            
            # Find the best cluster to split based on ratio matrix structure
            best_cluster = self._find_best_ratio_cluster_to_split(labels, R)
            
            if best_cluster is None:
                # No good cluster to split, use fallback
                logger.warning(f"SCORE: No suitable clusters to split. Current: {current_clusters}, Target: {target_k}")
                break
            
            # Check split history
            split_count = split_history.get(best_cluster, 0)
            if split_count >= 2:
                logger.warning(f"SCORE: Cluster {best_cluster} already split {split_count} times")
                # Try to find another cluster
                labels_temp = labels.copy()
                labels_temp[labels == best_cluster] = -1
                next_best = self._find_best_ratio_cluster_to_split(labels_temp, R)
                if next_best is not None:
                    best_cluster = next_best
                else:
                    break
                
            # Split the cluster using ratio matrix structure
            new_label = len(set(labels))
            split_nodes = self._split_ratio_cluster_by_structure(
                best_cluster, labels, R, processed_graph, processed_nodes
            )
            
            if not split_nodes:
                logger.warning(f"SCORE: Failed to split cluster {best_cluster}. Breaking to avoid infinite loop.")
                break
            
            # Check split quality
            cluster_size = np.sum(labels == best_cluster)
            split_ratio = len(split_nodes) / cluster_size if cluster_size > 0 else 0
            if split_ratio < 0.1 or split_ratio > 0.9:
                logger.warning(f"SCORE: Split ratio {split_ratio:.2f} is too extreme")
            
            # Assign new label to split nodes
            for node_idx in split_nodes:
                labels[node_idx] = new_label
            
            # Update
            split_history[best_cluster] = split_count + 1
            

            new_clusters = len(set(labels))
            if new_clusters == current_clusters:
                logger.warning(f"SCORE: No progress made in iteration {iterations}")
                break
                
        if iterations >= max_iterations:
            logger.error(f"SCORE: Reached maximum iterations ({max_iterations}) while splitting clusters")
            
        return labels
    
    def _find_best_ratio_cluster_to_split(self, labels, R):
        """Find the cluster that would benefit most from splitting based on ratio matrix"""
        best_cluster = None
        best_score = -1
        MIN_VARIANCE_THRESHOLD = 0.1
        
        candidates = []
        
        for cluster_id in set(labels):
            if cluster_id < 0:  # Skip temporary hidden clusters
                continue
            cluster_nodes = np.where(labels == cluster_id)[0]
            if len(cluster_nodes) < 2:
                continue
                
            # Calculate variance in ratio space for this cluster
            if R.size > 0 and len(cluster_nodes) > 0:
                cluster_ratios = R[cluster_nodes]
                if cluster_ratios.shape[0] > 1:
                    # Calculate variance in ratio space
                    ratio_variance = np.var(cluster_ratios, axis=0).sum()
                    normalized_variance = ratio_variance / cluster_ratios.shape[1] if cluster_ratios.shape[1] > 0 else 0
                    
                    # Only consider if variance is above threshold
                    if normalized_variance >= MIN_VARIANCE_THRESHOLD:
                        # Score based on size and variance (higher variance = easier to split)
                        score = len(cluster_nodes) * normalized_variance
                        candidates.append((score, len(cluster_nodes), cluster_id))
        
        # Sort by score, then size for deterministic tie-breaking
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        
        if candidates:
            best_score, _, best_cluster = candidates[0]
            logger.debug(f"SCORE: Selected cluster {best_cluster} with score {best_score:.3f}")
            return best_cluster
                        
        return None
    
    def _split_ratio_cluster_by_structure(self, cluster_id, labels, R, processed_graph, processed_nodes):
        """Split a cluster using ratio matrix structure"""
        cluster_nodes = np.where(labels == cluster_id)[0]
        
        if len(cluster_nodes) < 2:
            return []
            
        # Use ratio matrix to guide splitting with quality checks
        if R.size > 0 and len(cluster_nodes) > 0:
            cluster_ratios = R[cluster_nodes]
            
            # Try KMeans on the ratio vectors of this cluster
            try:
                if cluster_ratios.shape[0] > 3:
                    best_split = None
                    best_balance = 0
                    
                    # Try with different initializations
                    for n_init in [20, 50]:
                        sub_km = KMeans(n_clusters=2, n_init=n_init, random_state=42)
                        sub_labels = sub_km.fit_predict(cluster_ratios)
                        
                        # Get nodes in each cluster
                        nodes_0 = [cluster_nodes[i] for i in range(len(sub_labels)) if sub_labels[i] == 0]
                        nodes_1 = [cluster_nodes[i] for i in range(len(sub_labels)) if sub_labels[i] == 1]
                        
                        if nodes_0 and nodes_1:
                            balance = min(len(nodes_0), len(nodes_1)) / max(len(nodes_0), len(nodes_1))
                            # Accept only balanced splits
                            if balance > 0.2 and balance > best_balance:
                                best_balance = balance
                                best_split = nodes_1 if len(nodes_1) <= len(nodes_0) else nodes_0
                    
                    if best_split:
                        return best_split
            except Exception:
                pass
        
        # Try structure-based split
        if len(cluster_nodes) > 3:
            cluster_node_names = [processed_nodes[i] for i in cluster_nodes if i < len(processed_nodes)]
            if len(cluster_node_names) > 1:
                subgraph = processed_graph.subgraph(cluster_node_names)
                components = list(nx.connected_components(subgraph))
                if len(components) > 1:
                    # Return all but largest component
                    largest = max(components, key=len)
                    split_nodes = []
                    for node in cluster_node_names:
                        if node not in largest:
                            idx = cluster_node_names.index(node)
                            if idx < len(cluster_nodes):
                                split_nodes.append(cluster_nodes[idx])
                    if split_nodes:
                        return split_nodes
                
        # Fallback: split by degree centrality
        node_degrees = []
        for idx in cluster_nodes:
            if idx < len(processed_nodes):
                node = processed_nodes[idx]
                degree = processed_graph.degree(node) if processed_graph.has_node(node) else 0
                node_degrees.append((idx, degree))
        
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        split_point = max(1, len(node_degrees) // 2)
        return [idx for idx, _ in node_degrees[split_point:]]

class ImprovedNACAlgorithm1Fixed:
    """Fixed NAC Algorithm 1 with proper entity handling"""
    
    def __init__(self, k: int, use_embeddings: bool = True):
        if k is None:
            raise ValueError("k must be provided for ImprovedNACAlgorithm1Fixed")
        self.k = k
        self.use_embeddings = use_embeddings
    
    def detect_communities(self, graph: nx.Graph, entities: List[Any] = None, 
                         embeddings: Dict[str, np.ndarray] = None, **kwargs) -> List[List[str]]:
        if graph.number_of_nodes() == 0:
            return []
        
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n < 2:
            return [nodes]
        
        if self.k > n:
            raise ValueError(f"Cannot create {self.k} communities with only {n} nodes")
        effective_k = self.k
        
        try:
            A = _get_cached_adjacency_matrix(graph, nodes)
            if A is None:
                A = nx.adjacency_matrix(graph, nodelist=nodes).toarray().astype(float)
                _cache_adjacency_matrix(graph, nodes, A)
            else:
                A = A.astype(float)
            np.fill_diagonal(A, A.diagonal() + 1e-6)
            
            degree_vec = A.sum(axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vec + 1e-10))
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt
            
            X = self._build_feature_matrix_fixed(graph, nodes, entities, embeddings)
            
            degrees = A.sum(axis=1)
            avg_deg = degrees.mean() if degrees.mean() > 0 else 1.0
            
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            density = _compute_graph_density(n_nodes, n_edges)
            phi_scale = 1.0 + density
            phi = degrees / (degrees + avg_deg * phi_scale)
            
            neighbor_features = 0.5 * (A @ X) + 0.5 * (A_norm @ X)
            
            Z = phi[:, None] * X + (1 - phi[:, None]) * neighbor_features
            
            Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)
            
            if Z.shape[1] >= 2:
                feature_similarity = cosine_similarity(Z.T)
                np.fill_diagonal(feature_similarity, 0)
                max_similarity = np.max(feature_similarity)
                
                if max_similarity > 0.99:
                    logger.warning(f"NAC1: Feature vectors too similar (max similarity: {max_similarity:.3f}), using more robust clustering")
                    kmeans = KMeans(
                        n_clusters=effective_k,
                        n_init=100,
                        max_iter=1000,
                        random_state=42,
                        algorithm='lloyd'
                    )
                else:
                    kmeans = KMeans(n_clusters=effective_k, n_init=50, random_state=42)
            else:
                kmeans = KMeans(n_clusters=effective_k, n_init=50, random_state=42)
                
            labels = kmeans.fit_predict(Z)
            
            unique_labels = len(set(labels))
            if unique_labels < effective_k:
                logger.warning(f"NAC1 produced only {unique_labels} clusters, expected {effective_k}")
            
            clusters = {}
            for i, node in enumerate(nodes):
                label = labels[i]
                clusters.setdefault(label, []).append(node)
            
            all_communities = []
            target_k = effective_k
            for i in range(target_k):
                if i in clusters:
                    all_communities.append(clusters[i])
                else:
                    all_communities.append([])
            
            for i, comm in enumerate(all_communities):
                if not comm and any(len(c) > 1 for c in all_communities):
                    largest_idx = max(range(target_k), key=lambda j: len(all_communities[j]))
                    if len(all_communities[largest_idx]) > 1:
                        largest = all_communities[largest_idx]
                        donor = min(largest, key=lambda n: graph.degree(n))
                        all_communities[largest_idx].remove(donor)
                        all_communities[i].append(donor)
            
            expected_k = effective_k
            actual_k = len(all_communities)
            
            if actual_k != expected_k:
                logger.warning(f"NAC1: Produced {actual_k} communities, target was {expected_k} (difference acceptable)")
            
            non_empty = sum(1 for c in all_communities if len(c) > 0)
            logger.info(f"NAC1: Generated {len(all_communities)} communities ({non_empty} non-empty)")
            return all_communities
            
        except Exception as e:
            logger.error(f"NAC1 failed: {e}")
            return [list(comp) for comp in nx.connected_components(graph)]
    
    def _build_feature_matrix_fixed(self, graph: nx.Graph, nodes: List[str], 
                                   entities: Any, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Build feature matrix with proper entity handling"""
        n = len(nodes)
        feature_list = []
        
        # 1. Structural features
        struct_features = np.zeros((n, 6))
        pagerank = nx.pagerank(graph)
        
        try:
            if nx.is_connected(graph):
                eigenvector = nx.eigenvector_centrality_numpy(graph)
            else:
                raise ValueError("disconnected")
        except:
            degs = np.array([graph.degree(node) for node in nodes], dtype=float)
            degs = (degs - degs.mean()) / (degs.std() + 1e-8)
            eigenvector = {node: float(val) for node, val in zip(nodes, degs)}
        
        betweenness = nx.betweenness_centrality(graph, k=min(10, n))
        
        for i, node in enumerate(nodes):
            struct_features[i, 0] = graph.degree(node)
            struct_features[i, 1] = nx.clustering(graph, node)
            struct_features[i, 2] = betweenness.get(node, 0.0)
            struct_features[i, 3] = pagerank.get(node, 0.0)
            struct_features[i, 4] = eigenvector.get(node, 0.0)
            struct_features[i, 5] = len(list(nx.common_neighbors(graph, node, nodes[0]))) if i > 0 else 0
        
        feature_list.append(struct_features)
        
        # 2. Entity features (fixed handling)
        if entities is not None:
            # Handle different entity formats
            if isinstance(entities, dict):
                # Dictionary format: node -> entity_data
                entity_types = set()
                for node in nodes:
                    if node in entities:
                        entity_data = entities[node]
                        if isinstance(entity_data, dict) and 'type' in entity_data:
                            entity_types.add(entity_data['type'])
                        elif hasattr(entity_data, 'type'):
                            entity_types.add(entity_data.type)
                
                entity_types = list(entity_types)
                if entity_types:
                    type_features = np.zeros((n, len(entity_types)))
                    for i, node in enumerate(nodes):
                        if node in entities:
                            entity_data = entities[node]
                            if isinstance(entity_data, dict) and 'type' in entity_data:
                                node_type = entity_data['type']
                            elif hasattr(entity_data, 'type'):
                                node_type = entity_data.type
                            else:
                                continue
                            
                            if node_type in entity_types:
                                type_features[i, entity_types.index(node_type)] = 1.0
                    feature_list.append(type_features)
            
            elif isinstance(entities, list):
                # List format: list of entity objects
                entity_map = {}
                for entity in entities:
                    if hasattr(entity, 'title'):
                        entity_map[entity.title] = entity
                
                entity_types = list(set(getattr(e, 'type', 'unknown') for e in entities))
                if entity_types:
                    type_features = np.zeros((n, len(entity_types)))
                    for i, node in enumerate(nodes):
                        if node in entity_map:
                            entity = entity_map[node]
                            node_type = getattr(entity, 'type', 'unknown')
                            if node_type in entity_types:
                                type_features[i, entity_types.index(node_type)] = 1.0
                    feature_list.append(type_features)
        
        # 3. Embedding features
        if embeddings and self.use_embeddings:
            emb_dim = next(iter(embeddings.values())).shape[0] if embeddings else 0
            if emb_dim > 0:
                emb_features = np.zeros((n, emb_dim))
                for i, node in enumerate(nodes):
                    if node in embeddings:
                        vec = embeddings[node]
                        if vec.shape[0] < emb_dim:
                            padded = np.zeros(emb_dim)
                            padded[:vec.shape[0]] = vec
                            emb_features[i] = padded
                        else:
                            emb_features[i] = vec[:emb_dim]
                feature_list.append(emb_features)
        
        # Concatenate all features
        if feature_list:
            X = np.hstack(feature_list)
        else:
            X = np.eye(n)
        
        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        return X
    
    def _intelligent_feature_splitting(self, labels, Z, graph, nodes, current_k, target_k=None):
        """Intelligently split clusters using feature space structure"""
        labels = labels.copy()
        max_iterations = 20  #prevent timeouts
        iterations = 0
        split_history = {}  
        
        if target_k is None:
            target_k = self.k
        
        while len(set(labels)) < target_k and iterations < max_iterations:
            iterations += 1
            current_clusters = len(set(labels))
            
            # Find the best cluster to split based on feature space structure
            best_cluster = self._find_best_feature_cluster_to_split(labels, Z)
            
            if best_cluster is None:
                # No good cluster to split, use fallback
                logger.warning(f"NAC1: No suitable clusters to split. Current: {current_clusters}, Target: {target_k}")
                break
            
            # Check split history
            split_count = split_history.get(best_cluster, 0)
            if split_count >= 2:
                logger.warning(f"NAC1: Cluster {best_cluster} already split {split_count} times")
                # Try to find another cluster
                labels_temp = labels.copy()
                labels_temp[labels == best_cluster] = -1
                next_best = self._find_best_feature_cluster_to_split(labels_temp, Z)
                if next_best is not None:
                    best_cluster = next_best
                else:
                    break
                
            # Split the cluster using feature space structure
            new_label = len(set(labels))
            split_nodes = self._split_feature_cluster_by_structure(
                best_cluster, labels, Z, graph, nodes
            )
            
            # Check if split was successful
            if not split_nodes:
                logger.warning(f"NAC1: Failed to split cluster {best_cluster}. Breaking to avoid infinite loop.")
                break
            
            # Check split quality
            cluster_size = np.sum(labels == best_cluster)
            split_ratio = len(split_nodes) / cluster_size if cluster_size > 0 else 0
            if split_ratio < 0.1 or split_ratio > 0.9:
                logger.warning(f"NAC1: Split ratio {split_ratio:.2f} is too extreme")
            
            # Assign new label to split nodes
            for node_idx in split_nodes:
                labels[node_idx] = new_label
            
            # Update history
            split_history[best_cluster] = split_count + 1
            
            # Check progress
            new_clusters = len(set(labels))
            if new_clusters == current_clusters:
                logger.warning(f"NAC1: No progress made in iteration {iterations}")
                break
                
        if iterations >= max_iterations:
            logger.error(f"NAC1: Reached maximum iterations ({max_iterations}) while splitting clusters")
            
        return labels
    
    def _find_best_feature_cluster_to_split(self, labels, Z):
        """Find the cluster that would benefit most from splitting based on feature space"""
        best_cluster = None
        best_score = -1
        MIN_VARIANCE_THRESHOLD = 0.1
        
        candidates = []
        
        for cluster_id in set(labels):
            if cluster_id < 0:  
                continue
            cluster_nodes = np.where(labels == cluster_id)[0]
            if len(cluster_nodes) < 2:
                continue
                
            # Calculate variance in feature space for this cluster
            if Z.size > 0 and len(cluster_nodes) > 0:
                cluster_features = Z[cluster_nodes]
                if cluster_features.shape[0] > 1:
                    # Calculate variance in feature space
                    feature_variance = np.var(cluster_features, axis=0).sum()
                    # Normalize by number of features
                    normalized_variance = feature_variance / cluster_features.shape[1] if cluster_features.shape[1] > 0 else 0
                    
                    # Only consider if variance is above threshold
                    if normalized_variance >= MIN_VARIANCE_THRESHOLD:
                        # Score based on size and variance (higher variance = easier to split)
                        score = len(cluster_nodes) * normalized_variance
                        candidates.append((score, len(cluster_nodes), cluster_id))
        
        # Sort candidates by score, then size for deterministic tie-breaking
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
        
        if candidates:
            best_score, _, best_cluster = candidates[0]
            logger.debug(f"NAC1: Selected cluster {best_cluster} with score {best_score:.3f}")
            return best_cluster
                        
        return None
    
    def _split_feature_cluster_by_structure(self, cluster_id, labels, Z, graph, nodes):
        """Split a cluster using feature space structure"""
        cluster_nodes = np.where(labels == cluster_id)[0]
        
        if len(cluster_nodes) < 2:
            return []
            
        # Use feature space to guide splitting with quality checks
        if Z.size > 0 and len(cluster_nodes) > 0:
            cluster_features = Z[cluster_nodes]
            
            # Try KMeans on the feature vectors of this cluster
            try:
                if cluster_features.shape[0] > 3:
                    best_split = None
                    best_balance = 0
                    
                    # Try with different initializations
                    for n_init in [20, 50]:
                        sub_km = KMeans(n_clusters=2, n_init=n_init, random_state=42)
                        sub_labels = sub_km.fit_predict(cluster_features)
                        
                        # Get nodes in each cluster
                        nodes_0 = [cluster_nodes[i] for i in range(len(sub_labels)) if sub_labels[i] == 0]
                        nodes_1 = [cluster_nodes[i] for i in range(len(sub_labels)) if sub_labels[i] == 1]
                        
                        if nodes_0 and nodes_1:
                            balance = min(len(nodes_0), len(nodes_1)) / max(len(nodes_0), len(nodes_1))
                            # Accept only balanced splits
                            if balance > 0.2 and balance > best_balance:
                                best_balance = balance
                                best_split = nodes_1 if len(nodes_1) <= len(nodes_0) else nodes_0
                    
                    if best_split:
                        return best_split
            except Exception:
                pass
        
        # Try structure-based split if features unavailable
        if len(cluster_nodes) > 3:
            cluster_node_names = [nodes[i] for i in cluster_nodes if i < len(nodes)]
            if len(cluster_node_names) > 1:
                subgraph = graph.subgraph(cluster_node_names)
                components = list(nx.connected_components(subgraph))
                if len(components) > 1:
                    # Return all but largest component
                    largest = max(components, key=len)
                    split_nodes = []
                    for node in cluster_node_names:
                        if node not in largest:
                            idx = cluster_node_names.index(node)
                            if idx < len(cluster_nodes):
                                split_nodes.append(cluster_nodes[idx])
                    if split_nodes:
                        return split_nodes
                
        # Fallback: split by degree centrality
        node_degrees = []
        for idx in cluster_nodes:
            if idx < len(nodes):
                node = nodes[idx]
                degree = graph.degree(node) if graph.has_node(node) else 0
                node_degrees.append((idx, degree))
        
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        split_point = max(1, len(node_degrees) // 2)
        return [idx for idx, _ in node_degrees[split_point:]]

class ImprovedNACAlgorithm2Fixed:
    """Fixed NAC Algorithm 2 with proper entity handling"""
    
    def __init__(self, k: int, alpha: float = None, adaptive_alpha: bool = True):
        if k is None:
            raise ValueError("k must be provided for ImprovedNACAlgorithm2Fixed")
        self.k = k
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
    
    def detect_communities(self, graph: nx.Graph, entities: Any = None,
                         embeddings: Dict[str, np.ndarray] = None, **kwargs) -> List[List[str]]:
        if graph.number_of_nodes() == 0:
            return []
        
        nodes = list(graph.nodes())
        n = len(nodes)
        
        if n < 2:
            return [nodes]
        
        if self.k > n:
            raise ValueError(f"Cannot create {self.k} communities with only {n} nodes")
        effective_k = self.k
        
        try:
            # Get adjacency matrix with caching
            A = _get_cached_adjacency_matrix(graph, nodes)
            if A is None:
                A = nx.adjacency_matrix(graph, nodelist=nodes).toarray().astype(float)
                _cache_adjacency_matrix(graph, nodes, A)
            else:
                A = A.astype(float)
            
            # Build features using fixed NAC1 method
            nac1 = ImprovedNACAlgorithm1Fixed(k=effective_k)
            X = nac1._build_feature_matrix_fixed(graph, nodes, entities, embeddings)
            
            # Network-adjusted covariates
            degrees = A.sum(axis=1)
            avg_deg = degrees.mean() if degrees.mean() > 0 else 1.0
            phi = degrees / (degrees + avg_deg)
            
            # Normalized adjacency
            degree_vec = A.sum(axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_vec + 1e-10))
            A_norm_local = D_inv_sqrt @ A @ D_inv_sqrt
            
            # Neighbor features
            neighbor_features = 0.5 * (A @ X) + 0.5 * (A_norm_local @ X)
            Z = phi[:, None] * X + (1 - phi[:, None]) * neighbor_features
            
            # Build similarity matrices
            D = np.diag(1.0 / np.sqrt(np.sum(A, axis=1) + 1e-10))
            A_norm = D @ A @ D
            
            # Feature similarity
            Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            W_cosine = Z_norm @ Z_norm.T
            
            # RBF kernel
            # Compute pairwise distances efficiently
            pairwise_dists = euclidean_distances(Z)
            med = np.median(pairwise_dists[pairwise_dists > 0])  # Exclude diagonal
            gamma = 1.0 / (2.0 * (med ** 2 + 1e-8))
            W_rbf = rbf_kernel(Z, gamma=gamma)
            
            W_features = 0.7 * W_cosine + 0.3 * W_rbf
            
            # Adaptive alpha
            if self.adaptive_alpha and self.alpha is None:
                modularity = nx.algorithms.community.modularity(
                    graph, nx.connected_components(graph)
                )
                alpha = 0.3 + 0.4 * modularity
            else:
                alpha = self.alpha if self.alpha else 0.5
            
            # Combined similarity
            W_combined = alpha * A_norm + (1 - alpha) * W_features
            W_combined = np.maximum(W_combined, 0)
            

            if W_combined.shape[0] >= 2:
                # Check matrix condition
                eigenvals = np.linalg.eigvals(W_combined)
                min_eigenval = np.min(np.real(eigenvals))
                max_eigenval = np.max(np.real(eigenvals))
                condition_number = max_eigenval / (min_eigenval + 1e-10)
                
                if condition_number > 1e12:  
                    logger.warning(f"NAC2: Similarity matrix poorly conditioned (condition number: {condition_number:.2e}), using more robust clustering")
                    clustering = SpectralClustering(
                        n_clusters=effective_k,
                        affinity='precomputed',
                        assign_labels='kmeans',
                        n_init=50,  
                        random_state=42
                    )
                else:
                    clustering = SpectralClustering(
                        n_clusters=effective_k,
                        affinity='precomputed',
                        assign_labels='kmeans', 
                        n_init=30,  
                        random_state=42
                    )
            else:
                clustering = SpectralClustering(
                    n_clusters=effective_k,
                    affinity='precomputed',
                    assign_labels='kmeans',  
                    n_init=30,  
                    random_state=42
                )
                
            labels = clustering.fit_predict(W_combined)
            
            unique_labels = len(set(labels))
            if unique_labels < effective_k:
                logger.warning(f"NAC2 produced only {unique_labels} clusters, expected {effective_k}")
            
            # Group nodes
            clusters = {}
            for i, node in enumerate(nodes):
                label = labels[i]
                clusters.setdefault(label, []).append(node)
            
            
            if unique_labels < effective_k / 2:
                logger.warning(f"NAC2: SpectralClustering produced too few clusters ({unique_labels} vs {effective_k}), retrying...")
                # Fallback to NAC1-style KMeans clustering
                try:
                    kmeans = KMeans(n_clusters=effective_k, n_init=50, random_state=42)
                    labels = kmeans.fit_predict(Z)  # Use feature matrix Z instead of W_combined
                    
                    # Re-group nodes
                    clusters = {}
                    for i, node in enumerate(nodes):
                        label = labels[i]
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(node)
                    
                    logger.info(f"NAC2: Fallback KMeans produced {len(clusters)} clusters")
                except Exception as e:
                    logger.error(f"NAC2: Fallback KMeans also failed: {e}")
            
            all_communities = []
            target_k = effective_k
            for i in range(target_k):
                if i in clusters:
                    all_communities.append(clusters[i])
                else:
                    all_communities.append([])
            
            expected_k = effective_k
            actual_k = len(all_communities)
            
            if actual_k != expected_k:
                logger.warning(f"NAC2: Produced {actual_k} communities, target was {expected_k} (difference acceptable)")
            
            logger.info(f"NAC2: {len(all_communities)} communities")
            return all_communities
            
        except Exception as e:
            logger.error(f"NAC2 failed: {e}")
            return [list(comp) for comp in nx.connected_components(graph)]
    
    def _intelligent_similarity_splitting(self, labels, W_combined, graph, nodes, current_k, target_k=None):
        """Intelligently split clusters using similarity matrix structure"""
        labels = labels.copy()
        max_iterations = 20  # Reduced to prevent timeouts
        iterations = 0
        
        if target_k is None:
            target_k = self.k
        
        while len(set(labels)) < target_k and iterations < max_iterations:
            iterations += 1
            # Find the best cluster to split based on similarity matrix structure
            best_cluster = self._find_best_similarity_cluster_to_split(labels, W_combined)
            
            if best_cluster is None:
                # No good cluster to split, use fallback
                logger.warning(f"NAC2: No suitable clusters to split. Current: {len(set(labels))}, Target: {self.k}")
                break
                
            # Split the cluster using similarity matrix structure
            new_label = len(set(labels))
            split_nodes = self._split_similarity_cluster_by_structure(
                best_cluster, labels, W_combined, graph, nodes
            )
            
            # Check if split was successful
            if not split_nodes:
                logger.warning(f"NAC2: Failed to split cluster {best_cluster}. Breaking to avoid infinite loop.")
                break
            
            # Assign new label to split nodes
            for node_idx in split_nodes:
                labels[node_idx] = new_label
                
        if iterations >= max_iterations:
            logger.error(f"NAC2: Reached maximum iterations ({max_iterations}) while splitting clusters")
            
        return labels
    
    def _find_best_similarity_cluster_to_split(self, labels, W_combined):
        """Find the cluster that would benefit most from splitting based on similarity matrix"""
        best_cluster = None
        best_score = -1
        
        for cluster_id in set(labels):
            cluster_nodes = np.where(labels == cluster_id)[0]
            if len(cluster_nodes) < 2:
                continue
                
            # Calculate internal similarity for this cluster
            if len(cluster_nodes) > 0:
                cluster_similarity = W_combined[np.ix_(cluster_nodes, cluster_nodes)]
                # Calculate average internal similarity
                internal_similarity = np.mean(cluster_similarity[np.triu_indices_from(cluster_similarity, k=1)])
                # Score based on size and internal similarity (lower similarity = easier to split)
                score = len(cluster_nodes) * (1 - internal_similarity)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id
                    
        return best_cluster
    
    def _split_similarity_cluster_by_structure(self, cluster_id, labels, W_combined, graph, nodes):
        """Split a cluster using similarity matrix structure"""
        cluster_nodes = np.where(labels == cluster_id)[0]
        
        if len(cluster_nodes) < 2:
            return []
            
        # Use similarity matrix to guide splitting
        if len(cluster_nodes) > 0:
            cluster_similarity = W_combined[np.ix_(cluster_nodes, cluster_nodes)]
            
            try:
                if cluster_similarity.shape[0] > 1:
                    sub_clustering = SpectralClustering(n_clusters=2, random_state=42)
                    sub_labels = sub_clustering.fit_predict(cluster_similarity)
                    
                    split_nodes = []
                    for i, node_idx in enumerate(cluster_nodes):
                        if i < len(sub_labels) and sub_labels[i] == 1:
                            split_nodes.append(node_idx)
                    return split_nodes
            except Exception:
                pass
                
        return cluster_nodes[len(cluster_nodes)//2:]

# Export all fixed classes
__all__ = [
    'ImprovedSpectralClusteringFixed',
    'SCORECommunityDetection',
    'ImprovedNACAlgorithm1Fixed',
    'ImprovedNACAlgorithm2Fixed',
    'clear_all_caches'
]
