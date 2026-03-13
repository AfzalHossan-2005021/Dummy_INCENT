import numpy as np
import anndata
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from incent.INCENT import pairwise_align

def is_dual_hemisphere(adata: anndata.AnnData, silhouette_threshold: float = 0.5) -> tuple[bool, np.ndarray]:
    """
    Detects if a slice contains both hemispheres based on spatial coordinates.
    Returns a boolean and the cluster labels if True.
    """
    coords = adata.obsm['spatial']
    
    # Try to find 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    # Calculate how distinct the two clusters are
    score = silhouette_score(coords, labels)
    
    # If the score is high, there are two distinct, well-separated spatial blobs (dual hemisphere)
    is_dual = score > silhouette_threshold
    
    return is_dual, labels

def smart_pairwise_align(sliceA, sliceB, **kwargs):
    """
    Automatically detects dual/single hemisphere mismatch and flexibly aligns them.
    Kwargs are passed directly to INCENT's pairwise_align.
    """
    
    # 1. Detect structures
    is_dual_A, labels_A = is_dual_hemisphere(sliceA)
    is_dual_B, labels_B = is_dual_hemisphere(sliceB)
    
    print(f"[Smart Align] Slice A dual: {is_dual_A} | Slice B dual: {is_dual_B}")
    
    # Ensure return_obj is True so we can compare costs
    original_return_obj = kwargs.get('return_obj', False)
    kwargs['return_obj'] = True
    
    # We will use the 'final_obj_gene' to evaluate the best mapping (index 4 in the returned tuple)
    
    # Case 1: Slice A is single, Slice B is dual
    if not is_dual_A and is_dual_B:
        print("[Smart Align] Splitting Slice B and finding best hemisphere match...")
        idx_B0 = np.where(labels_B == 0)[0]
        idx_B1 = np.where(labels_B == 1)[0]
        
        sliceB_0 = sliceB[idx_B0].copy()
        sliceB_1 = sliceB[idx_B1].copy()
        
        # Test mapping to hemisphere 0
        kwargs['sliceB_name'] = kwargs.get('sliceB_name', 'B') + "_hemi0"
        res0 = pairwise_align(sliceA, sliceB_0, **kwargs)
        cost_0 = res0[4] 
        
        # Test mapping to hemisphere 1
        kwargs['sliceB_name'] = kwargs.get('sliceB_name', 'B') + "_hemi1"
        res1 = pairwise_align(sliceA, sliceB_1, **kwargs)
        cost_1 = res1[4]
        
        # Choose winner
        best_res = res0 if cost_0 < cost_1 else res1
        best_pi = best_res[0]
        winning_idx = idx_B0 if cost_0 < cost_1 else idx_B1
        
        print(f"[Smart Align] Chose Hemisphere {'0' if cost_0 < cost_1 else '1'} of Slice B (Cost: {min(cost_0, cost_1):.4f})")
        
        # Reconstruct full Pi matrix
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[:, winning_idx] = best_pi
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 2: Slice A is dual, Slice B is single
    elif is_dual_A and not is_dual_B:
        print("[Smart Align] Splitting Slice A and finding best hemisphere match...")
        idx_A0 = np.where(labels_A == 0)[0]
        idx_A1 = np.where(labels_A == 1)[0]
        
        sliceA_0 = sliceA[idx_A0].copy()
        sliceA_1 = sliceA[idx_A1].copy()
        
        # Test mapping to hemisphere 0
        kwargs['sliceA_name'] = kwargs.get('sliceA_name', 'A') + "_hemi0"
        res0 = pairwise_align(sliceA_0, sliceB, **kwargs)
        cost_0 = res0[4] 
        
        # Test mapping to hemisphere 1
        kwargs['sliceA_name'] = kwargs.get('sliceA_name', 'A') + "_hemi1"
        res1 = pairwise_align(sliceA_1, sliceB, **kwargs)
        cost_1 = res1[4]
        
        # Choose winner
        best_res = res0 if cost_0 < cost_1 else res1
        best_pi = best_res[0]
        winning_idx = idx_A0 if cost_0 < cost_1 else idx_A1
        
        print(f"[Smart Align] Chose Hemisphere {'0' if cost_0 < cost_1 else '1'} of Slice A (Cost: {min(cost_0, cost_1):.4f})")
        
        # Reconstruct full Pi matrix
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[winning_idx, :] = best_pi
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 3: Both are dual, or both are single
    else:
        print("[Smart Align] Slices are structurally identical (both single or both dual). Proceeding with standard alignment.")
        best_res_list = list(pairwise_align(sliceA, sliceB, **kwargs))

    # Obey the user's original request for return objects
    if not original_return_obj:
        return best_res_list[0] # Just the pi matrix
    return tuple(best_res_list)