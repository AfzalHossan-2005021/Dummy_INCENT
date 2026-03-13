import numpy as np
import pandas as pd
import anndata
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from .INCENT import pairwise_align

def get_surviving_indices(slice1, slice2):
    """
    Finds the original indices of cells that survive cell-type filtering.
    """
    shared_cell_types = pd.Index(slice1.obs['cell_type_annot']).unique().intersection(pd.Index(slice2.obs['cell_type_annot']).unique())
    survivors_1 = np.where(slice1.obs['cell_type_annot'].isin(shared_cell_types))[0]
    survivors_2 = np.where(slice2.obs['cell_type_annot'].isin(shared_cell_types))[0]
    return survivors_1, survivors_2

def is_dual_hemisphere(adata: anndata.AnnData, silhouette_threshold: float = 0.40) -> tuple[bool, np.ndarray]:
    """
    Detects if a slice contains both hemispheres based on spatial coordinates.
    Returns a boolean and the cluster labels if True.
    """
    coords = adata.obsm['spatial']
    
    # Gaussian Mixture Models handle unequal cluster sizes and non-spherical shapes better than KMeans
    gmm = GaussianMixture(n_components=2, random_state=42, covariance_type='full', n_init=5)
    labels = gmm.fit_predict(coords)
    
    # Calculate how distinct the two clusters are
    score = silhouette_score(coords, labels)
    
    # If the score is high, there are two distinct spatial blobs (dual hemisphere)
    # We lowered the threshold slightly to 0.40 to account for unbalanced tissue sizes and biological noise
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
        
        surv_A0, surv_B0_sub = get_surviving_indices(sliceA, sliceB_0)
        surv_A1, surv_B1_sub = get_surviving_indices(sliceA, sliceB_1)
        
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
        
        print(f"[Smart Align] Chose Hemisphere {'0' if cost_0 < cost_1 else '1'} of Slice B (Cost: {min(cost_0, cost_1):.4f})")
        
        # Reconstruct full Pi matrix - completely robust to dropped cell types
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        if cost_0 < cost_1:
            full_pi[np.ix_(surv_A0, idx_B0[surv_B0_sub])] = best_pi
        else:
            full_pi[np.ix_(surv_A1, idx_B1[surv_B1_sub])] = best_pi
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 2: Slice A is dual, Slice B is single
    elif is_dual_A and not is_dual_B:
        print("[Smart Align] Splitting Slice A and finding best hemisphere match...")
        idx_A0 = np.where(labels_A == 0)[0]
        idx_A1 = np.where(labels_A == 1)[0]
        
        sliceA_0 = sliceA[idx_A0].copy()
        sliceA_1 = sliceA[idx_A1].copy()
        
        surv_A0_sub, surv_B0 = get_surviving_indices(sliceA_0, sliceB)
        surv_A1_sub, surv_B1 = get_surviving_indices(sliceA_1, sliceB)
        
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
        
        print(f"[Smart Align] Chose Hemisphere {'0' if cost_0 < cost_1 else '1'} of Slice A (Cost: {min(cost_0, cost_1):.4f})")
        
        # Reconstruct full Pi matrix - completely robust to dropped cell types
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        if cost_0 < cost_1:
            full_pi[np.ix_(idx_A0[surv_A0_sub], surv_B0)] = best_pi
        else:
            full_pi[np.ix_(idx_A1[surv_A1_sub], surv_B1)] = best_pi
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 3: Both are dual, or both are single
    else:
        print("[Smart Align] Slices are structurally identical (both single or both dual). Proceeding with standard alignment.")
        surv_A, surv_B = get_surviving_indices(sliceA, sliceB)
        best_res_list = list(pairwise_align(sliceA, sliceB, **kwargs))
        
        # Reconstruct full Pi matrix in case standard pairwise_align dropped cells
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(surv_A, surv_B)] = best_res_list[0]
        best_res_list[0] = full_pi

    # Obey the user's original request for return objects
    if not original_return_obj:
        return best_res_list[0] # Just the pi matrix
    return tuple(best_res_list)