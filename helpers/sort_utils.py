import numpy as np
import json

def bar_sort_order(mat: np.ndarray) -> np.ndarray:
    """Return indices that order samples by winning component, then activity."""
    winners = np.argmax(mat, axis=1)
    order = []
    for comp in range(mat.shape[1]):
        idx = np.argsort(-mat[:, comp])
        order.extend(idx[winners[idx] == comp])
    return np.asarray(order, dtype=int)


def get_alphabetical_sort(sample_ids: list) -> np.ndarray:
    """Sort samples alphabetically by ID."""
    return np.argsort(sample_ids)


def get_cancer_type_sort(H: np.ndarray, cancer_types: list) -> np.ndarray:
    """Sort by cancer type, then by component contribution within each type."""
    cancer_order = np.argsort(cancer_types)
    # For samples with same cancer, sub-sort by dominant component
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    
    # Create a composite key for stable sub-sorting
    result = np.zeros(len(sample_ids), dtype=int)
    unique_cancers = sorted(set(cancer_types))
    
    for i, cancer in enumerate(unique_cancers):
        mask = np.array(cancer_types) == cancer
        sub_indices = np.where(mask)[0]
        sub_order = bar_sort_order(H_ord[sub_indices])
        result[cancer_order[i:i+len(sub_indices)]] = sub_indices[sub_order]
        
    return result


def get_organ_system_sort(H: np.ndarray, cancer_types: list, config_path: str) -> np.ndarray:
    """Sort by organ system groupings, then by component within each group."""
    from helpers.data_loader import load_cfg
    
    # Load config
    cfg = load_cfg(config_path)
    organ_systems = []
    
    # Load organ system data
    try:
        with open(cfg.get("JSON_FILENAME_VOCAB") or "vocab.json", "r") as f:
            grouping_data = json.load(f).get("organ_system_groupings", [])
            
        # Map cancer types to organ systems
        cancer_to_organ = {}
        for group in grouping_data:
            for code in group["cancer_codes"]:
                code_prefix = code[:4]  # Get first 4 chars
                cancer_to_organ[code_prefix] = group["group_name"]
        
        # Get organ system for each sample
        for cancer in cancer_types:
            prefix = cancer[:4]
            organ_systems.append(cancer_to_organ.get(prefix, "Unknown"))
            
        # Create compound sort key: organ system first, then component contribution
        organ_order = np.argsort(organ_systems)
        
        # Now sub-sort each organ system group by component contribution
        result = np.zeros(len(cancer_types), dtype=int)
        unique_organs = sorted(set(organ_systems))
        
        for organ in unique_organs:
            mask = np.array(organ_systems) == organ
            indices = np.where(mask)[0]
            
            # For this organ system, sort by dominant component
            comp_order = np.argsort(-H.sum(axis=0))
            H_ord = H[:, comp_order]
            sub_order = bar_sort_order(H_ord[indices])
            
            # Place these indices in the right position
            organ_indices = np.where(np.array(organ_systems) == organ)[0]
            result[organ_order[:len(indices)]] = organ_indices[sub_order]
            organ_order = organ_order[len(indices):]
            
        return result
        
    except Exception as e:
        print(f"Error in organ system sorting: {e}")
        # Fall back to component sorting
        return bar_sort_order(H)


def get_embryonic_layer_sort(H: np.ndarray, cancer_types: list, config_path: str) -> np.ndarray:
    """Sort by embryonic layer, then by component within each layer."""
    # This would be similar to get_organ_system_sort but using embryonic layer data
    # Implementation would depend on how your embryonic layer data is structured
    pass


def get_sample_order(sort_method: str, H: np.ndarray, sample_ids: list, 
                    cancer_types: list, config_path: str) -> np.ndarray:
    """Get sample ordering indices based on different sorting methods."""
    
    if sort_method == "component":
        # Sort by highest component contribution
        comp_order = np.argsort(-H.sum(axis=0))
        H_ord = H[:, comp_order]
        return bar_sort_order(H_ord)
    
    elif sort_method == "alphabetical":
        # Alphabetical by sample ID
        return get_alphabetical_sort(sample_ids)
    
    elif sort_method == "cancer_type":
        # Sort by cancer type then by component within each type
        return get_cancer_type_sort(H, cancer_types)
    
    elif sort_method == "organ_system":
        # Sort by organ system groupings
        return get_organ_system_sort(H, cancer_types, config_path)
        
    elif sort_method == "embryonic_layer":
        # Sort by embryonic layer
        return get_embryonic_layer_sort(H, cancer_types, config_path)
    
    # Default to component sorting if method not recognized
    return bar_sort_order(H)