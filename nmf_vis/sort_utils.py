import numpy as np
import json


def bar_sort_order(mat: np.ndarray) -> np.ndarray:
    """Return indices that order samples by winning component, then activity."""
    winners = np.argmax(mat, axis=1)
    order = []
    for comp in range(mat.shape[1]):
        # Get samples sorted by activity for the current component
        idx = np.argsort(-mat[:, comp])
        # Filter this list, keeping only samples whose "winner" is the current component
        order.extend(idx[winners[idx] == comp])
    return np.asarray(order, dtype=int)


def get_alphabetical_sort(sample_ids: list) -> np.ndarray:
    """Sort samples alphabetically by ID."""
    return np.argsort(sample_ids)


def get_cancer_type_sort(H: np.ndarray, cancer_types: list) -> np.ndarray:
    """Correctly sort by cancer type, then by component within each type."""
    final_order = []
    # Pre-sort component columns by total activity for consistent sub-sorting
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    # Get a sorted list of unique cancer types for the primary sort order
    unique_cancers = sorted(list(set(cancer_types)))

    for cancer in unique_cancers:
        # Find original indices for all samples of the current cancer type
        sub_indices = np.where(np.array(cancer_types) == cancer)[0]

        if len(sub_indices) == 0:
            continue

        # Get the subset of the H matrix for these samples
        H_subset = H_ord[sub_indices, :]

        # Sort these samples by component. sub_order has indices relative to the subset.
        sub_order = bar_sort_order(H_subset)

        # Map these relative indices back to the original indices and add to the final list
        final_order.extend(sub_indices[sub_order])

    return np.array(final_order, dtype=int)


def get_organ_system_sort(
    H: np.ndarray, cancer_types: list, sample_ids: list, config_path: str
) -> np.ndarray:
    """Correctly sort by organ system, then by component within each group."""
    from nmf_vis.data_utils import load_cfg

    try:
        cfg = load_cfg(config_path)
        # It's safer to use the sample_ids to map to cancer codes than the full cancer_types string
        cancer_codes = [sid[:4] for sid in sample_ids]

        # Load organ system data from the correct JSON file specified in config
        organ_system_file = cfg.get(
            "JSON_FILENAME_ORGAN_SYSTEM", "tissue_source_tcga.json"
        )
        with open(organ_system_file, "r") as f:
            grouping_data = json.load(f).get("organ_system_groupings", [])

        # Map cancer codes to organ systems
        code_to_organ = {}
        for group in grouping_data:
            for code in group["cancer_codes"]:
                code_to_organ[code] = group["group_name"]

        # Get organ system for each sample, defaulting to "Unknown"
        organ_systems = [code_to_organ.get(code, "Unknown") for code in cancer_codes]

        # --- This part now mirrors the corrected cancer_type_sort logic ---
        final_order = []
        comp_order = np.argsort(-H.sum(axis=0))
        H_ord = H[:, comp_order]

        unique_organs = sorted(list(set(organ_systems)))

        for organ in unique_organs:
            sub_indices = np.where(np.array(organ_systems) == organ)[0]

            if len(sub_indices) == 0:
                continue

            H_subset = H_ord[sub_indices, :]
            sub_order = bar_sort_order(H_subset)
            final_order.extend(sub_indices[sub_order])

        return np.array(final_order, dtype=int)

    except Exception as e:
        # print(f"Error in organ system sorting: {e}. Falling back to component sort.")
        comp_order = np.argsort(-H.sum(axis=0))
        H_ord = H[:, comp_order]
        return bar_sort_order(H_ord)


def get_embryonic_layer_sort(
    H: np.ndarray, cancer_types: list, config_path: str
) -> np.ndarray:
    """Sort by embryonic layer, then by component within each layer."""
    # Fall back to component sorting for now
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    return bar_sort_order(H_ord)


def get_sample_order(
    sort_method: str,
    H: np.ndarray,
    sample_ids: list,
    cancer_types: list,
    config_path: str,
) -> np.ndarray:
    """Get sample ordering indices based on different sorting methods."""

    # Always pre-sort components for consistent sub-sorting
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    if sort_method == "component":
        return bar_sort_order(H_ord)

    elif sort_method == "alphabetical":
        return get_alphabetical_sort(sample_ids)

    elif sort_method == "cancer_type":
        return get_cancer_type_sort(H, cancer_types)

    elif sort_method == "organ_system":
        # Pass sample_ids to organ sort to reliably get cancer codes
        return get_organ_system_sort(H, cancer_types, sample_ids, config_path)

    elif sort_method == "embryonic_layer":
        return get_embryonic_layer_sort(H, cancer_types, config_path)

    # Default to component sorting if method not recognized
    return bar_sort_order(H_ord)
