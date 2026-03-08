import numpy as np
import cv2

def postprocess_3class(
    pred_mask: np.ndarray,
    background_id=0,
    class_ids=(1, 2),
    priority=(2, 1, 0),      # who wins if masks overlap after processing
    close_k=5,
    min_area=200,
):
    """
    pred_mask: HxW int labels in {0,1,2}
    Returns: HxW int labels
    """
  
    def fill_holes_binary(mask_bool: np.ndarray) -> np.ndarray:
        # True = foreground
        mask = (mask_bool.astype(np.uint8) * 255)
        inv = cv2.bitwise_not(mask)

        h, w = inv.shape
        flood = inv.copy()
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff, (0, 0), 0)   # remove exterior background
        holes = (flood != 0)                  # remaining are holes
        return mask_bool | holes

    def close_binary(mask_bool: np.ndarray, k: int) -> np.ndarray:
        kernel = np.ones((k, k), np.uint8)
        m = (mask_bool.astype(np.uint8) * 255)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        return closed > 0

    def remove_small_components_binary(mask_bool: np.ndarray, min_area: int) -> np.ndarray:
        m = mask_bool.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m, dtype=np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        return keep.astype(bool)

    H, W = pred_mask.shape
    processed = {}

    # process each foreground class as binary
    for c in class_ids:
        m = (pred_mask == c)
        if close_k and close_k > 1:
            m = close_binary(m, close_k)          # merge small gaps
        m = fill_holes_binary(m)                  # fill surrounded holes
        if min_area and min_area > 0:
            m = remove_small_components_binary(m, min_area)  # drop tiny islands
        processed[c] = m

    # rebuild final mask using priority
    out = np.full((H, W), background_id, dtype=np.int64)
    for c in priority:
        if c == background_id:
            continue
        out[processed.get(c, False)] = c
    return out
