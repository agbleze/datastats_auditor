import numpy as np
from scipy.spatial.distance import jensenshannon 
from scipy.stats import wasserstein_distance, wasserstein_distance_nd



def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p /= p.sum()
    q /= q.sum()
    res = np.sum(p * np.log(p / q))  
    return res


def kl_divergence_between_distributions(df1, df2, field_name, labels):
    p = df1[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)
    q = df2[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)
    kl = kl_divergence(p, q)
    return kl


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    #js = 0.5 * (kl_pm + kl_qm)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js

def js_divergence_between_distributions(df1, df2, field_name, labels):
    p = df1[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)
    q = df2[field_name].value_counts(normalize=True).reindex(labels, fill_value=0)
    js = js_divergence(p, q)
    return js


def compute_spatial_drift(spatial_A, spatial_B,
                          xy_colname="heatmap",
                          x_colname="px",
                          y_colname="py",
                          ):
    H_A, H_B = spatial_A[xy_colname], spatial_B[xy_colname]
    px_A, px_B = spatial_A[x_colname], spatial_B[x_colname]
    py_A, py_B = spatial_A[y_colname], spatial_B[y_colname]

    # 2D JS divergence
    js_2d = jensenshannon(H_A.ravel(), H_B.ravel())**2

    # 1D JS on marginals
    js_x = jensenshannon(px_A, px_B)**2
    js_y = jensenshannon(py_A, py_B)**2

    # 1D Wasserstein
    bins_1d = np.linspace(0, 1, len(px_A))
    w1_x = wasserstein_distance(bins_1d, bins_1d, px_A, px_B)
    w1_y = wasserstein_distance(bins_1d, bins_1d, py_A, py_B)

    # 2D Wasserstein
    x_centers = 0.5 * (spatial_A["xedges"][:-1] + spatial_A["xedges"][1:])
    y_centers = 0.5 * (spatial_A["yedges"][:-1] + spatial_A["yedges"][1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    support = np.stack([X.ravel(), Y.ravel()], axis=1)

    w1_2d = wasserstein_distance_nd(
        support,
        support,
        u_weights=H_A.ravel(),
        v_weights=H_B.ravel(),
    )
    combined = js_2d + 0.5*(js_x + js_y) + 0.5*(w1_x + w1_y) + w1_2d

    return {"js_2d": js_2d,
            "js_x": js_x,
            "js_y": js_y,
            "w1_x": w1_x,
            "w1_y": w1_y,
            "w1_2d": w1_2d,
            "combined_score": combined,
            }



def compute_quadrant_masses(heatmap):
    h = heatmap
    mid = h.shape[0] // 2

    Q1 = h[:mid, :mid].sum()
    Q2 = h[:mid, mid:].sum()
    Q3 = h[mid:, :mid].sum()
    Q4 = h[mid:, mid:].sum()

    return np.array([Q1, Q2, Q3, Q4])


def compute_quadrant_drift(spatial_A, spatial_B):
    qA = compute_quadrant_masses(spatial_A["heatmap"])
    qB = compute_quadrant_masses(spatial_B["heatmap"])
    # Normalize
    qA = qA / qA.sum()
    qB = qB / qB.sum()
    js_quad = jensenshannon(qA, qB)**2
    l1_quad = np.abs(qA - qB).sum()
    return {"quadrant_A": qA,
            "quadrant_B": qB,
            "js_quadrant": js_quad,
            "l1_quadrant": l1_quad,
            }
