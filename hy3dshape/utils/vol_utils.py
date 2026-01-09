import numpy as np
import torch
from ThirdParty.michelangelo.graphics.primitives import generate_dense_grid_points
from sklearn.cluster import DBSCAN

def get_vol(bounds=(-0.5, 0.0, -0.5, 0.5, 1.0, 0.5), octree_depth=6):

    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=octree_depth,
        indexing="ij"
    )
    xyz_samples = torch.FloatTensor(xyz_samples)  # ((2^d)+1)^3

    return xyz_samples

def get_co(vox, bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), dtype = torch.float32):

    bbox_min = torch.tensor(bounds[0:3]).to(vox.device)
    bbox_max = torch.tensor(bounds[3:6]).to(vox.device)
    bbox_size = bbox_max - bbox_min

    # ind = torch.argwhere(vox)
    # ind = ind.to(dtype) / (vox.shape[0]) * bbox_size + bbox_min
    ind = vox
    ind = ind.to(dtype) / 64 * bbox_size + bbox_min

    return ind.to(dtype)

def get_gt(vol, joints, octree_depth=6):
    sigma = 2 / 2**octree_depth

    dist = torch.cdist(vol, joints)
    # print(dist.min(), dist.max())

    dist = dist.min(dim=1).values
    
    gt = torch.exp(-dist**2 / 2 / sigma**2)
    
    return gt

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)
    
def extract_keypoints(y_pred, vox):

    y_pred = y_pred.detach().cpu().numpy()
    vox = vox.detach().cpu().numpy()
    volume = np.zeros([64, 64, 64])
    volume[...] = -100
    volume[vox[:, 0], vox[:, 1], vox[:, 2]] = y_pred.squeeze(-1)
    
    clusters = []
    cluster_model = DBSCAN(eps=1.8, min_samples=1)

    level = min((0.85 * y_pred.max() + 0.15 * y_pred.min()).item(), 0)
    potential_points = np.argwhere(volume >= level)
    clustering = cluster_model.fit(potential_points)
    for cluster in set(clustering.labels_):
        if cluster == -1:
            print('got noise', len(potential_points[clustering.labels_ == cluster]))
            continue
        clusters.append(potential_points[clustering.labels_ == cluster])

    while True:
        if np.all(np.array([(len(cluster) < 10) for cluster in clusters])):
            break
        new_clusters = []
        for points in clusters:
            if len(points) < 10:
                new_clusters.append(points)
                continue

            value = volume[points[:, 0], points[:, 1], points[:, 2]]

            potential_points = points[value >= (0.1 * value.max() + 0.9 * value.min())]
            clustering = cluster_model.fit(potential_points)
            for cluster in set(clustering.labels_):
                if cluster == -1:
                    print('got noise', len(potential_points[clustering.labels_ == cluster]))
                    continue
                new_clusters.append(potential_points[clustering.labels_ == cluster])

        clusters = new_clusters

    key_points = np.array([cluster.mean(axis=0) for cluster in clusters])
    key_points = key_points / 32 - 1
    
    return key_points