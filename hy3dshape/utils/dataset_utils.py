import numpy as np
import torch
import trimesh
# from ThirdParty.Rignet_utils import binvox_rw


def sparse_to_index(sparse_matrix):
    index = []
    weight = []
    for j in range(len(sparse_matrix)):
        if sparse_matrix[j] > 0:
            index.append(j)
            weight.append(sparse_matrix[j])

    return index, weight

def index_to_sparse(index, weight, shape):
    sparse_matrix = np.zeros([shape[0], shape[1], shape[2]+1])

    row_indices, col_indices = np.meshgrid(np.arange(sparse_matrix.shape[0]), np.arange(sparse_matrix.shape[1]), indexing='ij')

    row_indices = np.expand_dims(row_indices, axis=-1)
    col_indices = np.expand_dims(col_indices, axis=-1)
    
    sparse_matrix[row_indices, col_indices, index] = weight
    

    return torch.from_numpy(sparse_matrix[:, :, :-1])

def index_to_sparse_con(index, shape):
    
    sparse_matrix = np.zeros([shape[0], shape[1], shape[2]+1],dtype=np.int8)
    row_indices, col_indices = np.meshgrid(np.arange(sparse_matrix.shape[0]), np.arange(sparse_matrix.shape[1]), indexing='ij')

    row_indices = np.expand_dims(row_indices, axis=-1)
    col_indices = np.expand_dims(col_indices, axis=-1)
    
    sparse_matrix[row_indices, col_indices, index] = 1
    

    return torch.from_numpy(sparse_matrix[:, :, :-1])

def create_mask(n, max_len=64):
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[:n] = 1
    return mask

def reduce(vox):
    new_data = np.zeros((vox.dims[0] // 2, vox.dims[1] // 2, vox.dims[2] // 2)).astype(bool)
    new_data = np.logical_or(new_data, vox.data[::2, ::2, ::2])
    new_data = np.logical_or(new_data, vox.data[1::2, ::2, ::2])
    new_data = np.logical_or(new_data, vox.data[::2, 1::2, ::2])
    new_data = np.logical_or(new_data, vox.data[::2, ::2, 1::2])
    new_data = np.logical_or(new_data, vox.data[1::2, 1::2, ::2])
    new_data = np.logical_or(new_data, vox.data[1::2, ::2, 1::2])
    new_data = np.logical_or(new_data, vox.data[::2, 1::2, 1::2])
    new_data = np.logical_or(new_data, vox.data[1::2, 1::2, 1::2])
    # dilate the new voxel
    new_data[:-1, :, :] = np.logical_or(new_data[:-1, :, :], new_data[1:, :, :])
    new_data[:, :-1, :] = np.logical_or(new_data[:, :-1, :], new_data[:, 1:, :])
    new_data[:, :, :-1] = np.logical_or(new_data[:, :, :-1], new_data[:, :, 1:])
    return binvox_rw.Voxels(new_data, new_data.shape, vox.translate, vox.scale, vox.axis_order)

def align(vox, y_max):
    new_data = np.zeros(vox.dims).astype(bool)
    ind = np.argwhere(vox.data)
    ind = ind + (np.array(vox.translate) - np.array([-0.5, -0.5 * (1 - y_max), -0.5])) * vox.dims[0]
    # round to the nearest integer
    # ind = np.round(ind).astype(int)
    ind = np.ceil(ind).astype(int)
    # clip to the valid range
    ind = np.clip(ind, 0, vox.dims[0] - 1)
    # new_data[ind[:, 0], ind[:, 1], ind[:, 2]] = True
    return ind

def get_skin_direction(joint_idx, data, parent_index, joints_matrix):
    # Get points influenced by this joint (weight > 0)
    weights = index_to_sparse(data['skins_index'].unsqueeze(0), data['skins_weight'].unsqueeze(0), [1, 8192, data['bones_num']])[0][:,joint_idx]
    mask = weights > 0

    if not torch.any(mask):
        # If no points are influenced, return the opposite direction of its parent 
        parent_idx = parent_index[joint_idx].item()
        if parent_idx == joint_idx:
            return torch.tensor([0, 0, 0.001])
        parent_pos = joints_matrix[parent_idx, :3]
        joint_pos = joints_matrix[joint_idx, :3]
        direction = joint_pos - parent_pos
        norm = torch.norm(direction)
        if norm < 1e-8:  # Add check for zero norm
            return torch.tensor([0, 0, 0.001])
        normalized_direction = direction / norm
        return normalized_direction * 0.01
    
    # Get joint position
    joint_pos = joints_matrix[joint_idx, :3]
    
    # Get weighted average direction from joint to influenced points
    points = data['pc'][mask][:,:3]
    point_weights = weights[mask]
    
    # Calculate directions from joint to each point
    directions = points - joint_pos
    
    # Calculate weighted average direction
    avg_direction = torch.sum(directions * point_weights.unsqueeze(1), dim=0) / torch.sum(point_weights)
    if torch.norm(avg_direction) < 1e-5:
        return torch.tensor([0, 0, 0.001])
    return avg_direction * 1.25

def obj2mesh(obj_path):
    # open the obj as txt
    vertices = []
    faces = []
    with open(obj_path, 'r') as f:
        obj = f.readlines()
        for line in obj:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                faces.append(list(map(int, [i.split('/')[0] for i in line.split()[1:]])))
    vertices = np.array(vertices)
    faces = np.array(faces) - 1
    # print(vertices.shape, faces.shape)

    # create trimesh mesh with given vertices and faces
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    # print(mesh.vertices.shape, mesh.faces.shape)
    return mesh