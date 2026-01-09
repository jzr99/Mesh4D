import trimesh
import numpy as np
import torch

from Anymate.utils.utils import load_checkpoint, get_joints, get_connectivity
from Anymate.args import anymate_args
from Anymate.utils.render_utils import empty, add_co, add_mesh, add_joints, add_conn, add_skin, setup_armature, save_scene

def visualize_results(mesh_file=None, joints=None, connectivity=None, skinning=None):

    import bpy
    # Create a scene with both original and processed meshes
    vis_file = "Anymate/tmp/vis_scene.glb"
    print('fffffffff')

    # empty()
    bpy.ops.wm.read_homefile(use_empty=True)
    
    if mesh_file is not None:
        # add_mesh(mesh_file)
        bpy.ops.wm.obj_import(filepath=mesh_file)

    if joints is not None:
        add_joints(joints)

        if connectivity is not None:
            add_conn(connectivity, joints)
    
    if skinning is not None:
        add_skin(mesh_file, skinning)
        
    # setup_armature()
    # save_scene(vis_file)
    bpy.ops.wm.save_as_mainfile(filepath=vis_file)
    return vis_file


def process_mesh_to_pc(obj_path, sample_num = 8192, save_path = None):
    # mesh_list : list of trimesh
    try :
        mesh = trimesh.load_mesh(obj_path)

        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]

        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)


        if save_path is not None:
            np.save(save_path, pc_normal)
        
        return pc_normal
    except Exception as e:
        print(f"Error: {obj_path} {e}")
        return None
    

def normalize_mesh(mesh):
    # Get vertices and compute bounding box
    vertices = mesh.vertices
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Find center and scale
    center = (bbox_min + bbox_max) * 0.5
    scale = 2.0 / (bbox_max - bbox_min).max()
    
    # Center and scale vertices
    vertices = (vertices - center) * scale
    
    # Create new mesh with normalized vertices
    normalized_mesh = trimesh.Trimesh(vertices=vertices, 
                                    faces=mesh.faces,
                                    face_normals=mesh.face_normals,
                                    vertex_normals=mesh.vertex_normals)
    
    return normalized_mesh


def vis_joint(normalized_mesh_file, joints):
    vis_file = visualize_results(mesh_file=normalized_mesh_file, joints=joints)
    return vis_file

def vis_connectivity(normalized_mesh_file, joints, connectivity):
    vis_file = visualize_results(mesh_file=normalized_mesh_file, joints=joints, connectivity=connectivity)
    return vis_file

def vis_skinning(skinning):
    vis_file = visualize_results(skinning=skinning)
    return vis_file


def process_input(mesh_file):
    """
    Function to handle input changes and initialize visualization
    
    Args:
        mesh_file: Path to input mesh file
        joint_checkpoint: Path to joint prediction checkpoint
        conn_checkpoint: Path to connectivity prediction checkpoint 
        skin_checkpoint: Path to skinning prediction checkpoint
    
    Returns:
        vis_file: Path to visualization file
    """

    # For now just visualize the input mesh
    
    normalized_mesh = normalize_mesh(trimesh.load(mesh_file))
    normalized_mesh_file = "Anymate/tmp/normalized_mesh.obj"
    normalized_mesh.export(normalized_mesh_file)
    vis_file = visualize_results(mesh_file=normalized_mesh_file)
    pc = process_mesh_to_pc(normalized_mesh_file)
    pc = torch.from_numpy(pc).to(anymate_args.device).to(torch.float32)

    print(pc.shape, pc.max(dim=0), pc.min(dim=0))

    return normalized_mesh_file, vis_file, pc, None, None, None


def get_model(checkpoint):
    model = load_checkpoint(checkpoint, anymate_args.device, anymate_args.num_joints)
    return model, True

def get_result_joint(model, pc):
    return get_joints(pc, model, anymate_args.device)

def get_result_connectivity(model, pc, joints):
    return get_connectivity(pc, joints, model, anymate_args.device)

def get_result_skinning(model, pc):
    with torch.no_grad():
        skinning = model(pc)
        return skinning