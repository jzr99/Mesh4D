import sys
sys.path.insert(0, '../hy3dpaint')
import trimesh
import os
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from tqdm import tqdm

def rewrite_obj_with_material(input_path, output_path):
    """
    Opens an OBJ file, skips the first three lines, 
    writes the rest to a new file, and adds:
        mtllib textured_gen_mesh.mtl
        usemtl Material
    to the top of the new file.
    """
    with open(input_path, 'r') as infile:
        lines = infile.readlines()

    # Take lines starting from the 4th line (index 3)
    content_from_4th = lines[3:]

    # Lines to add at the top
    header = [
        "mtllib textured_gen_mesh.mtl\n",
        "usemtl Material\n"
    ]

    # Write to the new file
    with open(output_path, 'w') as outfile:
        outfile.writelines(header + content_from_4th)

    print(f"✅ Wrote modified file to: {output_path}")


# base_dir = '/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture_newtest_batt6_new_40000'
# output_dir = '/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture_newtest_batt6_new_40000_sep'
base_dir = '/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture_newtest_batt6_new_40000_nonalign_infer'
output_dir = '/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture_newtest_batt6_new_40000_nonalign_infer_sep'

for seq_idx in tqdm(os.listdir(base_dir)):
    if not os.path.isdir(os.path.join(base_dir, seq_idx)):
        continue
    for obj_idx in os.listdir(os.path.join(base_dir, seq_idx)):
        # import pdb;pdb.set_trace()
        obj_path = os.path.join(base_dir, seq_idx, obj_idx, 'textured_gen_mesh.obj')
        gen_mesh = trimesh.load(obj_path, process=False)
        for frame_idx in range(6):
            deformed_obj_path = os.path.join(base_dir, seq_idx, obj_idx, 'gen', f'gen_{frame_idx:02d}.obj')
            deformed_gen_mesh = trimesh.load(deformed_obj_path, process=False)
            gen_mesh.vertices = deformed_gen_mesh.vertices
            gen_mesh_mtl_path = obj_path.replace('.obj', f'registered_with_mtl_{frame_idx:02d}.obj')
            gen_mesh.export(gen_mesh_mtl_path)
            gen_mesh_pbr_mtl_path = gen_mesh_mtl_path.replace('.obj', '_with_pbr_mtl.obj')
            rewrite_obj_with_material(gen_mesh_mtl_path, gen_mesh_pbr_mtl_path)
            os.makedirs(os.path.join(output_dir, seq_idx, obj_idx), exist_ok=True)
            convert_obj_to_glb(gen_mesh_pbr_mtl_path, os.path.join(output_dir, seq_idx, obj_idx, f'gen_{frame_idx:02d}.glb'))

        # output_mesh_path = os.path.join(output_folder, batch['model_name'][0], 'textured_gen_mesh.obj')

# mesh = trimesh.load(output_mesh_path)

# gen_mesh_mtl_path = os.path.join(output_path, batch['model_name'][0], 'registered_gen_mesh_with_mtl.obj')
# gen_mesh.vertices = TY
# gen_mesh.export(gen_mesh_mtl_path)
# gen_mesh_pbr_mtl_path = gen_mesh_mtl_path.replace('.obj', '_with_pbr_mtl.obj')
# rewrite_obj_with_material(gen_mesh_mtl_path, gen_mesh_pbr_mtl_path)
# from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
# convert_obj_to_glb(gen_mesh_pbr_mtl_path, gen_mesh_mtl_path.replace('.obj', '.glb'))