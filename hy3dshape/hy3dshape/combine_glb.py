# join_as_shapes_and_export.py
import bpy
import os
import sys


def combine_glb_objs(input_files, output_glb):
    # --- User settings: change these ---
    # input_files = [
    #     "/path/to/base_mesh.glb",      # first file should be the base (basis) shape
    #     "/path/to/variant1.glb",
    #     "/path/to/variant2.glb",
    #     # add more files...
    # ]
    # output_glb = "/path/to/output_with_morphs.glb"
    # -------------------------------

    # In Blender 4.0+, OBJ importer is built-in and uses wm.obj_import operator
    # For older versions, it uses import_scene.obj operator

    def import_file(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext in (".obj",):
            # Try Blender 4.0+ operator first, fall back to older version
            try:
                bpy.ops.wm.obj_import(filepath=path)
            except AttributeError:
                # Fall back to older import_scene.obj operator
                try:
                    bpy.ops.import_scene.obj(filepath=path)
                except AttributeError:
                    raise RuntimeError("OBJ importer not available. Please enable the OBJ importer addon in Blender preferences.")
        else:
            raise ValueError(f"Unsupported file type: {path}")

    # def main():
    # clear scene (optional)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    imported_objects = []
    for p in input_files:
        import_file(p)
        # assume importer adds new objects and makes them selected; collect selected mesh objects
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                imported_objects.append(obj)

    if not imported_objects:
        raise RuntimeError("No mesh objects imported.")

    # Choose the first imported mesh as base (active)
    base = imported_objects[0]
    # Ensure base is active and selected
    bpy.ops.object.select_all(action='DESELECT')
    base.select_set(True)
    bpy.context.view_layer.objects.active = base

    # Select the other objects (targets) - order matters if you care about shape key names
    for obj in imported_objects[1:]:
        obj.select_set(True)

    # Join as shapes: this creates shape keys on the active object from other selected objects
    bpy.ops.object.join_shapes()  # operator: joins selected objects as shape keys of active

    # Optionally remove the now-unneeded target objects
    for obj in imported_objects[1:]:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    # (Optional) rename shape keys nicely
    if base.data.shape_keys:
        for i, key_block in enumerate(base.data.shape_keys.key_blocks):
            # key_block.name: key_block[0] is "Basis" typically
            pass

    # Export to GLB
    bpy.ops.object.select_all(action='DESELECT')
    base.select_set(True)
    bpy.context.view_layer.objects.active = base

    # Try Blender 4.0+ API first (use_selection instead of export_selected)
    try:
        bpy.ops.export_scene.gltf(filepath=output_glb,
                                export_format='GLB',
                                use_selection=True,
                                export_colors=True,
                                export_materials='EXPORT',
                                export_apply=False)  # do not apply modifiers (change as you need)
    except TypeError:
        # Fall back to older API with export_selected
        bpy.ops.export_scene.gltf(filepath=output_glb,
                                export_format='GLB',
                                export_selected=True,
                                export_colors=True,
                                export_materials='EXPORT',
                                export_apply=False)

    print("Exported:", output_glb)


if __name__ == "__main__":
    # input_files = [
    #     "/path/to/base_mesh.glb",      # first file should be the base (basis) shape
    #     "/path/to/variant1.glb",
    #     "/path/to/variant2.glb",
    #     # add more files...
    # ]
    input_files = ['/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/registered_gen_mesh_with_mtl.glb', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_01.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_02.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_03.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_04.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_05.obj']
    output_glb = "./test.glb"
    combine_glb_objs(input_files, output_glb)
