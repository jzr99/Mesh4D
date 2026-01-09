# combine_glb_anim.py - Creates an animatable GLB where OBJ files are animation keyframes
import bpy
import os
import sys


def combine_glb_objs(input_files, output_glb, fps=24, frame_duration=10, transition_frames=5):
    """
    Combines one GLB file with multiple OBJ files into an animatable GLB.
    Each OBJ file becomes a keyframe in the animation with smooth transitions.
    
    Args:
        input_files: List of file paths, first should be GLB (base), rest are OBJ files (keyframes)
        output_glb: Output GLB file path
        fps: Frames per second for the animation (default: 24)
        frame_duration: Number of frames each keyframe holds at full weight (default: 10)
        transition_frames: Number of frames for transition between keyframes (default: 5)
    """
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

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Import base GLB (first file)
    base_file = input_files[0]
    obj_files = input_files[1:]
    
    if not obj_files:
        raise ValueError("Need at least one OBJ file to create animation keyframes.")
    
    import_file(base_file)
    base_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    
    if not base_objects:
        raise RuntimeError("No mesh objects imported from base GLB file.")
    
    base = base_objects[0]
    bpy.ops.object.select_all(action='DESELECT')
    base.select_set(True)
    bpy.context.view_layer.objects.active = base

    # Import OBJ files and create shape keys
    imported_obj_objects = []
    for obj_path in obj_files:
        import_file(obj_path)
        # Collect imported mesh objects
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                imported_obj_objects.append(obj)
    
    if not imported_obj_objects:
        raise RuntimeError("No mesh objects imported from OBJ files.")
    
    # Create shape keys from OBJ files
    shape_key_names = []
    for i, obj in enumerate(imported_obj_objects):
        # Select base and target
        bpy.ops.object.select_all(action='DESELECT')
        base.select_set(True)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = base
        
        # Join as shape key
        bpy.ops.object.join_shapes()
        
        # Get the shape key name (last created)
        if base.data.shape_keys:
            sk = base.data.shape_keys.key_blocks[-1]
            shape_key_names.append(sk.name)
    
    # Remove the now-unneeded OBJ objects
    for obj in imported_obj_objects:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    # Create animation action
    if not base.data.shape_keys:
        raise RuntimeError("Failed to create shape keys.")
    
    # Ensure base has an animation data block
    if not base.animation_data:
        base.animation_data_create()
    
    # Create or get action
    action_name = f"{base.name}_Animation"
    if action_name in bpy.data.actions:
        action = bpy.data.actions[action_name]
    else:
        action = bpy.data.actions.new(name=action_name)
    
    base.animation_data.action = action
    
    # Set up shape key animation data paths
    shape_keys = base.data.shape_keys
    shape_keys.animation_data_create()
    
    if not shape_keys.animation_data.action:
        shape_keys.animation_data.action = bpy.data.actions.new(name=f"{base.name}_ShapeKeys")
    
    sk_action = shape_keys.animation_data.action
    
    # Set FPS for timeline
    scene = bpy.context.scene
    scene.render.fps = fps
    
    # Create keyframes with smooth transitions between them
    # Each OBJ becomes a keyframe with smooth fade in/out transitions
    # Start at frame 1 if frame_duration is 1 (to reserve frame 0 for base mesh)
    current_frame = 1 if (frame_duration == 1 or frame_duration == 0) else 0
    
    # Initialize all shape key fcurves
    fcurves = {}
    for sk_name in shape_key_names:
        if sk_name in shape_keys.key_blocks:
            fcurve = None
            for fc in sk_action.fcurves:
                if fc.data_path == f'key_blocks["{sk_name}"].value':
                    fcurve = fc
                    break
            if fcurve is None:
                fcurve = sk_action.fcurves.new(data_path=f'key_blocks["{sk_name}"].value')
            fcurves[sk_name] = fcurve
    
    # Set initial frame (all shape keys at 0)
    for sk_name in shape_key_names:
        if sk_name in fcurves:
            fcurves[sk_name].keyframe_points.insert(0, 0.0)
            fcurves[sk_name].keyframe_points[-1].interpolation = 'LINEAR'
    
    # Create smooth keyframe animations
    for i, sk_name in enumerate(shape_key_names):
        if sk_name not in fcurves:
            continue
        
        # Previous shape key (to fade out)
        prev_sk_name = shape_key_names[i - 1] if i > 0 else None
        
        # Transition in: fade from 0 to 1 over transition_frames
        transition_start = current_frame
        if transition_frames > 0:
            # Start of transition: current shape key at 0
            fcurves[sk_name].keyframe_points.insert(transition_start, 0.0)
            fcurves[sk_name].keyframe_points[-1].interpolation = 'LINEAR'
            
            # End of transition: current shape key at 1.0
            transition_end = transition_start + transition_frames
            fcurves[sk_name].keyframe_points.insert(transition_end, 1.0)
            fcurves[sk_name].keyframe_points[-1].interpolation = 'LINEAR'
            
            # If there's a previous shape key, fade it out during transition
            if prev_sk_name and prev_sk_name in fcurves:
                fcurves[prev_sk_name].keyframe_points.insert(transition_start, 1.0)
                fcurves[prev_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
                fcurves[prev_sk_name].keyframe_points.insert(transition_end, 0.0)
                fcurves[prev_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
            
            # Ensure all other shape keys stay at 0 during transition
            for other_sk_name in shape_key_names:
                if other_sk_name != sk_name and other_sk_name != prev_sk_name and other_sk_name in fcurves:
                    # Ensure it's at 0 at transition start
                    fcurves[other_sk_name].keyframe_points.insert(transition_start, 0.0)
                    fcurves[other_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
                    fcurves[other_sk_name].keyframe_points.insert(transition_end, 0.0)
                    fcurves[other_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
        else:
            # No transition: instant switch
            if prev_sk_name and prev_sk_name in fcurves:
                fcurves[prev_sk_name].keyframe_points.insert(transition_start, 0.0)
                fcurves[prev_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
            fcurves[sk_name].keyframe_points.insert(transition_start, 1.0)
            fcurves[sk_name].keyframe_points[-1].interpolation = 'LINEAR'
            transition_end = transition_start
            
            # Ensure all other shape keys stay at 0
            for other_sk_name in shape_key_names:
                if other_sk_name != sk_name and other_sk_name != prev_sk_name and other_sk_name in fcurves:
                    fcurves[other_sk_name].keyframe_points.insert(transition_start, 0.0)
                    fcurves[other_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
        
        # Hold at 1.0 for the duration
        hold_end = transition_end + frame_duration
        fcurves[sk_name].keyframe_points.insert(hold_end, 1.0)
        fcurves[sk_name].keyframe_points[-1].interpolation = 'LINEAR'
        
        # Ensure all other shape keys stay at 0 during hold period
        for other_sk_name in shape_key_names:
            if other_sk_name != sk_name and other_sk_name in fcurves:
                fcurves[other_sk_name].keyframe_points.insert(hold_end, 0.0)
                fcurves[other_sk_name].keyframe_points[-1].interpolation = 'LINEAR'
        
        # Move to next keyframe position (transition start for next)
        current_frame = hold_end
    
    # Update frame range
    scene.frame_start = 0
    scene.frame_end = current_frame - 1
    
    # Update fcurves
    for fcurve in sk_action.fcurves:
        fcurve.update()
    
    # Export to GLB with animation
    bpy.ops.object.select_all(action='DESELECT')
    base.select_set(True)
    bpy.context.view_layer.objects.active = base

    # Export with animation
    try:
        bpy.ops.export_scene.gltf(filepath=output_glb,
                                export_format='GLB',
                                use_selection=True,
                                export_colors=True,
                                export_materials='EXPORT',
                                export_apply=False,
                                export_animations=True,  # Export animations
                                export_current_frame=False)  # Export animation, not just current frame
    except TypeError:
        # Fall back to older API
        bpy.ops.export_scene.gltf(filepath=output_glb,
                                export_format='GLB',
                                export_selected=True,
                                export_colors=True,
                                export_materials='EXPORT',
                                export_apply=False)

    print(f"Exported animatable GLB with {len(shape_key_names)} keyframes: {output_glb}")


if __name__ == "__main__":
    # input_files = [
    #     "/path/to/base_mesh.glb",      # first file should be the base (basis) shape
    #     "/path/to/keyframe1.obj",      # OBJ files will be animation keyframes
    #     "/path/to/keyframe2.obj",
    #     # add more OBJ files...
    # ]
    input_files = ['/scratch/shared/beegfs/zeren/Hunyuan3D-2.1/hy3dshape/output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/registered_gen_mesh_with_mtl.glb', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_01.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_02.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_03.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_04.obj', 'output_folder/test/overfitting_depth_16_token_4096_lr1e4_motion_2k_pretrain_170000_predict_85_registered_nongtinit_bone_test_texture/000-156/053ae369dda94d3c98bb32d195b468d0_0_1/gen/gen_05.obj']
    output_glb = "./test_animated_smooth_d0_t2_new.glb"
    fps = 24  # Frames per second
    frame_duration = 0  # Number of frames each keyframe holds at full weight
    transition_frames = 2  # Number of frames for smooth transition between keyframes
    combine_glb_objs(input_files, output_glb, fps=fps, frame_duration=frame_duration, transition_frames=transition_frames)
