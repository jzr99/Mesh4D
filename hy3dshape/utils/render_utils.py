import bpy
import numpy as np
from mathutils import Vector, Matrix
from tqdm import tqdm
import glob
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
cmap = plt.get_cmap('viridis')
import torch
import torchvision.io as io
import cv2
import trimesh

def get_data(ids, root, animate=False, shift_rig=True, id2=None, rignet=False):
    dataset= torch.load('/data2/aod/testJointDataSet_9.pt')
    joints = []
    conns = []
    skins = []

    for id in ids:
        if id2 is None:
            for data in dataset:
                if id in data['name']:
                    print(data['name'])
                    break
        else:
            for data in dataset:
                if id2 in data['name']:
                    print(data['name'])
                    break
        
        joint = torch.tensor(torch.load(root + '/joints/' + id + '.pt')).cpu()
        if shift_rig and id2 is None:
            y_max = data['points_cloud'][:,1].max()
            joint = joint/2 + torch.tensor([0,y_max/2,0])
        temp = joint[:, 1].clone()
        joint[:, 1] = -joint[:, 2]
        joint[:, 2] = temp

        conn = torch.tensor(torch.load(root + '/connectivity/' + id + '.pt')).long()
        if not animate:
            skin = torch.load(root + '/skinning/' + id + '.pt')
            if rignet:
                skins.append(skin[0])
            elif id2 is None:
                skins.append(skin[0].softmax(dim=-1).cpu().numpy())
            else:
                skins.append(skin)

        joints.append(joint)
        conns.append(conn)
    
    return joints, conns, skins

def index_to_sparse(index, weight, shape):
    sparse_matrix = np.zeros([shape[0], shape[1], shape[2]+1])

    row_indices, col_indices = np.meshgrid(np.arange(sparse_matrix.shape[0]), np.arange(sparse_matrix.shape[1]), indexing='ij')

    row_indices = np.expand_dims(row_indices, axis=-1)
    col_indices = np.expand_dims(col_indices, axis=-1)
    
    sparse_matrix[row_indices, col_indices, index] = weight
    

    return torch.from_numpy(sparse_matrix[:, :, :-1])

def get_gt(ids, root):
    dataset= torch.load('/data2/aod/testJointDataSet_9.pt')
    joints = []
    conns = []
    skins = []

    for id in ids:
        for data in dataset:
            if id in data['name']:
                print(data['name'])
                break

        joint = data['joints_matrix'][:data['joints_num'], :3]
        y_max = data['points_cloud'][:,1].max()
        joint = joint/2 + torch.tensor([0,y_max/2,0])
        temp = joint[:, 1].clone()
        joint[:, 1] = -joint[:, 2]
        joint[:, 2] = temp

        conn = data['parent_index'][:data['joints_num']].long().unsqueeze(1)

        skin = index_to_sparse(data['skin_index'].unsqueeze(0), data['skin_weight'].unsqueeze(0), [1, 8192, data['joints_num']])

        joints.append(joint)
        conns.append(conn)
        skins.append(skin[0])
    
    return joints, conns, skins

def empty():
    bpy.ops.wm.read_homefile(use_empty=True)
    # Delete all mesh objects from the scene
    # for obj in bpy.context.scene.objects:
    #     bpy.data.objects.remove(obj, do_unlink=True)

def add_mesh(filepath, co=None, tex=False, color=(0.5, 0.5, 0.5, 1)):
    bpy.ops.wm.obj_import(filepath=filepath)
    obj = bpy.context.object

    if not tex:
        # give the mesh a material
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        mat = bpy.data.materials.new(name='mat')
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.8
        # mat.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
        # mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.5
        mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
    if co is not None:
        obj.parent = co

def create_sphere(location, size=0.01, color=(1.0, 0.0, 0.0, 1.0), reduced=False):
    if reduced:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location, segments=8, ring_count=4)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=location)
    sphere = bpy.context.active_object
    
    material_name = f"ColorMaterial_{color}"
    material = bpy.data.materials.get(material_name)
    
    if not material:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        material.node_tree.nodes.clear()
        bsdf = material.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color
            
    sphere.data.materials.append(material)

    return sphere

def add_co(location=(0,0,0), rotation=(0,0,0), scale=(1,1,1)):
    co = bpy.data.objects.new("CoordinateSystem", None)
    bpy.context.collection.objects.link(co)
    bpy.context.view_layer.objects.active = co
    co.empty_display_size = 0.1
    co.empty_display_type = 'ARROWS'
    co.location = location
    co.rotation_euler = rotation
    co.scale = scale

    return co

def add_joint(joints_matrix, co=None):

    for i, joint in enumerate(joints_matrix):
        sphere = create_sphere((joint[0], joint[1], joint[2]), size=0.01)
        if co is not None:
            sphere.parent = co

def create_blue_cone(base_point, apex_point, radius=0.1):
    # Calculate the radius and length of the cone
    direction = apex_point - base_point
    length = direction.length
    
    # Create cone mesh
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=radius, depth=length, location=(base_point + direction * 0.5))
    cone = bpy.context.active_object
    
    # Create or get the blue material
    blue_material = bpy.data.materials.get("BlueMaterial")
    if not blue_material:
        blue_material = bpy.data.materials.new(name="BlueMaterial")
        blue_material.use_nodes = True
        blue_material.node_tree.nodes.clear()
        bsdf = blue_material.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        output = blue_material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        blue_material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        blue_material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.0, 0.0, 1.0, 1.0)
    
    cone.data.materials.append(blue_material)
    
    # Set the cone's orientation
    cone.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
        
    return cone

def add_conn(con_index, joints_matrix, co=None):
    for i, parent in enumerate(con_index):
        parent = parent.item()
        if parent != i:
            parent_co = Vector((joints_matrix[parent][0], joints_matrix[parent][1], joints_matrix[parent][2]))
            position = Vector((joints_matrix[i][0], joints_matrix[i][1], joints_matrix[i][2]))
            cone = create_blue_cone(parent_co, position, radius=0.008)
            if co is not None:
                cone.parent = co

def merge_images(img1, img2, output_path, alpha=1):
    image_mesh = Image.open(img1)
    image_rig = Image.open(img2)

    if alpha == 1:
        image_mesh.paste(image_rig, (0, 0), image_rig)
        image_mesh.save(output_path)
        return
    
    data = image_rig.getdata()
    data2 = image_mesh.getdata()
    new_data = []
    for item, item2 in zip(data, data2):
        if item[3] == 0:
            new_data.append(item2)
        else:
            new_data.append((int(item[0]*alpha + item2[0]*(1-alpha)), int(item[1]*alpha + item2[1]*(1-alpha)), int(item[2]*alpha + item2[2]*(1-alpha)), 255))
    image_mesh.putdata(new_data)

    # image_mesh.paste(image_rig, (0, 0), image_rig)

    image_mesh.save(output_path)

def merge_videos(video1, video2, output_path):
    
    # overlap two videos together, video1 is the background, video2 is the foreground
    # os.system(f'ffmpeg -i {video1} -i {video2} -filter_complex "[0:v][1:v] overlay=0:0:enable=\'between(t,0,60)\'" -pix_fmt yuv420p -c:a copy {output_path}')

    frames_path_1 = glob.glob(video1 + '*.png')
    total_frames = len(frames_path_1)
    combined_frames = []
    for i in range(total_frames):
        frame1 = Image.open(f'{video1}{i:04d}.png')
        frame2 = Image.open(f'{video2}{i:04d}.png')
        frame1.paste(frame2, (0, 0), frame2)
        combined_frames.append(frame1)

    # paste the combined frames on a pure white background
    combined_frames_white = []
    for frame in combined_frames:
        white = Image.new('RGB', frame.size, (255, 255, 255))
        white.paste(frame, (0, 0), frame)
        combined_frames_white.append(white)
    
    combined_frames=combined_frames_white 

    combined_videos = torch.stack([torch.tensor(np.array(frame)) for frame in combined_frames])[..., :3]
    
    # write the video with high quality
    # io.write_video(output_path, combined_videos, 24)
    io.write_video(output_path, combined_videos, 24, video_codec='libx264', options={'crf': '18'})
    
    # comvert the frames to mp4 video

    # video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), 30, (frame1.size[0], frame1.size[1]))
    # for frame in combined_frames:
    #     video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    # video.release()

    # video_1, audio_1, fps_1 = io.read_video(video1, pts_unit="sec")
    # video_2, audio_2, fps_2 = io.read_video(video2, pts_unit="sec")
    # non_zero = video_2.sum(dim=-1) != 0
    # non_zero = torch.stack([non_zero, non_zero, non_zero], dim=-1)
    # video_1[non_zero] = video_2[non_zero]
    # io.write_video(output_path, video_1, int(fps_1['video_fps']))

def add_skin(filepath, skin, bone_index, co=None, pc=None):
    bpy.ops.wm.obj_import(filepath=filepath)
    obj = bpy.context.object

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    if co is not None:
        obj.parent = co

    if pc is not None:
        skin = np.array(skin)
        pc = pc[:, :3].numpy()
        y_max = pc[:, 1].max()
        pc = pc + np.array([0, y_max, 0])
        pc = pc / 2
        new_skin = np.zeros((len(obj.data.vertices), skin.shape[1]))
        for i, v in enumerate(obj.data.vertices):
            v_co = np.array(v.co)
            
            dist = np.linalg.norm(pc - v_co, axis=1)
            # min_idx = np.argmin(dist)
            # sort, and then get top 3 index
            min_idx_list = np.argsort(dist)[:3]

            for min_idx in min_idx_list:
                # get inverse distance weight
                interpolate_weight = np.square(1 / dist[min_idx]) / np.square(1 / dist[min_idx_list]).sum()
                new_skin[i] = new_skin[i] + interpolate_weight * skin[min_idx]

        skin = new_skin

    color_list = skin

    color_list = color_list[:,bone_index]

    vertex_colors = obj.data.vertex_colors.new()

    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:

            vertex_index = obj.data.loops[loop_index].vertex_index
            # Get the weight for the vertex
            weight = color_list[vertex_index]

            color = cmap(weight)
            
            # Assign the weight to the vertex color (RGBA)
            vertex_colors.data[loop_index].color = color  # Use the weight for RGB

    # let bsdf use vertex color and then output to surface
    mat = bpy.data.materials.new(name='mat')
    # delete all material of obj
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    vertex_color = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(vertex_color.outputs['Color'], bsdf.inputs['Base Color'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.5

    

def add_pc(points):
    base_sphere = create_sphere((points[0][0], points[0][1], points[0][2]), size=0.003, color=cmap(0), reduced=True)
    # copy the base sphere to create the rest of the spheres
    for i in tqdm(range(1, points.shape[0])):
        new_sphere = base_sphere.copy()
        new_sphere.location = (points[i][0], points[i][1], points[i][2])
        bpy.context.collection.objects.link(new_sphere)

def add_floor(back=False):
    # create a plane as floor
    bpy.ops.mesh.primitive_plane_add(size=50, enter_editmode=False, align='WORLD', location=(0, 20, 0))
    floor = bpy.context.object
    floor.name = 'floor'
    # set white material for floor
    mat = bpy.data.materials.new(name='floor_mat')
    floor.data.materials.append(mat)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 1, 1, 1)

    if back:
        # create a plane as background
        bpy.ops.mesh.primitive_plane_add(size=30, enter_editmode=False, align='WORLD', location=(0, 15, 0), rotation=(-0.5*np.pi, 0, 0))
        background = bpy.context.object
        background.name = 'background'
        # set white material for background
        mat = bpy.data.materials.new(name='background_mat')
        background.data.materials.append(mat)
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        output = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        mat.node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = (1, 1, 1, 1)

def setup_render():
    # color management
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # set the render engine to Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    # enable cuda
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'

    # set render background to transparent
    bpy.context.scene.render.film_transparent = True

def render(output_path, shadow=True, shading=True, quick=False):

    if shadow:
        add_floor()
    
    if shading:
        # create a sun light
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(-1, -1, 3))
        light = bpy.context.object
        light.data.energy = 5
        # angle pointing to the origin
        light.rotation_euler = (0.1*np.pi, 0, 0)
        # set angle
        light.data.angle = 0.08*np.pi

    else:
        # global illumination by create world light
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
        world.use_nodes = True
        world_light = world.node_tree.nodes['Background']
        world_light.inputs['Strength'].default_value = 1
        world_light.inputs['Color'].default_value = (1, 1, 1, 1)

    # create a camera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((2, -1.5, 2))
    look_at = Vector((0, 0, 0.36))
    # compute the rotation
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = (camera.location - look_at).to_track_quat('Z', 'Y')
    # set size
    camera.data.sensor_width = 26
    # set the camera to be active
    bpy.context.scene.camera = camera

    

    # make the rendered image square
    bpy.context.scene.render.resolution_x = 2048
    bpy.context.scene.render.resolution_y = 2048

    setup_render()

    if quick:
        # reduce the number of samples
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.preview_samples = 128
        bpy.context.scene.cycles.max_bounces = 1
        bpy.context.scene.cycles.min_bounces = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
    else:
        bpy.context.scene.cycles.samples = 1024
        bpy.context.scene.cycles.preview_samples = 1024
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.min_bounces = 4
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4

    # output path
    # output_path = '/home/ydengbd/objaverse/test.png'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def render_spin(output_path, co, shadow=True, shading=True, quick=False):
    # create a new coordinate system at the origin
    new_co = add_co(location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
    # set the object to be the child of the new coordinate system
    co.parent = new_co

    # add spin animation to the new coordinate system
    new_co.rotation_mode = 'XYZ'
    new_co.rotation_euler = (0, 0, 0)
    new_co.keyframe_insert(data_path='rotation_euler', index=2, frame=0)
    new_co.rotation_euler = (0, 0, 2*np.pi)
    new_co.keyframe_insert(data_path='rotation_euler', index=2, frame=60)
    
    if shadow:
        add_floor()
    
    if shading:
        # create a sun light
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(-1, -1, 3))
        light = bpy.context.object
        light.data.energy = 5
        # angle pointing to the origin
        light.rotation_euler = (0.1*np.pi, 0, 0)
        # set angle
        light.data.angle = 0.08*np.pi

    else:
        # global illumination by create world light
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
        world.use_nodes = True
        world_light = world.node_tree.nodes['Background']
        world_light.inputs['Strength'].default_value = 1
        world_light.inputs['Color'].default_value = (1, 1, 1, 1)

    # create a camera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((2, -1.5, 2))
    look_at = Vector((0, 0, 0.36))
    # compute the rotation
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = (camera.location - look_at).to_track_quat('Z', 'Y')
    # set size
    camera.data.sensor_width = 26
    # set the camera to be active
    bpy.context.scene.camera = camera


    # render the animation
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 60

    # make the rendered image square
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024

    setup_render()

    if quick:
        # reduce the number of samples
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.preview_samples = 128
        bpy.context.scene.cycles.max_bounces = 1
        bpy.context.scene.cycles.min_bounces = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
    else:
        bpy.context.scene.cycles.samples = 512
        bpy.context.scene.cycles.preview_samples = 512
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.min_bounces = 4
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4

    # output path
    bpy.context.scene.render.filepath = output_path
    if output_path.endswith('.mp4'):
        # render a mp4 video
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'

    bpy.ops.render.render(animation=True, write_still=True)

def setup_anim(armature, arti):
    # enter pose mode
    print('Arti shape', arti.shape)
    bpy.ops.object.mode_set(mode='POSE')
    print('total bones', len(armature.pose.bones))
    for i, pose_bone in enumerate(armature.pose.bones):
        pose_bone.rotation_mode = 'XYZ'
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=0)
        
        pose_bone.rotation_euler = arti[i]
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=30)

        pose_bone.rotation_euler = Vector((0, 0, 0))
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=60)
    bpy.ops.object.mode_set(mode='OBJECT')

def render_anim(output_path, armature, arti, quick=False):
    # enter pose mode
    setup_anim(armature, arti)

    # save blend file
    # bpy.ops.wm.save_as_mainfile(filepath='/data2/ydengbd/objaverse/test.blend')

    add_floor()

    # create a sun light
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(-1, -1, 3))
    light = bpy.context.object
    light.data.energy = 5
    # angle pointing to the origin
    light.rotation_euler = (50/180*np.pi, 0, -20/180*np.pi)
    # set angle
    light.data.angle = 12/180*np.pi

    # create a camera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((0, -3, 1.3))
    camera.rotation_euler = Vector((1.309, 0, 0))
    # set size
    camera.data.sensor_width = 36
    # set the camera to be active
    bpy.context.scene.camera = camera

    # render the animation
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 60

    # make the rendered image square
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    setup_render()

    if quick:
        # reduce the number of samples
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.preview_samples = 128
        bpy.context.scene.cycles.max_bounces = 1
        bpy.context.scene.cycles.min_bounces = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
    else:
        bpy.context.scene.cycles.samples = 1024
        bpy.context.scene.cycles.preview_samples = 1024
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.min_bounces = 4
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4

    # output path
    bpy.context.scene.render.filepath = output_path
    if output_path.endswith('.mp4'):
        # render a mp4 video
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'

    bpy.ops.render.render(animation=True, write_still=True)


def render_animspin(output_path, co, armature, arti, shadow=True, shading=True, quick=False):
    # enter pose mode
    print('Arti shape', arti.shape)
    bpy.ops.object.mode_set(mode='POSE')
    print('total bones', len(armature.pose.bones))
    for i, pose_bone in enumerate(armature.pose.bones):
        pose_bone.rotation_mode = 'XYZ'
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=0)
        
        pose_bone.rotation_euler = arti[i]
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=30)

        pose_bone.rotation_euler = Vector((0, 0, 0))
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=60)

        pose_bone.rotation_euler = arti[i]
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=90)
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=150)

        pose_bone.rotation_euler = Vector((0, 0, 0))
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=180)
    bpy.ops.object.mode_set(mode='OBJECT')

    # create a new coordinate system at the origin
    new_co = add_co(location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
    # set the object to be the child of the new coordinate system
    co.parent = new_co

    # add spin animation to the new coordinate system
    new_co.rotation_mode = 'XYZ'
    new_co.rotation_euler = (0, 0, 0)
    new_co.keyframe_insert(data_path='rotation_euler', index=2, frame=90)
    new_co.rotation_euler = (0, 0, 2*np.pi)
    new_co.keyframe_insert(data_path='rotation_euler', index=2, frame=150)

    if shadow:
        add_floor()
    
    if shading:
        # create a sun light
        bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(-1, -1, 3))
        light = bpy.context.object
        light.data.energy = 5
        # angle pointing to the origin
        light.rotation_euler = (0.1*np.pi, 0, 0)
        # set angle
        light.data.angle = 0.08*np.pi

    else:
        # global illumination by create world light
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
        world.use_nodes = True
        world_light = world.node_tree.nodes['Background']
        world_light.inputs['Strength'].default_value = 1
        world_light.inputs['Color'].default_value = (1, 1, 1, 1)

    # create a camera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((2, -1.5, 2))
    look_at = Vector((0, 0, 0.36))
    # compute the rotation
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = (camera.location - look_at).to_track_quat('Z', 'Y')
    # set size
    camera.data.sensor_width = 26
    # set the camera to be active
    bpy.context.scene.camera = camera


    # render the animation
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 180

    # make the rendered image square
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024

    setup_render()

    if quick:
        # reduce the number of samples
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.preview_samples = 128
        bpy.context.scene.cycles.max_bounces = 1
        bpy.context.scene.cycles.min_bounces = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
    else:
        bpy.context.scene.cycles.samples = 512
        bpy.context.scene.cycles.preview_samples = 512
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.min_bounces = 4
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4

    # output path
    bpy.context.scene.render.filepath = output_path
    if output_path.endswith('.mp4'):
        # render a mp4 video
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'

    bpy.ops.render.render(animation=True, write_still=True)

def render_scene(output_path, shadow=True):

    if shadow:
        add_floor()


    # create a sun light
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(-1, -1, 3))
    light = bpy.context.object
    light.data.energy = 5
    # angle pointing to the origin
    light.rotation_euler = (50/180*np.pi, 0, -20/180*np.pi)
    # set angle
    light.data.angle = 12/180*np.pi

    # create a camera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((0, -10, 5))
    camera.rotation_euler = Vector((1.22, 0, 0))
    # set size
    camera.data.sensor_width = 26
    # set the camera to be active
    bpy.context.scene.camera = camera

    

    # make the rendered image square
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    setup_render()

    

    # output path
    # output_path = '/home/ydengbd/objaverse/test.png'
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def render_teaser(output_path, shadow=True, quick=False):
    
    if shadow:
        add_floor(back=True)

    # create a sun light
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(-1, -1, 3))
    light = bpy.context.object
    light.data.energy = 5
    # angle pointing to the origin
    light.rotation_euler = (50/180*np.pi, 0, -20/180*np.pi)
    # set angle
    light.data.angle = 12/180*np.pi

    # create a camera
    cam = bpy.data.cameras.new("Camera")
    cam_ob = bpy.data.objects.new("Camera", cam)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.collection.objects.link(camera)
    camera.location = Vector((0, -3, 1.3))
    camera.rotation_euler = Vector((80/180*np.pi, 0, 0))
    # set size
    camera.data.sensor_width = 48
    # set the camera to be active
    bpy.context.scene.camera = camera

    # render the animation
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 60

    # make the rendered image square
    bpy.context.scene.render.resolution_x = 2400
    bpy.context.scene.render.resolution_y = 1080

    setup_render()

    if quick:
        # reduce the number of samples
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.cycles.preview_samples = 128
        bpy.context.scene.cycles.max_bounces = 1
        bpy.context.scene.cycles.min_bounces = 1
        bpy.context.scene.cycles.diffuse_bounces = 1
        bpy.context.scene.cycles.glossy_bounces = 1
    else:
        bpy.context.scene.cycles.samples = 1024
        bpy.context.scene.cycles.preview_samples = 1024
        bpy.context.scene.cycles.max_bounces = 4
        bpy.context.scene.cycles.min_bounces = 4
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4

    # output path
    bpy.context.scene.render.filepath = output_path
    if output_path.endswith('.mp4'):
        # render a mp4 video
        bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        bpy.context.scene.render.ffmpeg.codec = 'H264'

    bpy.ops.render.render(animation=True, write_still=True)

def setup_armature(path, tex=False, save=True):
    joints_matrix = torch.load(os.path.join(path, 'joints.pt'))
    connectivity = torch.load(os.path.join(path, 'conns.pt'))
    skinning_weights = torch.load(os.path.join(path, 'skins.pt'))
    obj_file_path = os.path.join(path, 'object.obj')

    # bpy.ops.wm.obj_import(filepath=obj_file_path)
    add_mesh(obj_file_path, tex=tex)
    mesh_object = bpy.context.selected_objects[0]
    
    # pack textures
    bpy.ops.file.pack_all() 

    temp = torch.tensor(joints_matrix)[:, 1].clone()
    joints_matrix[:, 1] = -joints_matrix[:, 2]
    joints_matrix[:, 2] = temp

    bpy.ops.object.armature_add()
    armature_obj = bpy.context.object


    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    world_matrix = Matrix([[1, 0, 0, 0],  
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
    armature_obj.matrix_world = world_matrix

    bone_dict = {}

    i_name = 0

    for i in range(len(joints_matrix)):

        if connectivity[i] == i:
            continue
        bone_name = str(i_name)
        bone = armature_obj.data.edit_bones.new(bone_name)
        bone.head = joints_matrix[connectivity[i]].cpu().numpy()
        bone.tail = joints_matrix[i].cpu().numpy()
        bone_dict[bone_name] = bone
        i_name += 1
    
    for bone_name, bone in bone_dict.items():
        # Find parent bone by checking if current bone's head matches any other bone's tail
        for other_bone_name, other_bone in bone_dict.items():
            if other_bone != bone and bone.head == other_bone.tail:
                bone.parent = other_bone
                break

    assert i_name == skinning_weights.shape[1]
    
    for i, skinning_weight in enumerate(skinning_weights):
        # print("skinning_weight", skinning_weight)
        vertex_index = i
        for j,weight in enumerate(skinning_weight):
            bone_name = str(j)
            bone_weight = float(weight)

            vertex_group_name = f"{bone_name}"
            vertex_group = mesh_object.vertex_groups.get(vertex_group_name)
            if vertex_group is None:
                vertex_group = mesh_object.vertex_groups.new(name=vertex_group_name)
            vertex_group.add([vertex_index], bone_weight, 'ADD')

    # for obj in bpy.context.scene.objects:
    #     if obj.type == 'MESH':
    modifier = mesh_object.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature_obj
    modifier.use_vertex_groups = True
    print("Armature modifier added to mesh:", mesh_object.name)

    bpy.ops.object.mode_set(mode='OBJECT')
    if save:
        bpy.ops.wm.save_as_mainfile(filepath= os.path.join(path, 'blender_output.blend'))

    return armature_obj

def reload_tensor_skinning(data, bone_name_list):
    
    # with open(json_file, "r") as f:
    #     skinning_data = json.load(f)

    armature_obj = bpy.data.objects.get("Armature")
    if not armature_obj:
        print("Error: Armature object 'Armature' not found.")
        return
    
    count = 0
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.parent = armature_obj
            count += 1
            
    print("total mesh count:", count)

    for obj in bpy.context.scene.objects:
        vertex_index = 0
        if obj.type == 'MESH':
            # mesh_name = obj.name
            # if mesh_name in skinning_data:
            #     skinning_info = skinning_data[mesh_name]
            #     if "weight" in skinning_info:
            #         print("Applying skinning data for mesh:", mesh_name)
            #         vertex_index = 0
            #         for vertex_weight in skinning_info["weight"]:
            #             for bone_name, weight_value in vertex_weight.items():
            #                 vertex_group = obj.vertex_groups.get(bone_name)
            #                 if vertex_group is None:
            #                     vertex_group = obj.vertex_groups.new(name=bone_name)
            #                     print("Vertex group created:", bone_name)
            #                 vertex_group.add([vertex_index], weight_value, 'REPLACE')
            #             vertex_index += 1
            # else:
            #     print("No skinning data found for mesh:", mesh_name)

            for i, v in enumerate(obj.data.vertices):
                v_co = np.array(v.co)
                pc = data['pc'][:, :3].numpy()
                y_max = pc[:, 1].max()
                pc = pc + np.array([0, y_max, 0])
                pc = pc / 2
                dist = np.linalg.norm(pc - v_co, axis=1)
                # min_idx = np.argmin(dist)
                # sort, and then get top 3 index
                min_idx_list = np.argsort(dist)[:3]

                for min_idx in min_idx_list:
                    # get inverse distance weight
                    interpolate_weight = np.square(1 / dist[min_idx]) / np.square(1 / dist[min_idx_list]).sum()

                    for idx, j in enumerate(data['skins_index'][min_idx]):
                        if j == -1:
                            break
                        bone_name = bone_name_list[j]
                        vertex_group = obj.vertex_groups.get(str(int(bone_name)))
                        if vertex_group is None:
                            vertex_group = obj.vertex_groups.new(name=str(int(bone_name)))
                            print("Vertex group created:", bone_name)

                        vertex_group.add([i], interpolate_weight * data['skins_weight'][min_idx][idx], 'ADD')

                
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
            modifier.object = armature_obj
            modifier.use_vertex_groups = True
            print("Armature modifier added to mesh:", obj.name)   

def reload_tensor(data, root='data', save=True):
    joints_matrix = data['joints'].clone()
    connectivity = data['conns']
    obj_file_path = os.path.join(root, data['name'], 'object.obj')

    # bpy.ops.wm.obj_import(filepath=obj_file_path)
    add_mesh(obj_file_path)
    mesh_object = bpy.context.selected_objects[0]
    
    # pack textures
    bpy.ops.file.pack_all() 

    y_max = data['pc'][:, 1].max()
    joints_matrix = joints_matrix + torch.tensor([0, y_max, 0])
    joints_matrix = joints_matrix / 2

    temp = joints_matrix[:, 1].clone()
    joints_matrix[:, 1] = -joints_matrix[:, 2]
    joints_matrix[:, 2] = temp

    bpy.ops.object.armature_add()
    armature_obj = bpy.context.object


    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    
    world_matrix = Matrix([[1, 0, 0, 0],  
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])
    armature_obj.matrix_world = world_matrix

    bone_dict = {}
    bone_name_list = np.zeros(data['bones_num'])
    i_name = 0

    for i in range(len(joints_matrix)):

        if connectivity[i] == i:
            continue
        bone_name = str(i_name)
        bone = armature_obj.data.edit_bones.new(bone_name)
        bone.head = joints_matrix[connectivity[i]].cpu().numpy()
        bone.tail = joints_matrix[i].cpu().numpy()
        bone_dict[bone_name] = bone
        for j, skinbone in enumerate(data['bones']):
            if torch.equal(skinbone[:3], data['joints'][connectivity[i]]) and torch.equal(skinbone[3:], data['joints'][i]):
                bone_name_list[j] = i_name
        i_name += 1
    
    for bone_name, bone in bone_dict.items():
        # Find parent bone by checking if current bone's head matches any other bone's tail
        for other_bone_name, other_bone in bone_dict.items():
            if other_bone != bone and bone.head == other_bone.tail:
                bone.parent = other_bone
                break

    print(bone_name_list)

    reload_tensor_skinning(data, bone_name_list)

    print("Armature modifier added to mesh:", mesh_object.name)

    bpy.ops.object.mode_set(mode='OBJECT')
    if save:
        bpy.ops.wm.save_as_mainfile(filepath= os.path.join('/data2/ydengbd/Anymate/Anymate/data', data['name'], 'blender_output.blend'))

    return armature_obj

def load_blender(blender_path):
    
    bpy.ops.wm.read_homefile(use_empty=True)
    # bpy.ops.wm.append(directory=object_path, link=False)
    # load_object(object_path)
    bpy.ops.wm.open_mainfile(filepath=blender_path)
    armature_obj = []
    mesh_obj = []
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            armature_obj.append(obj)
        if obj.type == "MESH":
            mesh_obj.append(obj)
        
    print('mesh obj:', len(mesh_obj))
            
    
    
    # start retrieve the information of mesh, skining and rigging
    
    #1. retrieve the information of rigging, save the world matrix of the amature object
    total_armature_info = {}
    joints_matrix = []
    bone_dict = {}
    parent_name= []
    bone_count = 0
    for obj in armature_obj:
        # depsgraph = bpy.context.evaluated_depsgraph_get()
        # obj = obj.evaluated_get(depsgraph)
        armature_info = {}
        armature_info["world_matrix"] = [list(row) for row in obj.matrix_world.copy()]
        translation = obj.matrix_world.translation
        for bone in obj.pose.bones:
   
            joints_matrix.append(np.array(list((obj.matrix_world.to_3x3() @ bone.head+translation).copy())))

            if bone.parent:
                parent_name.append(bone.parent.name)
            else:
                parent_name.append('root')
            bone_dict[bone.name] = bone_count
            bone_count += 1
    connectivity = torch.zeros(bone_count, dtype=torch.int32)
    
    for i, bone_name in enumerate(parent_name):
        if bone_name == 'root':
            connectivity[i] = i
        else:
            connectivity[i] = bone_dict[bone_name]
    joints_matrix = torch.from_numpy(np.array(joints_matrix))
        
    skinning_weight = torch.zeros(len(mesh_obj[0].data.vertices), joints_matrix.shape[0])
    
    vertex_index = 0
    for obj in mesh_obj:
        vertex_groups = obj.vertex_groups

        
        for vertex in obj.data.vertices:
            vertex_info = {}
            for group in vertex.groups:
                name = vertex_groups[group.group].name

                weight = group.weight
                skinning_weight[vertex.index][bone_dict[name]] = weight

    obj_save_path = blender_path.replace('.blend', '.obj')
    bpy.ops.wm.obj_export(filepath=obj_save_path, export_materials=False)
    return joints_matrix,connectivity, skinning_weight


def save_scene(scene_path):
    # export the scene as a glb file
    if scene_path.endswith('.glb'):
        bpy.ops.export_scene.gltf(filepath=scene_path)
        bpy.ops.wm.save_as_mainfile(filepath=scene_path.replace('.glb', '.blend'))
    elif scene_path.endswith('.blend'):
        bpy.ops.wm.save_as_mainfile(filepath=scene_path)
    elif scene_path.endswith('.obj'):
        bpy.ops.wm.obj_export(filepath=scene_path, export_materials=False)
    else:
        raise ValueError(f"Unsupported file extension: {scene_path}")

if __name__ == '__main__':
    # load the mesh
    empty()
    add_mesh('/home/ydengbd/objaverse/obj/0001.obj')
    # load the joints
    joints_matrix = np.load('/home/ydengbd/objaverse/joints/0001.npy')
    add_joint(joints_matrix)
    # load the connections
    con_index = np.load('/home/ydengbd/objaverse/connections/0001.npy')
    add_conn(con_index)
    # load the skin