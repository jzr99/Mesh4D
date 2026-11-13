#!/usr/bin/env python3
"""Render an animated GLB or static OBJ with Blender, without loading Mesh4D models.

The scene, camera and animation workflow follows the Hunyuan3D-2.1 helpers in
hy3dshape/tools/render/render.py and hy3dshape/utils/render_utils.py. This entry
point adds animation-aware bounds, explicit frame selection and portable paths.
"""
import argparse
import json
import math
import sys
from pathlib import Path

VIEWS = ('front', 'right', 'back', 'left')
POSITIONS = {'front': (0, -2, 0), 'right': (2, 0, 0),
             'back': (0, 2, 0), 'left': (-2, 0, 0)}


def frame_numbers(start, end, step):
    if start < 0 or end < start or step < 1:
        raise ValueError('Frames require 0 <= start <= end and a positive step.')
    return list(range(start, end + 1, step))


def read_cameras(directory, views):
    cameras = {}
    for view in views:
        data = json.loads((directory / f'{view}.json').read_text())
        matrix = data['matrix_world']
        if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                or not all(math.isfinite(x) for row in matrix for x in row)):
            raise ValueError(f'{view}.json needs a finite 4x4 matrix_world.')
        x_fov, y_fov = data['x_fov'], data['y_fov']
        if not (0 < x_fov < math.pi and 0 < y_fov < math.pi):
            raise ValueError(f'{view}.json has an invalid field of view (radians).')
        if not math.isclose(x_fov, y_fov, rel_tol=1e-5):
            raise ValueError('This renderer requires square cameras with equal x_fov/y_fov.')
        cameras[view] = data
    return cameras


def scene_bounds(bpy, frames):
    """Evaluate deformations and transforms before measuring each requested frame."""
    from mathutils import Vector
    lower, upper = [math.inf] * 3, [-math.inf] * 3
    scene = bpy.context.scene
    for frame in frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in scene.objects:
            if obj.type != 'MESH' or obj.hide_render or not obj.data.vertices:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            for corner in evaluated.bound_box:
                point = evaluated.matrix_world @ Vector(corner)
                for axis in range(3):
                    lower[axis] = min(lower[axis], point[axis])
                    upper[axis] = max(upper[axis], point[axis])
    if not all(math.isfinite(x) for x in lower + upper):
        raise ValueError('The input has no visible mesh geometry.')
    return lower, upper


def normalize_scene(bpy, frames, mode):
    from mathutils import Vector
    if mode == 'none':
        return {'mode': mode, 'scale': 1.0, 'offset': [0.0, 0.0, 0.0]}
    lower, upper = scene_bounds(bpy, frames if mode == 'sequence' else frames[:1])
    extent = max(hi - lo for lo, hi in zip(lower, upper))
    if extent <= 0:
        raise ValueError('Cannot normalize a zero-size mesh.')
    center = (Vector(lower) + Vector(upper)) / 2
    # An extra, unanimated parent preserves animated root translations and scales.
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    parent = bpy.data.objects.new('Mesh4DNormalization', None)
    bpy.context.scene.collection.objects.link(parent)
    for obj in roots:
        obj.parent = parent
    parent.scale = (1 / extent,) * 3
    parent.location = -center / extent
    bpy.context.view_layer.update()
    return {'mode': mode, 'scale': 1 / extent, 'offset': list(parent.location),
            'source_bounds': [lower, upper]}


def mesh4d_coordinates(bpy):
    """Match the research prediction renderer: canonical bounds, then -90 degrees about X."""
    from mathutils import Matrix
    normalization = normalize_scene(bpy, [0], 'first-frame')
    rotation = Matrix.Rotation(-math.pi / 2, 4, 'X')
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = rotation @ obj.matrix_world
    bpy.context.view_layer.update()
    normalization.update(mode='mesh4d-first-pose', reference_frame=0,
                         rotation=[list(row) for row in rotation])
    return normalization


def setup_render(bpy, args):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.cycles.device = 'CPU'
    if args.device != 'CPU':
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = args.device
        preferences.get_devices()
        devices = [device for device in preferences.devices if device.type == args.device]
        if not devices:
            raise RuntimeError(f'No {args.device} device found; use --device CPU.')
        for device in preferences.devices:
            device.use = device in devices
        scene.cycles.device = 'GPU'
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new('Mesh4DWorld')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.5
    scene.world = world
    from mathutils import Vector
    for name, location, energy in (
            ('Key', (4, -4, 6), 500), ('Fill', (-4, -2, 3), 250), ('Rim', (0, 4, 4), 350)):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.size = energy, 5
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = location
        light.rotation_euler = (-Vector(location)).to_track_quat('-Z', 'Y').to_euler()
    if args.shading == 'clay':
        material = bpy.data.materials.new('Mesh4DClay')
        material.use_nodes = True
        shader = material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (0.55, 0.65, 0.8, 1)
        shader.inputs['Roughness'].default_value = 0.7
        for obj in scene.objects:
            if obj.type == 'MESH':
                obj.data.materials.clear()
                obj.data.materials.append(material)


def render(args):
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError('Run with Blender 4.0 or a Python environment containing bpy.') from exc
    from mathutils import Matrix
    cameras = read_cameras(args.camera_dir, args.views) if args.camera_dir else {}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # glTF stores seconds. Set FPS before import so its keyframes use this timebase.
    scene.render.fps = args.fps
    if args.input.suffix.lower() in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    else:
        bpy.ops.wm.obj_import(filepath=str(args.input.resolve()))
    for obj in list(scene.objects):
        if obj.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(obj, do_unlink=True)
    for name in args.exclude_object:
        obj = scene.objects.get(name)
        if obj is None or obj.type != 'MESH':
            raise ValueError(f'Excluded mesh object not found: {name!r}')
        obj.hide_render = True
    if not any(obj.type == 'MESH' and not obj.hide_render and obj.data.vertices for obj in scene.objects):
        raise ValueError('No visible meshes remain after import and object exclusions.')
    ranges = [action.frame_range for action in bpy.data.actions]
    detected_start = max(0, math.floor(min(r[0] for r in ranges))) if ranges else 0
    detected_end = math.ceil(max(r[1] for r in ranges)) if ranges else 0
    start = args.frame_start if args.frame_start is not None else detected_start
    end = args.frame_end if args.frame_end is not None else detected_end
    frames = frame_numbers(start, end, args.frame_step)
    views = ['turntable'] if args.turntable else args.views
    outputs = [args.output_dir / f'{view}_frame{frame:04d}.png'
               for frame in frames for view in views]
    if not args.overwrite and any(path.exists() for path in outputs + [args.output_dir / 'render.json']):
        raise FileExistsError('Render outputs already exist; use another directory or --overwrite.')
    normalization = (mesh4d_coordinates(bpy) if args.mesh4d_coordinates
                     else normalize_scene(bpy, frames, args.normalize))
    setup_render(bpy, args)
    camera = bpy.data.objects.new('Mesh4DCamera', bpy.data.cameras.new('Mesh4DCamera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = 'PERSP'
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = camera.data.sensor_height = 36
    camera.data.lens = 50
    camera.data.clip_start, camera.data.clip_end = 0.001, 1000
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        scene.frame_set(frame)
        for view in views:
            if view in cameras:
                camera.matrix_world = Matrix(cameras[view]['matrix_world'])
                camera.data.angle = cameras[view]['x_fov']
            else:
                if args.turntable:
                    angle = 2 * math.pi * index / len(frames)
                    elevation = math.radians(15)
                    camera.location = (2 * math.sin(angle) * math.cos(elevation),
                                       -2 * math.cos(angle) * math.cos(elevation),
                                       2 * math.sin(elevation))
                else:
                    camera.location = POSITIONS[view]
                camera.rotation_euler = (-camera.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()
            filename = f'{view}_frame{frame:04d}.png'
            scene.render.filepath = str((args.output_dir / filename).resolve())
            bpy.ops.render.render(write_still=True)
            records.append({'file': filename, 'frame': frame, 'view': view,
                            'matrix_world': [list(row) for row in camera.matrix_world],
                            'x_fov': camera.data.angle_x, 'y_fov': camera.data.angle_y})
    metadata = {'input': str(args.input.resolve()), 'blender': bpy.app.version_string,
                'frames': frames, 'views': views, 'fps': args.fps,
                'detected_animation_range': [detected_start, detected_end],
                'normalization': normalization, 'resolution': args.resolution,
                'samples': args.samples, 'device': args.device, 'shading': args.shading,
                'excluded_objects': args.exclude_object,
                'lighting': 'fixed_studio', 'transparent': True, 'cameras': records}
    (args.output_dir / 'render.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Rendered {len(records)} images to {args.output_dir}', flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Animated GLB/glTF or static OBJ.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--views', nargs='+', choices=VIEWS, default=list(VIEWS))
    parser.add_argument('--frame-start', type=int, help='Inclusive; defaults to the imported animation start.')
    parser.add_argument('--frame-end', type=int, help='Inclusive; defaults to the imported animation end.')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--fps', type=int, default=24, help='glTF import timebase; Mesh4D exports use 24.')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--device', choices=('CPU', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'), default='CPU')
    parser.add_argument('--shading', choices=('material', 'clay'), default='material')
    parser.add_argument('--exclude-object', action='append', default=[], metavar='NAME',
                        help='Hide an imported mesh by its exact Blender name; repeat for multiple meshes.')
    parser.add_argument('--normalize', choices=('sequence', 'first-frame', 'none'), default='sequence',
                        help='Fit all selected frames into a unit cube, use the first frame, or preserve coordinates.')
    parser.add_argument('--camera-dir', type=Path, help='Directory with <view>.json benchmark cameras; requires --normalize none.')
    parser.add_argument('--mesh4d-coordinates', action='store_true',
                        help='Apply the research Mesh4D GLB canonical normalization/axis conversion; requires --normalize none.')
    parser.add_argument('--turntable', action='store_true', help='Orbit once over the selected frames instead of rendering fixed views.')
    parser.add_argument('--overwrite', action='store_true')
    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    try:
        if not args.input.is_file() or args.input.suffix.lower() not in ('.glb', '.gltf', '.obj'):
            raise ValueError('--input must be an existing .glb, .gltf, or .obj file.')
        if min(args.frame_step, args.fps, args.samples) < 1 or args.resolution < 16:
            raise ValueError('Use positive FPS, step and samples, and resolution >= 16.')
        if args.camera_dir and (args.normalize != 'none' or args.turntable):
            raise ValueError('--camera-dir requires --normalize none and fixed views.')
        if args.mesh4d_coordinates and args.normalize != 'none':
            raise ValueError('--mesh4d-coordinates requires --normalize none to avoid normalizing twice.')
        args.views = list(dict.fromkeys(args.views))
        render(args)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
