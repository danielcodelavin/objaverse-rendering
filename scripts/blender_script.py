"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import math
import json
import os
import random
import sys
import time
import urllib.request
from typing import Tuple

import bpy
from mathutils import Vector
import addon_utils
import bpy, addon_utils, time

addon_utils.enable("cycles", default_set=True, persistent=True)

prefs      = bpy.context.preferences
cy_prefs   = prefs.addons["cycles"].preferences
cy_prefs.compute_device_type = "OPTIX"       # or "CUDA" if OptiX not installed

cy_prefs.get_devices()                       # refresh list in Blender ≤ 3.4
# cy_prefs.refresh_devices()                 # use this in Blender ≥ 3.5

# ---- print BEFORE enabling ------------------------------------------
print(f"[{time.time():.6f}]  devices BEFORE enabling:",
      [(d.name, d.type, d.use) for d in cy_prefs.devices],
      flush=True)

for dev in cy_prefs.devices:                 # enable every visible GPU
    dev.use = (dev.type != 'CPU')

# ---- print AFTER enabling -------------------------------------------
print(f"[{time.time():.6f}]  devices AFTER enabling:",
      [(d.name, d.type, d.use) for d in cy_prefs.devices],
      flush=True)

assert cy_prefs.has_active_device(), "No GPU enabled – aborting."

bpy.context.scene.cycles.device = "GPU"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--camera_dist", type=int, default=1.5)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 256
render.resolution_y = 256
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.use_adaptive_sampling = True
scene.cycles.samples = 8
scene.cycles.use_persistent_data = True
scene.cycles.max_bounces = 1
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.use_auo_tile = False
scene.cycles.transparent_max_bounces = 1
scene.cycles.transmission_bounces = 1
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.render.use_simplify = True
scene.render.simplify_subdivision_render = 0


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]

    normalize_scene()
    add_lighting()

    cam, cam_constraint = setup_camera()
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    frames_meta = []   # ## CHANGE: collect metadata per frame

    for i in range(args.num_images):
        # ## CHANGE: random camera within specified shell & height range
        # -----------------------------------------------------------
        r = random.uniform(1.5, 2.4)
        # sample a random point on unit sphere
        x, y, z = sample_point_on_sphere(1.0)
        # enforce height range [-0.75, 1.60]
        # scale z to desired range while keeping (x,y) direction
        z = random.uniform(-0.75, 1.60) / r
        xy_scale = math.sqrt(max(1 - z ** 2, 0))
        x *= xy_scale
        y *= xy_scale
        cam.location = (r * x, r * y, r * z)
        # -----------------------------------------------------------

        render_path = os.path.join(args.output_dir, object_uid, f"{i:05d}.png")
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # ## CHANGE: compute intrinsics & store w2c for this frame
        sensor_w = cam.data.sensor_width
        sensor_h = cam.data.sensor_height if cam.data.sensor_fit == 'VERTICAL' else sensor_w
        fx = cam.data.lens * render.resolution_x / sensor_w
        fy = cam.data.lens * render.resolution_y / sensor_h
        cx = render.resolution_x / 2
        cy = render.resolution_y / 2

        frame_meta = {
            "image_path": os.path.abspath(render_path),
            "fxfycxcy": [fx, fy, cx, cy],
            "w2c": [list(row) for row in cam.matrix_world.inverted()],
        }
        frames_meta.append(frame_meta)

    # ## CHANGE: write per‑object metadata json
    meta_dir = os.path.join(os.path.dirname(args.output_dir.rstrip(os.sep)), "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{object_uid}.json"), "w") as f:
        json.dump({"scene_name": object_uid, "frames": frames_meta}, f, indent=2)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
