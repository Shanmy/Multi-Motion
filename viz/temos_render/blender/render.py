import bpy
import os
import numpy as np
import io
from contextlib import redirect_stdout

from .scene import setup_scene  # noqa
from .floor import show_traj, plot_floor, get_trajectory
from .vertices import prepare_vertices
from .tools import load_numpy_vertices_into_blender, delete_objs, mesh_detect
from .camera import Camera
from .sampler import get_frameidx


def render_current_frame(path):
    
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=False, write_still=True)

def render(npydata, frames_folder, *, mode, faces_path, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,
           res="high", init=True):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender)

    is_mesh = mesh_detect(npydata)
    assert is_mesh, "data is not in mesh format!"

    img_name, ext = os.path.splitext(frames_folder)
    if always_on_floor:
        img_name += "_of"
    img_path = f"{img_name}{ext}"

    from .meshes import Meshes
    data = Meshes(npydata, gt=gt, mode=mode,
                    faces_path=faces_path,
                    canonicalize=canonicalize,
                    always_on_floor=always_on_floor)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory TODO: remove trajectory
    # show_traj(data.trajectory)

    # initialize the camera
    camera = Camera(
       first_root=data.get_root(0), mode=mode,
       is_mesh=is_mesh
    )

    frameidx = get_frameidx(
        mode=mode, nframes=nframes,
        exact_frame=exact_frame, frames_to_keep=num
    )

    nframes_to_render = len(frameidx)

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        frac = index / (nframes_to_render-1) if nframes_to_render != 1 else index
        mat = data.get_sequence_mat(frac)

        objname = data.load_in_blender(frameidx, mat)
        imported_obj_names.append(objname)

        if index == (nframes_to_render-1):
            # Create a floor
            stdout = io.StringIO()
            with redirect_stdout(stdout):
            
                plot_floor(data.moved_data, small_plane=True, big_plane=False)
                render_current_frame(img_path)
                delete_objs(objname)

    # bpy.ops.wm.save_as_mainfile(filepath="./tmp.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    return img_path