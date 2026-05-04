import bpy
from .materials import floor_mat


def get_trajectory(data, is_mesh):
    if is_mesh:
        # mean of the vertices
        trajectory = data[:, :, [0, 1]].mean(1)
    else:
        # get the root joint
        trajectory = data[:, 0, [0, 1]]
    return trajectory


def plot_floor(data, small_plane=True, big_plane=True, fix_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    
    if fix_plane:
    
        minx, miny = -3, -10
        maxx, maxy = 3, 10

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    scale = ((maxx - minx)/2, (maxy - miny)/2, 1)
    if small_plane:

        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                use_proportional_projected=False, release_confirm=True)
        obj = bpy.data.objects["Plane"]
        obj.name = "SmallPlane"
        obj.data.name = "SmallPlane"

        obj.active_material = floor_mat(color=(0.4, 0.4, 0.4, 1))


def show_traj(coords):
    # create the Curve Datablock
    curveData = bpy.data.curves.new('myCurve', type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 2

    # map coords to spline
    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords)-1)
    for i, coord in enumerate(coords):
        x, y = coord
        polyline.points[i].co = (x, y, 0.001, 1)

    # create Object
    curveOB = bpy.data.objects.new('myCurve', curveData)
    curveData.bevel_depth = 0.01

    bpy.context.collection.objects.link(curveOB)