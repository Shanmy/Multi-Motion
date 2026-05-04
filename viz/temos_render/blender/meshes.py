import numpy as np
from PIL import Image, ImageDraw

from .materials import body_material
import colorsys

# green
# GT_SMPL = body_material(0.009, 0.214, 0.029)
GT_SMPL = body_material(0.035, 0.415, 0.122)

# blue
# GEN_SMPL = body_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
GEN_SMPL = body_material(0.035, 0.322, 0.615)


def rotation_matrix_x(angle):
    """Return matrix for rotating around the x-axis by `angle` degrees."""
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle):
    """Return matrix for rotating around the y-axis by `angle` degrees."""
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle):
    """Return matrix for rotating around the z-axis by `angle` degrees."""
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def hsv_to_rgb(h, s, v):
    """
    Convert HSV color space (with H as 0-360, S and V as 0-100) to RGB color space

    :param h: Hue (0-360)
    :param s: Saturation (0-100)
    :param v: Value (0-100)
    :return: (r, g, b) in the range 0-255
    """
    # Normalize H, S, V values
    h_normalized = h / 360
    s_normalized = s / 100
    v_normalized = v / 100

    r, g, b = colorsys.hsv_to_rgb(h_normalized, s_normalized, v_normalized)
    return int(r * 255), int(g * 255), int(b * 255)


def interpolate_hsv(color1, color2, factor):
    # Interpolating in HSV space
    h = color1[0] + (color2[0] - color1[0]) * factor
    s = color1[1] + (color2[1] - color1[1]) * factor
    v = color1[2] + (color2[2] - color1[2]) * factor
    return h, s, v


def _get_frame_color(factor, start_color, end_color):
    # Calculate the interpolation factor
    # factor = frame_index / total_frames

    # Interpolate between the start color and end color in HSV
    interpolated_hsv = interpolate_hsv(start_color, end_color, factor)

    # Convert the interpolated HSV color to RGB
    return hsv_to_rgb(*interpolated_hsv)



def generate_gradient(width, height, start_color, mid_color, end_color):
    """
    Generate a horizontal gradient with three colors from left to right.
    :param width: The width of the image
    :param height: The height of the image
    :param start_color: The color to start with (left-side)
    :param mid_color: The middle transition color
    :param end_color: The color to end with (right-side)
    :return: An image with a horizontal three-color gradient
    """
    # Create a new image with RGB mode
    image = Image.new("RGB", (width, height))
    
    # Initialize the draw object
    draw = ImageDraw.Draw(image)
    
    # Generate gradient
    for i in range(width):
        # Calculate the ratio of the current position in the gradient
        ratio = i / width
        if ratio <= 0.5:  # First half of the gradient
            r = int(start_color[0] + (mid_color[0] - start_color[0]) * (ratio * 2))
            g = int(start_color[1] + (mid_color[1] - start_color[1]) * (ratio * 2))
            b = int(start_color[2] + (mid_color[2] - start_color[2]) * (ratio * 2))
        else:  # Second half of the gradient
            r = int(mid_color[0] + (end_color[0] - mid_color[0]) * ((ratio - 0.5) * 2))
            g = int(mid_color[1] + (end_color[1] - mid_color[1]) * ((ratio - 0.5) * 2))
            b = int(mid_color[2] + (end_color[2] - mid_color[2]) * ((ratio - 0.5) * 2))
        
        # Draw a line with the calculated color
        draw.line((i, 0, i, height), fill=(r, g, b))
    
    return image

# Define the size of the image
image_width = 1024
image_height = 256

# Define the colors (RGB)
start_color_rgb = (209, 107, 165) # Hex #D16BA5
mid_color_rgb = (134, 158, 231)   # Hex #869EE7
end_color_rgb = (95, 251, 241)    # Hex #5FFBF1

# Generate the gradient
# gradient_image = generate_gradient(image_width, image_height, start_color_rgb, mid_color_rgb, end_color_rgb)

# # Save the image
# gradient_image_path = '/mnt/data/gradient_image.png'
# gradient_image.save(gradient_image_path)


# gradient_image_path

def pick_color(start_color, mid_color, end_color, frac):
    ratio = frac
    if ratio <= 0.5:  # First half of the gradient
        r = int(start_color[0] + (mid_color[0] - start_color[0]) * (ratio * 2))
        g = int(start_color[1] + (mid_color[1] - start_color[1]) * (ratio * 2))
        b = int(start_color[2] + (mid_color[2] - start_color[2]) * (ratio * 2))
    else:  # Second half of the gradient
        r = int(mid_color[0] + (end_color[0] - mid_color[0]) * ((ratio - 0.5) * 2))
        g = int(mid_color[1] + (end_color[1] - mid_color[1]) * ((ratio - 0.5) * 2))
        b = int(mid_color[2] + (end_color[2] - mid_color[2]) * ((ratio - 0.5) * 2))
    return (r, g, b)


class Meshes:
    def __init__(self, data, *, gt, mode, faces_path, canonicalize, always_on_floor, oldrender=True, **kwargs):
        data = prepare_meshes(data, canonicalize=canonicalize, always_on_floor=always_on_floor)

        self.faces = np.load(faces_path)
        self.data = data
        # print(data.shape)
        self.data = data @ rotation_matrix_x(0)
        # self.data = 
        self.data = self.data @ rotation_matrix_y(0)

        self.data = self.data @ rotation_matrix_z(0)
        self.data  = self.data - self.data[:, :, 2].min()

        self.mode = mode
        self.oldrender = oldrender

        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)

        if gt:
            self.mat = GT_SMPL
        else:
            self.mat = GEN_SMPL

        if mode == 'sequence':
            self.width = (self.data[:, :, 0].max() - self.data[:, :, 0].min()) * 0.0
        else:
            self.width = 0
        self.current_width = 0

        self.current_frame_id = 0

        self.moved_data = self.data

    def _update_moved_data(self, vertices):
        # Expand vertices to shape (1, nv, 3)
        expanded_vertices = np.expand_dims(vertices, axis=0)

        # Append expanded vertices to data
        self.moved_data = np.concatenate((self.moved_data, expanded_vertices), axis=0)

    def get_sequence_mat(self, frac):
        
        # cmap = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        start_color_rgb = (209, 107, 165) # Hex #D16BA5
        mid_color_rgb = (134, 158, 231)   # Hex #869EE7
        end_color_rgb = (95, 251, 241)    # Hex #5FFBF1
        # rgbcolor = cmap(begin + (end-begin)*frac)
        rgbcolor = (0.1791464821222607, 0.49287197231833907, 0.7354248366013072, 1.0)
        rgbcolor = pick_color(
            start_color_rgb, mid_color_rgb, end_color_rgb, frac
        )

        # yellow hsv way
        start_color = (40, 20, 70)
        # end_color = (30, 80, 70)
        # start_color = (30, 80, 70)

        # VVVVVVVVVV for human color VVVVVVVVVV
        start_color = (180, 10, 100)
        end_color = (200, 60, 100)
        # frac = 0.9
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        rgbcolor = _get_frame_color(
            frac, start_color, end_color
        )


        rgbcolor = (
            rgbcolor[0] / 255.,
            rgbcolor[1] / 255.,
            rgbcolor[2] / 255.,
            1.0
        )

        in_place = False
        
        if not in_place:
            if self.current_frame_id % 8 == 0:
                rgbcolor = (0.7, 0.3, 0.3, 0.7)
            elif self.current_frame_id % 8 == 2:
                rgbcolor = (0.3, 0.7, 0.3, 0.7)
            elif self.current_frame_id % 8 == 1:
                rgbcolor = (0.3, 0.3, 0.7, 0.7)
            elif self.current_frame_id % 8 == 3:
                rgbcolor = (0.3, 0.7, 0.7, 0.7)
            elif self.current_frame_id % 8 == 4:
                rgbcolor = (0.7, 0.7, 0.3, 0.7)
            elif self.current_frame_id % 8 == 5:
                rgbcolor = (0.7, 0.3, 0.7, 0.7)
            elif self.current_frame_id % 8 == 6:
                rgbcolor = (0.7, 0.7, 0.7, 0.7)
            elif self.current_frame_id % 8 == 7:
                rgbcolor = (0.3, 0.3, 0.3, 0.7)
            self.current_frame_id += 1
        else:
            num_person = 1
            id_in_seq = self.current_frame_id // num_person
            person_id = self.current_frame_id % num_person
            if person_id == 0:
                rgbcolor = (0.3 + id_in_seq * 0.1, 0.3, 0.3, 0.1)
            elif person_id == 2:
                rgbcolor = (0.3, 0.3 + id_in_seq * 0.1, 0.3, 1.0)
            elif person_id == 1:
                rgbcolor = (0.3, 0.3, 0.3 + id_in_seq * 0.1, 0.1)

            self.current_frame_id += 1

        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        # old version 1:
        # return self.moved_data.mean((0, 1))

        # new version 1:
        min_values = self.moved_data.min(axis=(0, 1))
        max_values = self.moved_data.max(axis=(0, 1))
        # Calculate the center as the midpoint between the min and max values
        center = (max_values + min_values) / 2
        return center

    def load_in_blender(self, index, mat):
        # handle vertices
        vertices = self.data[index]
        vertices[:, 0] = vertices[:, 0] + self.current_width
        self.current_width = self.current_width + self.width

        self._update_moved_data(vertices)

        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N


def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fix axis
    data[..., 1] = - data[..., 1]
    data[..., 0] = - data[..., 0]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data