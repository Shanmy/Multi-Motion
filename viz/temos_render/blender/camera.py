import bpy
import math


class Camera:
    def __init__(self, *, first_root, mode, is_mesh):
        camera = bpy.data.objects['Camera']

        # wider point of view

        if mode == 'sequence':
            # VVVVVVVVVVVVVVVVVVVVVVVVV the below is just test VVVVVVVVVVVVVVVVVVV
            # Position the camera
            # This will depend on the size of your figures and their arrangement
            camera.location.x = 7  # Adjust this value as needed
            camera.location.y = 0  # Move the camera back to fit all figures in the frame
            camera.location.z = 3.5  # Adjust the height to get the slight downward angle

            # Rotate the camera to point slightly downwards
            camera.rotation_euler.x = 1.2  # Adjust this value to change the angle
            camera.rotation_euler.y = 0.0
            camera.rotation_euler.z = 1.57 * 1
            # Set the focal length
            camera.data.lens = 50  # You can adjust this value to get the desired perspective
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        #self.camera.location.x += first_root[0]
        #self.camera.location.y += first_root[1]

        #self._root = first_root


    def update(self, newroot):
        pass
        # delta_root = newroot - self._root

        # self.camera.location.x += delta_root[0]
        # self.camera.location.y += delta_root[1]

        # self._root = newroot