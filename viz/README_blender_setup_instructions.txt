Instructions based on this doc: https://github.com/Mathux/TEMOS#rendering-motions-high_brightness

First locate the path to blender. The original setup is here:

/nfs/USRCSEA/IVA/Models/blender-2.93.18-linux-x64/blender

For the rest of this doc, this is refered to as "blender_path".

Next get the location of python blender by running the line:

blender_path --background --python-expr "import sys; import os; print('\nThe path to the installation of python of blender can be:'); print('\n'.join(['- '+x.replace('/lib/python', '/bin/python') for x in sys.path if 'python' in (file:=os.path.split(x)[-1]) and not file.endswith('.zip')]))"

For the rest of this doc, this path is referred to as "blender_python". The original "blender_python" is:

/nfs.auto/USRCSEA/IVA/Models/blender-2.93.18-linux-x64/2.93/python/bin/python3.9

Next, install pip:

blender_python -m ensurepip --upgrade

Then, install the following packages:

blender_python -m pip install numpy
blender_python -m pip install matplotlib
blender_python -m pip install hydra-core --upgrade
blender_python -m pip install hydra_colorlog --upgrade
blender_python -m pip install moviepy
blender_python -m pip install shortuuid

NOTE: After doing this one time, it should work for all users. The default path should already be set up correctly.
