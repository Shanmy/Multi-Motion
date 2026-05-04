VERTEX_NPY_PATH="$1"

/nfs/USRCSEA/IVA/Models/blender-2.93.18-linux-x64/blender --background --python render.py -- npy=$VERTEX_NPY_PATH
