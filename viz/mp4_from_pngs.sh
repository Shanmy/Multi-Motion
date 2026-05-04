PNG_FOLDER="$1"
FRAMERATE="$2"

ffmpeg -framerate $FRAMERATE -pattern_type glob -i "$PNG_FOLDER/*/*.png" -pix_fmt yuv420p $PNG_FOLDER/motion.mp4
