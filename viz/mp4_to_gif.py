# modified from https://gist.github.com/michaelosthege/cd3e0c3c556b70a79deba6855deb2cc8
import imageio
import os, sys
import argparse


def convertFile(inputpath, outputpath=None, targetFormat=".gif", verbose=False):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
    if outputpath is None:
        outputpath = os.path.splitext(inputpath)[0] + targetFormat
    if verbose:
        print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i, im in enumerate(reader):
        if verbose:
            sys.stdout.write("\rframe {0}".format(i))
            sys.stdout.flush()
        writer.append_data(im)
    if verbose:
        print("\r\nFinalizing...")
    writer.close()
    if verbose:
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mp4_file",
        type=str,
        default="/home/notebook/data/group/video-diffusion/tests-v1/webvid_sd_vid/webvid_sd_vid_v1-5_unfrozen1_16_gpu/23-09-07-01-16-14/viz/vids_model_380000.mp4",
    )
    args = parser.parse_args()
    convertFile(args.mp4_file)
