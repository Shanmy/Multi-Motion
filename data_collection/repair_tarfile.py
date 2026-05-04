import os
import shutil
import tarfile
import argparse


def repair_tarfile(
    tar_path='/home/us000240/test_tar/00645.tar',
    broken_id_list=[],
    work_dir='tmp/',
    sample_extns=['.jpg', '.json', '.txt'],
    backup_loc='/home/us000240/old_tar_archive',
):
    # create copy in local working dir
    os.makedirs(work_dir, exist_ok=True)
    local_tar_path = os.path.join(work_dir, 'copy_' + os.path.basename(tar_path))
    shutil.copyfile(tar_path, local_tar_path)
    extract_dir = os.path.join(work_dir, 'extract/')
    os.makedirs(extract_dir)

    # backup tarfile in case of issues
    backup_tarfile = os.path.join(backup_loc, os.path.basename(tar_path))
    if not os.path.isfile(backup_tarfile):
        shutil.copyfile(tar_path, backup_tarfile)

    # untar in working dir
    tf = tarfile.open(local_tar_path)
    for i, sub_file in enumerate(tf):
        try:
            tf.extract(sub_file.name, path=extract_dir)
        except:
            print(f'Extraction error, skipping file {sub_file.name}.')
    tf.close()
    print(f'Successfully extracted {i + 1} files, approximately {(i + 1) / 3} samples.')

    # create tar from all remaining valid samples
    out_tar_path = os.path.join(work_dir, os.path.basename(tar_path))
    tf = tarfile.open(out_tar_path, "w")
    extracted_ids = list(set([file_name.split('.')[0] for file_name in os.listdir(extract_dir)]))
    out_count = 0
    for id_num in extracted_ids:
        if check_sample(id_num, sample_extns, extract_dir) and id_num not in broken_id_list:
            for extn in sample_extns:
                out_count += 1
                file_name_out = str(id_num) + extn
                tf.add(os.path.join(extract_dir, file_name_out), arcname=file_name_out)
    tf.close()
    print(f'Created new tarfile with {out_count} files, {out_count / 3} complete samples.')

    # replace tarfile in original location, and cleanup
    shutil.rmtree(extract_dir)
    os.remove(local_tar_path)
    shutil.copyfile(out_tar_path, tar_path)
    os.remove(out_tar_path)


def check_sample(sample_id, extns, extract_dir):
    valid_sample = True
    for extn in extns:
        if not os.path.isfile(os.path.join(extract_dir, str(sample_id) + extn)):
            valid_sample = False
    return valid_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tar_path',
        default='/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data2/00647.tar',
        help='path to tarfile that should be repaired'
    )
    parser.add_argument(
        '--broken_ids',
        default='006451078',
        help='ids of files to remove, separated by commas',
    )
    args = parser.parse_args()
    broken_id_list = [int(item) for item in args.broken_ids.split(',')]
    repair_tarfile(args.tar_path, broken_id_list)
