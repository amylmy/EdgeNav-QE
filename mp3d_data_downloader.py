import argparse
import collections
import os
import tempfile
import urllib
from urllib import request
import requests
from time import sleep
from requests.exceptions import ConnectionError, ChunkedEncodingError, RequestException
import sys

OUT_DIR = "./matterport3d_data"
TASK_DATA = ["habitat"]
SCAN_ID = "ALL"
FILE_TYPES = None

BASE_URL = 'http://kaldir.vc.in.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    'cameras',
    'matterport_camera_intrinsics',
    'matterport_camera_poses',
    'matterport_color_images',
    'matterport_depth_images',
    'matterport_hdr_images',
    'matterport_mesh',
    'matterport_skybox_images',
    'undistorted_camera_parameters',
    'undistorted_color_images',
    'undistorted_depth_images',
    'undistorted_normal_images',
    'house_segmentations',
    'region_segmentations',
    'image_overlap_data',
    'poisson_meshes',
    'sens'
]
TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': ['mp3d_360/data_00.zip', 'mp3d_360/data_01.zip', 'mp3d_360/data_02.zip', 'mp3d_360/data_03.zip', 'mp3d_360/data_04.zip', 'mp3d_360/data_05.zip', 'mp3d_360/data_06.zip']
}

def get_release_scans(release_file):
    scan_lines = request.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_line = str(scan_line, 'utf-8')
        scan_id = scan_line.rstrip('\n')
        scans.append(scan_id)
    return scans

def download_release(release_scans, out_dir, file_types):
    print('Downloading MP release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types)
    print('Downloaded MP release.')

def download_file(url, out_file, max_retries=5, chunk_size=1024*1024):
    """
    Downloads a single file with resumable download and automatic retry, and prints real-time download progress.
    Compatible with Python2/3.
    """
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir)
        except OSError:
            pass

    head = requests.head(url, allow_redirects=True)
    if head.status_code != 200:
        raise IOError("Failed to get file size: {0}, status code {1}".format(url, head.status_code))
    total_size = int(head.headers.get('Content-Length', 0))

    resume = 0
    if os.path.exists(out_file):
        resume = os.path.getsize(out_file)

    if resume >= total_size:
        print("Skipping existing file %s" % out_file)
        return

    print("Starting download of %s (%0.2f MB)" % (out_file, total_size / 1024.0**2))

    retries = 0
    last_print = 0
    while resume < total_size and retries <= max_retries:
        headers = {"Range": "bytes=%d-%d" % (resume, total_size - 1)}
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            r.raise_for_status()

            fh, tmp_path = tempfile.mkstemp(dir=out_dir)
            with os.fdopen(fh, "wb") as tmpf:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    tmpf.write(chunk)
                    resume += len(chunk)

                    percent = int(resume * 100 / total_size)
                    if percent - last_print >= 5 or resume == total_size:
                        sys.stdout.write("\rDownload progress: %3d%% (%0.2f/%0.2f MB)" % (
                            percent,
                            resume / 1024.0**2,
                            total_size / 1024.0**2
                        ))
                        sys.stdout.flush()
                        last_print = percent

            with open(tmp_path, "rb") as tmpf, open(out_file, "ab") as outf:
                outf.write(tmpf.read())
            os.remove(tmp_path)
            r.close()
            break

        except (ConnectionError, ChunkedEncodingError, IOError) as e:
            retries += 1
            wait = 2 ** retries
            print("\n[Retry %d/%d] Downloaded %0.2f MB, waiting %d seconds before retrying..." % (
                retries, max_retries,
                resume / 1024.0**2,
                wait
            ))
            sleep(wait)

    if resume < total_size:
        raise IOError("Download failed: only obtained %d / %d bytes" % (resume, total_size))

    sys.stdout.write("\nDownload completed: %s\n" % out_file)

def download_scan(scan_id, out_dir, file_types):
    print('Downloading MP scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip'
        out_file = out_dir + '/' + ft + '.zip'
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)

def download_task_data(task_data, out_dir):
    print('Downloading MP task data for ' + str(task_data) + ' ...')
    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            file = TASK_FILES[task_data_id]
            for filepart in file:
                url = BASE_URL + RELEASE_TASKS + '/' + filepart
                localpath = os.path.join(out_dir, filepart)
                localdir = os.path.dirname(localpath)
                if not os.path.isdir(localdir):
                    os.makedirs(localdir)
                download_file(url, localpath)
            print('Downloaded task data ' + task_data_id)

def main():
    out_dir = OUT_DIR
    task_data = TASK_DATA
    scan_id = SCAN_ID
    file_types = FILE_TYPES if FILE_TYPES is not None else FILETYPES

    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file)

    if task_data:
        if set(task_data) & set(TASK_FILES.keys()):
            task_out_dir = os.path.join(out_dir, RELEASE_TASKS)
            download_task_data(task_data, task_out_dir)
        else:
            print('ERROR: Unrecognized task data id: ' + str(task_data))
        print('Done downloading task_data for ' + str(task_data))
        print("Skipping input: Continue to main dataset download...")

    if FILE_TYPES is not None:
        if not set(file_types) & set(FILETYPES):
            print('ERROR: Invalid file type: ' + str(file_types))
            return

    if scan_id and scan_id != 'ALL':
        if scan_id not in release_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            single_scan_out_dir = os.path.join(out_dir, RELEASE, scan_id)
            download_scan(scan_id, single_scan_out_dir, file_types)
    elif 'minos' not in task_data and (scan_id == 'ALL' or scan_id == 'all'):
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire MP release which requires ' + RELEASE_SIZE + ' of space.')
            print('WARNING: Ensure you have enough disk space!')
        else:
            print('WARNING: You are downloading all MP scans of type ' + str(file_types[0]))
        print('Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download.')
        print("Skipping input: Starting full dataset download...")
        
        full_out_dir = os.path.join(out_dir, RELEASE)
        download_release(release_scans, full_out_dir, file_types)

if __name__ == "__main__":
    main()