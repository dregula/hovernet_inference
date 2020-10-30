# parse_centroid_output.py

import os
import numpy as np
import configargparse
from sklearn.neighbors import NearestNeighbors
import cv2

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def parse_args():
    import configargparse
    import yaml

    # Parse command line arguments
    # ap = argparse.ArgumentParser(description="Image processing pipeline")
    # ap = configargparse.ArgumentParser(description="Image processing pipeline")
    ap = configargparse.getParser(default_config_files=["config_parse_output.yaml"],
                                  config_file_parser_class=configargparse.ConfigparserConfigFileParser)
    # from config_parse_output.yaml
    # ap.add_argument('--philips_exported_tiff', type=yaml.safe_load)

    # usually from commandline arguments
    ap.add_argument("-i", "--input_dir", help="path to input directory")
    ap.add_argument("-bim", "--base_image_file", help="path to base_image_file", required=True)
    ap.add_argument("-cp", "--centroid_points_file", default="centroid.npy",
                    help="path to centroid numpy file")
    ap.add_argument("-o", "--output_dir", default="output",
                    help="path to output directory")
    ap.add_argument("-co", "--centroid_output",
                    help="the centroid.npy file from the hovernet_inference analysis")

    ap.add_argument("-d", "--display", action="store_true", help="display video result")
    ap.add_argument("-os", "--out-summary", default="summary.json",
                    help="output JSON summary file name")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch size")
    ap.add_argument("-p", "--progress", action="store_true", help="display progress")

    return ap.parse_args()


def parse_centroid_output(args):
    base_image_file = os.path.realpath(args.base_image_file)
    base_image_file = base_image_file.replace("\\", "/")
    base_image = None
    this_filepath = os.path.realpath(__file__)
    this_filepath = this_filepath.replace("\\", "/")
    base_dir = os.path.dirname(this_filepath)
    base_dir = os.path.realpath(base_dir)
    base_dir = base_dir.replace("\\", "/")
    print(f"base_dir:{base_dir}")
    print(f"'bare' base_image path:{base_image_file}")
    if not os.path.exists(base_image_file):
        base_image_file = os.path.join(base_dir, base_image_file).replace("\\", "/")
        print(f"'rebased' base_image path:{base_image_file}")
    if not os.path.exists(base_image_file):
        raise TypeError(f"Cannot find base image:{args.base_image_file}")

    base_image = cv2.imread(args.base_image_file)

    centroid_points_file = args.centroid_points_file
    centroid_points = None
    if not os.path.exists(centroid_points_file):
        centroid_points_file = os.path.join(base_dir, centroid_points_file)
    if not os.path.exists(centroid_points_file):
        raise TypeError(f"Cannot find centroid points numpy file:{args.centroid_points_file}")

    centroid_points = np.load(centroid_points_file)
    print(f"centroid_points SHAPE: {centroid_points.shape}; DTYPE:{centroid_points.dtype}")
    # print(centroid_points)
    max_y = np.max(centroid_points[:, 0])
    max_x = np.max(centroid_points[:, 1])
    print(f"max_x(centroid_points):{max_x}; max_y(centroid_points):{max_y}")
    reduced_centroid_points = np.asarray(centroid_points//(2**6)).astype(int)
    print(f"reduced_centroid_points SHAPE:{reduced_centroid_points.shape}; DTYPE:{reduced_centroid_points.dtype}")
    max_y = np.max(reduced_centroid_points[:, 0])
    max_x = np.max(reduced_centroid_points[:, 1])
    print(f"max_x(reduced_centroid_points):{max_x}; max_y(reduced_centroid_points):{max_y}")

    # https://stackoverflow.com/questions/45127141/find-the-nearest-point-in-distance-for-all-the-points-in-the-dataset-python
    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    nearest_neighbors_centroid_points = nearest_neighbors.fit(centroid_points)
    distances, indices = nearest_neighbors_centroid_points.kneighbors(centroid_points)
    nearest_neighbor_distances = distances[:, 1]
    print(f"nearest_neighbor_distances SHAPE: {nearest_neighbor_distances.shape}")

    # num_bins = 100
    # plt.xlim(xmin=0, xmax=300)
    # n, bins, patches = plt.hist(nearest_neighbor_distances, num_bins, facecolor='blue', alpha=0.5)
    # plt.show()

    # TRY: 2020-10-27
    print(f"base_image SHAPE:{base_image.shape}; DTYPE:{base_image.dtype}")
    h, w, _ = base_image.shape
    mask = np.zeros(shape=(h, w), dtype=bool)
    print(f"mask SHAPE:{mask.shape}; DTYPE:{mask.dtype}")

    mask[reduced_centroid_points] = True
    centroid_coordinates = np.argwhere(mask)
    print(f"centroid_coordinates SHAPE:{centroid_coordinates.shape}; centroid_coordinates:{centroid_coordinates.dtype}")

    display_image = np.copy(base_image)

    # cv2.imshow('base_image', display_image)
    # cv2.waitKey()
    #
    # COMPLETELY TRASH
    # print(f"centroid_coordinates[0]:{centroid_coordinates[0]}")
    # print(f"display_image[centroid_coordinates[:, 0],centroid_coordinates[:, 1], ]- y:{centroid_coordinates[:, 0]}, x:{centroid_coordinates[:, 1]}, bgr:{display_image[centroid_coordinates[:, 0],centroid_coordinates[:, 1], ] }")
    # # print(f"display_image[centroid_coordinates[:, :], ]:{display_image[centroid_coordinates, ] }")
    # exit('debug')
    # # remember BGR!
    # display_image[centroid_coordinates[:, 0],centroid_coordinates[:, 1], ] = [0, 255, 255]
    # # display_image[mask, 2] = 255
    # cv2.imshow('yellow_centroids', display_image)
    # cv2.waitKey()


if __name__ == "__main__":
    args = parse_args()
    parse_centroid_output(args)
    exit("done!")
