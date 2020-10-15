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
    ap = configargparse.getParser(default_config_files=["config.yaml"],
                                  config_file_parser_class=configargparse.ConfigparserConfigFileParser)
    # from config.yaml
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
    base_image_file = args.base_image_file
    base_image = None
    this_filepath = os.path.realpath(__file__).replace("\\", "/")
    base_dir = os.path.dirname(this_filepath)
    if not os.path.exists(base_image_file):
        base_image_file = os.path.join(base_dir, base_image_file)
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
    print(f"centroid_points SHAPE: {centroid_points.shape}")

    # https://stackoverflow.com/questions/45127141/find-the-nearest-point-in-distance-for-all-the-points-in-the-dataset-python
    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    nearest_neighbors_centroid_points = nearest_neighbors.fit(centroid_points)
    distances, indices = nearest_neighbors_centroid_points.kneighbors(centroid_points)
    result = distances[:, 1]
    print(f"result SHAPE: {result.shape}")

    num_bins = 100
    plt.xlim(xmin=0, xmax=300)
    n, bins, patches = plt.hist(result, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    parse_centroid_output(args)
    exit("done!")
