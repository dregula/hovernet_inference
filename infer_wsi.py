import glob
import importlib
import math
import os
import sys
import time
import logging as plog
from collections import deque

import cv2
import numpy as np
import openslide as ops
# from tensorpack import PredictConfig, get_model_loader, OfflinePredictor
from tensorpack.utils import logger
# logger._getlogger().disabled = True # disable logging of network info

from hover.misc.utils import rm_n_mkdir, time_it
from hover.postproc import process_utils as proc_utils

from shared_infer_methods import SharedInferMethods

class InferWSI(SharedInferMethods):
    def __init__(self):
        self.nr_types = 6  # denotes number of classes (including BG) for nuclear type classification
        self.input_shape = [256, 256]
        self.mask_shape = [164, 164]
        self.input_norm = True # normalize RGB to 0-1 range
        #TODO: is this just the "starting level" to find the "correct size"
        #TODO: how does this relate to tiss_lvl ?
        self.proc_lvl = 0  # WSI level at which to process
        self.tiss_seg = True  # only process tissue areas

        # for inference during evalutation mode i.e run by infer.py
        self.input_tensor_names = ['images']
        self.output_tensor_names = ['predmap-coded']

    def read_region(self, location, level, patch_size, wsi_ext):
        """
        Loads a patch from an OpenSlide object

        Args:
            location: top left coordinates of patch
            level: level of WSI pyramid at which to extract
            patch_size: patch size to extract
            wsi_ext: WSI file extension

        Returns:
            patch: extracted patch (np array)
        """

        if wsi_ext == 'jp2':
            x1 = int(location[0] / pow(2, level)) + 1
            y1 = int(location[1] / pow(2, level)) + 1
            x2 = int(x1 + patch_size[0] -1)
            y2 = int(y1 + patch_size[1] -1)
            # this will read patch using matlab engine
            patch = self.wsiObj.read_region(self.full_filename, level, matlab.int32([y1,y2,x1,x2]))
            patch = np.array(patch._data).reshape(patch.size, order='F')
        else:
            patch = self.wsiObj.read_region(location, level, patch_size)
            r, g, b, _ = cv2.split(np.array(patch))
            patch = cv2.merge([r, g, b])
        return patch

    def load_wsi(self, wsi_ext):
        """
        Load WSI using OpenSlide. Note, if using JP2, appropriate
        matlab scripts need to be placed in the working directory

        Args:
            wsi_ext: file extension of the whole-slide image
        """

        if wsi_ext == 'jp2':
            try:
                self.wsiObj = engine.start_matlab()
            except:
                print ("Matlab Engine not started...")
            self.wsiObj.cd(os.getcwd() + '/hover', nargout=0)
            level_dim, level_downsamples, level_count  = self.wsiObj.JP2Image(self.full_filename, nargout=3)
            level_dim = np.float32(level_dim)
            self.level_dimensions = level_dim.tolist()
            self.level_count = np.int32(level_count)
            level_downsamples = np.float32(level_downsamples)
            self.level_downsamples = []
            for i in range(self.level_count):
                self.level_downsamples.append(level_downsamples[i][0])
            self.scan_resolution = [0.275, 0.275]  # scan resolution of the Omnyx scanner at UHCW
        else:
            self.wsiObj = ops.OpenSlide(self.full_filename)
            self.level_downsamples = self.wsiObj.level_downsamples
            self.level_count = self.wsiObj.level_count
            logger.info(f"OpenSlide with {self.level_count} levels...")
            self.scan_resolution = [float(self.wsiObj.properties.get('openslide.mpp-x')),
                                    float(self.wsiObj.properties.get('openslide.mpp-y'))]
            logger.info(f"Openslide scan-resolution: mpp-x{self.scan_resolution[0]},  mpp-y{self.scan_resolution[1]}")

            self.level_dimensions = []
            # flipping cols into rows (Openslide to python format)
            for i in range(self.level_count):
                self.level_dimensions.append([self.wsiObj.level_dimensions[i][1], self.wsiObj.level_dimensions[i][0]])
                logger.info(f"OpenSlide at level:{i}, width:{self.wsiObj.level_dimensions[i][1]}, height:{self.wsiObj.level_dimensions[i][0]}")

    ####

    def tile_coords(self):
        """
        Get the tile coordinates and dimensions for processing at level 0
        """

        self.im_w = self.level_dimensions[self.proc_lvl][1]
        self.im_h = self.level_dimensions[self.proc_lvl][0]

        self.nr_tiles_h = math.ceil(self.im_h / self.tile_size)
        self.nr_tiles_w = math.ceil(self.im_w / self.tile_size)

        step_h = self.tile_size
        step_w = self.tile_size

        self.tile_info = []

        for row in range(self.nr_tiles_h):
            for col in range(self.nr_tiles_w):
                start_h = row*step_h
                start_w = col*step_w
                if row == self.nr_tiles_h - 1:
                    extra_h = self.im_h - (self.nr_tiles_h * step_h)
                    dim_h = step_h + extra_h
                else:
                    dim_h = step_h
                if col == self.nr_tiles_w - 1:
                    extra_w = self.im_w - (self.nr_tiles_w * step_w)
                    dim_w = step_w + extra_w
                else:
                    dim_w = step_w
                self.tile_info.append(
                    (int(start_w), int(start_h), int(dim_w), int(dim_h)))
    ####

    def extract_patches(self, tile):
        """
        Extracts patches from the WSI before running inference.
        If tissue mask is provided, only extract foreground patches.

        Args:
            tile: tile number index
        """

        step_size = np.array(self.mask_shape)
        msk_size = np.array(self.mask_shape)
        win_size = np.array(self.input_shape)
        if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
            step_size = np.int64(step_size / 2)
            msk_size = np.int64(msk_size / 2)
            win_size = np.int64(win_size / 2)

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        last_h, self.nr_step_h = get_last_steps(self.tile_info[tile][3], msk_size[0], step_size[0])
        last_w, self.nr_step_w = get_last_steps(self.tile_info[tile][2], msk_size[1], step_size[1])

        start_h = self.tile_info[tile][1]
        start_w = self.tile_info[tile][0]
        last_h += start_h
        last_w += start_w

        self.sub_patches = []
        self.patch_idx = []
        self.patch_coords = []

        # Generating sub-patches from WSI
        idx = 0
        for row in range(start_h, last_h, step_size[0]):
            for col in range(start_w, last_w, step_size[1]):
                if self.tiss_seg is True:
                    win_tiss = self.tissue[
                               int(round(row / self.ds_factor_tiss)):int(round(row / self.ds_factor_tiss)) + int(
                                   round(win_size[0] / self.ds_factor_tiss)),
                               int(round(col / self.ds_factor_tiss)):int(round(col / self.ds_factor_tiss)) + int(
                                   round(win_size[1] / self.ds_factor_tiss))]
                    if np.sum(win_tiss) > 0:
                        self.patch_coords.append([row, col])
                        self.patch_idx.append(idx)
                else:
                    self.patch_coords.append([row, col])
                idx += 1

        # generate array of zeros - will insert patch predictions later
        self.zero_array = np.zeros([idx,self.mask_shape[0], self.mask_shape[1],9]) # 9 is the number of total output channels
    ####

    def load_batch(self, batch_coor, wsi_ext):
        """
        Loads a batch of images from provided coordinates.

        Args:
            batch_coor: list of coordinates in a batch
            wsi_ext   : file extension of the whole-slide image
        """

        batch = []
        win_size = self.input_shape
        if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
            win_size = np.int64(np.array(self.input_shape)/2)

        for coor in batch_coor:
            win = self.read_region((int(coor[1] * self.ds_factor), int(coor[0] * self.ds_factor)),
                                   self.proc_lvl, (win_size[0], win_size[1]), wsi_ext)
            if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
                win = cv2.resize(win, (win.shape[1]*2, win.shape[0]*2), cv2.INTER_LINEAR) # cv.INTER_LINEAR is good for zooming
            batch.append(win)
        return batch
    ####

    def run_inference(self, tile, wsi_ext):
        """
        Run inference for extracted patches and apply post processing.
        Results are then assembled to the size of the original image.

        Args:
            tile: tile number index
            wsi_ext: file extension of the whole-slide image
        """

        pred_map_list = deque()
        mask_list = []
        type_list = []
        cent_list = []
        offset = (self.input_shape[0] - self.mask_shape[0]) / 2
        idx = 0
        batch_count = np.floor(len(self.patch_coords) / self.batch_size)

        if len(self.patch_coords) > 0:
            while len(self.patch_coords) > self.batch_size:
                sys.stdout.write("\rBatch (%d/%d) of Tile (%d/%d)" % (
                idx + 1, batch_count, tile + 1, self.nr_tiles_h * self.nr_tiles_w))
                sys.stdout.flush()
                idx += 1
                mini_batch_coor = self.patch_coords[:self.batch_size]
                mini_batch = self.load_batch(mini_batch_coor, wsi_ext)
                self.patch_coords = self.patch_coords[self.batch_size:]
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, self.batch_size, axis=0)
                pred_map_list.extend(mini_output)

            # Deal with the case when the number of patches is not divisisible by batch size
            if len(self.patch_coords) != 0:
                mini_batch = self.load_batch(self.patch_coords, wsi_ext)
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, len(self.patch_coords), axis=0)
                pred_map_list.extend(mini_output)

            # Assemble back into full image
            output_patch_shape = np.squeeze(pred_map_list[0]).shape
            ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

            pred_map = self.zero_array
            pred_map[np.array(self.patch_idx)] = np.squeeze(np.array(pred_map_list))
            pred_map = np.reshape(pred_map, (self.nr_step_h, self.nr_step_w) + pred_map.shape[1:])
            pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                            np.transpose(pred_map, [0, 2, 1, 3])
            pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                            pred_map.shape[2] * pred_map.shape[3], ch))

            # crop back to original size
            if self.scan_resolution[0] > 0.35: # 20x
                pred_map = np.squeeze(pred_map[:self.tile_info[tile][3]*2,:self.tile_info[tile][2]*2])
            else:
                pred_map = np.squeeze(pred_map[:self.tile_info[tile][3],:self.tile_info[tile][2]])

            # post processing for a tile
            tile_coords = (self.tile_info[tile][0], self.tile_info[tile][1])
            mask_list, type_list, cent_list = proc_utils.process_instance_wsi(
                pred_map, self.nr_types, tile_coords, self.return_masks, offset=offset)

        else:
            mask_list = []
            type_list = []
            cent_list = []

        return mask_list, type_list, cent_list
    ####

    def process_wsi(self, filename):
        """
        Process an individual WSI. This function will:
        1) Load the OpenSlide WSI object
        2) Generate the tissue mask
        3) Get tile coordinate info
        4) Extract patches from foreground regions
        5) Run inference and return npz for each tile of
           masks, type predictions and centroid locations
        """

        # Load the OpenSlide WSI object
        self.full_filename = os.path.join(self.input_dir, filename)
        # self.input_dir + '/' + filename
        wsi_ext = self.full_filename.split('.')[-1]
        logger.info(f"inferWSI- filename:{self.full_filename}")
        # print(self.full_filename)
        # passes the file extension solely to correctly handle jp2 vs OpenSlide WSI
        self.load_wsi(wsi_ext)

        self.ds_factor = self.level_downsamples[self.proc_lvl]

        is_valid_tissue_level = True
        tissue_level = self.tiss_lvl
        if tissue_level < len(self.level_downsamples):  # if given tissue level exist
            self.ds_factor_tiss = self.level_downsamples[tissue_level] / self.level_downsamples[self.proc_lvl]
        elif len(self.level_downsamples) > 1:
            tissue_level = len(self.level_downsamples) - 1  # to avoid tissue segmentation at level 0
            self.ds_factor_tiss = self.level_downsamples[tissue_level] / self.level_downsamples[self.proc_lvl]
        else:
            is_valid_tissue_level = False

        if self.tiss_seg & is_valid_tissue_level:
            # Generate tissue mask
            ds_img = self.read_region(
                (0, 0),
                tissue_level,
                (self.level_dimensions[tissue_level][1], self.level_dimensions[tissue_level][0]),
                wsi_ext
            )

            # downsampling factor if image is largest dimension of the image is greater than 5000 at given tissue level
            # to reduce tissue segmentation time
            proc_scale = 1 / np.ceil(np.max(ds_img.shape) / 5000)

            self.tissue = proc_utils.get_tissue_mask(ds_img, proc_scale)

        # Coordinate info for tile processing
        self.tile_coords()
        logger.info(f"Number of tiles:{len(self.tile_info)-1}")
        #TODO: revise to use named logging Levels
        if int(logger._logger.level) < 40:
            np.save('%s/%s/%s.npy' % (self.output_dir, self.basename, "tile_info"),
                     self.tile_info
                    )

        # Run inference tile by tile - if self.tiss_seg == True, only process tissue regions
        mask_list_all = []
        type_list_all = []
        cent_list_all = []

        for idx_tile in range(len(self.tile_info)):
            start_tile_time = time.time()
            self.extract_patches(idx_tile)


            mask_list, type_list, cent_list = self.run_inference(idx_tile, wsi_ext)

            # add tile predictions to overall prediction list
            mask_list_all.extend(mask_list)
            type_list_all.extend(type_list)
            cent_list_all.extend(cent_list)

            # uncomment below if you want to save results per tile

            np.savez('%s/%s/%s_%s.npz' % (
            self.output_dir, self.basename, self.basename, str(idx_tile)),
            mask=mask_list, type=type_list, centroid=cent_list
            )
            end_tile_time = time.time()
            logger.info(f"inferWSI- Processed tile#{idx_tile+1}. Time: {time_it(start_tile_time, end_tile_time)} secs")

        if self.ds_factor != 1:
            cent_list = self.ds_factor * np.array(cent_list)
            cent_list = cent_list.tolist()
        np.savez('%s/%s/%s.npz' % (
            self.output_dir, self.basename, self.basename),
            mask=mask_list_all, type=type_list_all, centroid=cent_list_all
            )


    def process_all_wsi(self):
        """
        Process each WSI one at a time and save results as npz file
        """

        if os.path.isdir(self.output_dir) == False:
            rm_n_mkdir(self.output_dir)

        for filename in self.file_list:
            filename = os.path.basename(filename)
            self.basename = os.path.splitext(filename)[0]
            # this will overwrite file is it was processed previously
            rm_n_mkdir(self.output_dir + '/' + self.basename)
            start_time_wsi = time.time()
            self.process_wsi(filename)
            end_time_wsi = time.time()
            logger.info(f"inferWSI- {self.basename} FINISHED. Time: {time_it(start_time_wsi, end_time_wsi)} secs")

        logger.shutdown()
