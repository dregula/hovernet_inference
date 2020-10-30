import glob
# import importlib
import math
import os
import time
from collections import deque

import cv2
import numpy as np
# from tensorpack import PredictConfig, get_model_loader, OfflinePredictor

from hover.misc.utils import rm_n_mkdir
from hover.misc.viz_utils import visualize_instances
from hover.postproc import process_utils as proc_utils

from shared_infer_methods import SharedInferMethods

class InferROI(SharedInferMethods):
    def __init__(self,):
        self.nr_types = 6  # denotes number of classes (including BG) for nuclear type classification
        self.input_shape = [256, 256]
        self.mask_shape = [164, 164]
        self.input_norm = True  # normalize RGB to 0-1 range

        # for inference during evalutation mode i.e run by infer.py
        self.input_tensor_names = ['images']
        self.output_tensor_names = ['predmap-coded']

     # def get_model(self):
     #    model_constructor = importlib.import_module('hover.model.graph')
     #    model_constructor = model_constructor.Model_NP_HV
     #    return model_constructor # NOTE return alias, not object

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x        : input image to be segmented. It will be split into patches
                       to run the prediction upon before being assembled back
            predictor: A predictor built from a given config.
        """

        step_size = self.mask_shape
        msk_size = self.mask_shape
        win_size = self.input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        sub_patches = []
        # generating subpatches from original
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0],
                        col:col+win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.batch_size:
            mini_batch  = sub_patches[:self.batch_size]
            sub_patches = sub_patches[self.batch_size:]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        # Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        return pred_map

    ####
    # def load_model(self):
    #     print('Loading Model...')
    #     model_path = self.model_path
    #     model_constructor = self.get_model()
    #     pred_config = PredictConfig(
    #         model        = model_constructor(self.nr_types, self.input_shape, self.mask_shape, self.input_norm),
    #         session_init = get_model_loader(model_path),
    #         input_names  = self.input_tensor_names,
    #         output_names = self.output_tensor_names)
    #     self.predictor = OfflinePredictor(pred_config)

    def process(self):
        """
        Process image files within a directory.
        For each image, the function will:
        1) Load the image
        2) Extract patches the entire image
        3) Run inference
        4) Return output numpy file and overlay
        """

        # save_dir = self.output_dir
        # rm_n_mkdir(self.output_dir)

        # file_list = glob.glob('%s/*' %self.input_dir)
        # file_list.sort() # ensure same order

        for filename in self.file_list:
            filename = os.path.basename(filename)
            basename = os.path.splitext(filename)[0]
            # print(self.input_dir, basename, end=' ', flush=True)
            # print(filename)

            ###
            start_time_total = time.time()
            img = cv2.imread(self.input_dir + '/' + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ###
            pred_map = self.__gen_prediction(img, self.predictor)

            pred_inst, pred_type = proc_utils.process_instance(pred_map, nr_types=self.nr_types)

            overlaid_output = visualize_instances(img, pred_inst, pred_type)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

            # combine instance and type arrays for saving
            pred_inst = np.expand_dims(pred_inst, -1)
            pred_type = np.expand_dims(pred_type, -1)
            pred = np.dstack([pred_inst, pred_type])

            cv2.imwrite('%s/%s.png' % (self.output_dir, basename), overlaid_output)
            np.save('%s/%s.npy' % (self.output_dir, basename), pred)
            end_time_total = time.time()
            logger.info(f"inferROI- {self.basename} FINISHED. Time: {time_it(start_time_total, end_time_total)} secs")

        logger.shutdown()
