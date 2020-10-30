# shared_infer_methods.py

import glob
import os
import importlib

from hover.misc.utils import rm_n_mkdir

from tensorpack import PredictConfig, get_model_loader, OfflinePredictor, logger


class SharedInferMethods:
    ####
    def load_model(self):
        print('Loading Model...')
        model_path = self.model_path
        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(self.nr_types, self.input_shape, self.mask_shape, self.input_norm),
            session_init = get_model_loader(model_path),
            input_names  = self.input_tensor_names,
            output_names = self.output_tensor_names)
        self.predictor = OfflinePredictor(pred_config)

    def load_params(self, args):
        """
        Load arguments
        """

        # Tile Size
        self.tile_size = int(args['--tile_size'])
        # Paths
        self.model_path  = args['--model']
        # get absolute path for input directory - otherwise may give error in JP2Image.m
        self.input_dir = os.path.abspath(args['--input_dir'])
        self.output_dir = args['--output_dir']
        rm_n_mkdir(self.output_dir)
        self.logging_dir = args['--logging_dir']
        logging_dir = self.output_dir + '/' + self.logging_dir
        rm_n_mkdir(logging_dir)
        logger.set_logger_dir(logging_dir)

        self.logging_level = args['--logging_level']
        #TODO: this depends on tensorflow getting first crack at the logger (and adding the defailt std_out handler with INFO-level logging)
        logger._logger.handlers[0].setLevel(self.logging_level)
        logger._logger.setLevel(self.logging_level)

        # Processing
        self.batch_size = int(args['--batch_size'])
        # Below specific to WSI processing
        self.return_masks = args['--return_masks']

        self.tiss_lvl = 3 # default WSI level at which perform tissue segmentation
        print(f"'--tissue_level' provided:{args['--tissue_level']}")
        try:
            if args['--tissue_level'] and int(args['--tissue_level']) > 3:
                self.tiss_lvl = int(args['--tissue_level'])
        except:
            pass

    def get_model(self):
        model_constructor = importlib.import_module('hover.model.graph')
        model_constructor = model_constructor.Model_NP_HV
        return model_constructor # NOTE return alias, not object

    def load_filenames(self):
        """
        Get the list of all WSI files to process
        """
        self.file_list = glob.glob('%s/*' %self.input_dir)
        self.file_list.sort() # ensure same order
