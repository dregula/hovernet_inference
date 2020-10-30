"""run.

Usage:
  run.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--batch_size=<n>] [--input_dir=<path>] [--output_dir=<path>] [--logging_dir=<path>] [--logging_level=<level>] [--tile_size=<n>] [--return_masks] [--tissue_level=<n>]
  run.py (-h | --help)
  run.py --version

Options:
  -h --help            Show this string.
  --version            Show version.
  --gpu=<id>           GPU list.
  --mode=<mode>        Inference mode. 'roi' or 'wsi'.
  --model=<path>       Path to saved checkpoint.
  --batch_size=<n>     Batch size.
  --input_dir=<path>   Directory containing input images/WSIs.
  --output_dir=<path>  Directory where the output will be saved.
  --logging_dir=<path>     Subdirectory, under output/ where the logging will be saved.
  --logging_level=<level>  Minimum logging level to be logged. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
  --tile_size=<n>      Size of tiles (assumes square shape).
  --return_masks       Whether to return cropped nuclei masks
  --tissue_level=<n>   The image level in the WSI to use
"""


from docopt import docopt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import re


def validate_args(args_to_validate):
    # Raise exceptions for invalid / missing arguments
    if args_to_validate['--model'] is None:
        raise Exception('A model path must be supplied as an argument with --model.')
    if args_to_validate['--mode'] != 'roi' and args_to_validate['--mode'] != 'wsi':
        raise Exception('Mode not recognised. Use either "roi" or "wsi"')
    if args_to_validate['--input_dir'] is None:
        raise Exception('An input directory must be supplied as an argument with --input_dir.')
    if args_to_validate['--input_dir'] == args_to_validate['--output_dir']:
        raise Exception('Input and output directories should not be the same- otherwise input directory will be overwritten.')
    logging_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    logging_level = str(args_to_validate['--logging_level']).strip().upper()
    if re.compile('|'.join(logging_levels), re.IGNORECASE).search(logging_level):  # re.IGNORECASE is used to ignore case
        # TODO: will logger.level accept a bare string or does it require an enum, such as logging.ERROR
        args_to_validate['--logging_level'] = logging_level
    else:
        raise Exception(f"Unrecognized logging level:{args_to_validate['--logging_level']}")
    # Insert default values (docopts usually does this itself,
    # but since it is the highest-priority merge dictionary,
    # an command option will overwrite it's default over any json_config item
    if args_to_validate['--gpu'] is None:
        args_to_validate['--gpu'] = 0
    if args_to_validate['--mode'] is None:
        args_to_validate['--mode'] = "roi"
    if args_to_validate['--batch_size'] is None:
        args_to_validate['--batch_size'] = 25
    if args_to_validate['--output_dir'] is None:
        args_to_validate['--output_dir'] = "output/"
    if args_to_validate['--logging_dir'] is None:
        args_to_validate['--logging_dir'] = "output/logging/"
    if args_to_validate['--logging_level'] is None:
        args_to_validate['--logging_level'] = "CRITICAL"
    if args_to_validate['--tile_size'] is None:
        args_to_validate['--tile_size'] = 512

    return args_to_validate

# from https://github.com/docopt/docopt/blob/master/examples/config_file_example.py
def load_json_config():
    import json
    jd = open("./conf/config_main.json", "r")
    json_conf = json.load(jd)
    jd.close()
    return json_conf
    # # Pretend that we load the following JSON file:
    # source = '''
    #     {"--force": true,
    #      "--timeout": "10",
    #      "--baud": "9600"}
    # '''
    # return json.loads(source)


def load_ini_config():
    try:  # Python 2
        from ConfigParser import ConfigParser
        from StringIO import StringIO
    except ImportError:  # Python 3
        from configparser import ConfigParser
        from io import StringIO

    # By using `allow_no_value=True` we are allowed to
    # write `--force` instead of `--force=true` below.
    config = ConfigParser(allow_no_value=True)

    # # Pretend that we load the following INI file:
    # source = '''
    #     [default-arguments]
    #     --force
    #     --baud=19200
    #     <host>=localhost
    # '''
    #
    # # ConfigParser requires a file-like object and
    # # no leading whitespace.
    # config_file = StringIO('\n'.join(source.split()))
    # config.read_file(config_file)
#    config.read_file("./conf/config_main.ini")

    # ConfigParsers sets keys which have no value
    # (like `--force` above) to `None`. Thus we
    # need to substitute all `None` with `True`.
    # return dict((key, True if value is None else value)
    #             for key, value in config.items('default-arguments'))
# NOT IMPLEMENTED
    return None


def merge(dict_1, dict_2=None):
    """Merge two dictionaries.
    Values that evaluate to true take priority over falsy values.
    `dict_1` takes priority over `dict_2`.
    """
    if dict_1 is None: return None
    if dict_2 is None: return dict_1

    return dict((str(key), dict_1.get(key) or dict_2.get(key))
                for key in set(dict_2) | set(dict_1))


if __name__ == '__main__':
    json_config = load_json_config()
    # print(f"Json args:{json_config}")
    # ini_config = load_ini_config()
    arguments = docopt(__doc__, version='HoVer-Net Inference v1.0')
    # Arguments take priority over INI, INI takes priority over JSON:
    # print(f"merged json_args:{merge(json_config)}")
    args = merge(arguments, merge(json_config))   # ini_config,
    # print(f"final merged args:{args}")

    args = validate_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['--gpu'])

    # Import libraries for WSI processing
    if args['--mode'] == 'wsi':
        try:
            import matlab
            from matlab import engine
        except:
            pass

    if args['--mode'] == 'roi':
        from infer_roi import InferROI
        infer = InferROI()
        infer.load_params(args)
        infer.load_model()
        infer.load_filenames()
        infer.process()
    elif args['--mode'] == 'wsi': # currently saves results per tile
        from infer_wsi import InferWSI
        infer = InferWSI()
        infer.load_params(args)
        infer.load_model()
        infer.load_filenames()
        infer.process_all_wsi()

    exit("run.py completed!")
