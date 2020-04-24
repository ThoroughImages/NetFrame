import argparse
import importlib
import pprint
import os

import tensorflow as tf

from utils.config import Config

parser = argparse.ArgumentParser(description='NetFrame, a pipeline for model development')

parser.add_argument('--version', type=str, default='',
                    help='version of exp, eg: $VERSION_$SUBVERSION')
parser.add_argument('--config_file', type=str, default='./config/config.yaml',
                    help='Configuration file of yaml format')
parser.add_argument('--gpu', type=str, default='',
                    help='gpu indices, eg: 0,1')
parser.add_argument('--mode', type=str, default='train',
                    help='train, test or export')
parser.add_argument('--debug', type=bool, default=False,
                    help='True or False')
parser.add_argument('--restore_iters', type=int, default=0,
                    help='iters to export')
parser.add_argument('--model_path', type=str, default='',
                    help='model_path')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
args = parser.parse_args()


def initialize():
    cfg = Config(args.version, args.config_file, args.debug)
    # Update parameters of config object through command line.
    new_args = {k: v for k, v in args.__dict__.items() if parser.get_default(k) != v}
    cfg.update(new_args)

    if type(cfg.gpu) is str:
        cfg.gpu = cfg.gpu.split(',')

    pp = pprint.PrettyPrinter()
    print('--- Arguments ---')
    pp.pprint(args.__dict__)
    print('--- Configuration ---')
    pp.pprint(cfg.__dict__)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cfg.gpu))
    # Import module dynamicly
    project_data = importlib.import_module('project.' + cfg.data_module)
    project_model = importlib.import_module('project.' + cfg.model_module)

    data = project_data.Data(cfg, args.mode)
    model = project_model.Model(cfg, data, args.mode)

    return args, cfg, data, model


def main():
    args, cfg, data, model = initialize()

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if cfg.restore_iters > 0 and args.mode != 'export':
            # pass
            sess.run(tf.global_variables_initializer())
            model.load(sess, '{}/model-{}.ckpt'.format(model.snapshots_dir, cfg.restore_iters))
        else:
            sess.run(tf.global_variables_initializer())
        # Execute
        if args.mode == 'train':
            model.train(sess)
        elif args.mode == 'test':
            model.test(sess)
        else:
            raise ValueError('Choose mode among train, test and export.')


if __name__ == '__main__':
    main()

