import os, sys
import argparse
import logging
from optimizer.lr_scheduler import LR_SCHEDULER_REGISTRY
from utils.utils import Timer

# Tensorboard configuration
from datetime import date, datetime
from tensorboardX import SummaryWriter


def parse_args():
    from optimizer.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
    from optimizer.adam_optimizer import AdamOptimizer


    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluation'], default='train', 
                        help='train or evaluate the model')
    parser.add_argument('--dataset', choices=['Activitynet', 'Tacos', 'Charades', 'Didemo', 'Youcook2', 'Msrvtt', 'Tvr'], default='ActivityNet',
                        help='dataset for training and evaluation')
    parser.add_argument('--model', type=str, default=None, help='select the model for experimental study e.g., ianet, cmin, csmgan..')
    parser.add_argument('--cfg', type=str, default=None, help='configuration for model params')
    parser.add_argument('--set-cgs', dest='set_cfgs',  help='Set config keys. Key value sequence seperate by whitespace.'
                        'e.g. [key] [value] [key] [value]\n This has higher priority'
                        'than cfg file but lower than other args. (You can only overwrite'
                        'arguments that have alerady been defined in config file.)',
                        default=[], nargs='+')
    parser.add_argument('--train-data', type=str,
                        default=None,
                        help='')
    parser.add_argument('--val-data', type=str, default=None,
                        help='')
    parser.add_argument('--test-data', type=str, default=None,
                        help='')
    parser.add_argument('--word2vec-path', type=str, default='glove_model.bin',
                        help='')
    parser.add_argument('--feature-path', type=str, default='data/activity-c3d',
                        help='')
    parser.add_argument('--model-saved-path', type=str, default='../results/',
                        help='')
    parser.add_argument('--model-load-path', type=str, default='',
                        help='')
    parser.add_argument('--display-n-batches', type=int, default=50,
                        help='')
    parser.add_argument('--max-num-epochs', type=int, default=20,
                        help='')
    parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                        help='weight decay')
    parser.add_argument('--lr-scheduler', default='inverse_sqrt',
                        choices=LR_SCHEDULER_REGISTRY.keys(),
                        help='Learning Rate Scheduler')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')

    InverseSquareRootSchedule.add_args(parser)
    AdamOptimizer.add_args(parser)

    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None :
        from utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' %k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)


    return args


def build_logfile(args) :
    log_filename = args.model_saved_path + '/'
    log_filename += args.mode + '_'
    log_filename += time
    log_filename += args.dataset
    log_filename += ".log"

    return log_filename

def main(args, writer):
    print(args)
    from runners.runner_final import Runner
    runner = Runner(args, writer)
    if args.mode == 'train' :
        runner.train() 
    elif args.mode == 'evaluation' :
        # runner.eval_new() # recall 1, 5, 10
        # runner.eval(epoch=0)
        runner.eval_save()


if __name__ == '__main__':
    args = parse_args()

    time_format = "%Y_%m_%dT%H_%M_%S"
 
    time = Timer().get_time_hhmmss(None, format=time_format)
    args.model_saved_path = os.path.join(args.model_saved_path, args.model, args.dataset, time)
    
    writer = SummaryWriter(args.model_saved_path)

    if not os.path.exists(args.model_saved_path):
        os.makedirs(args.model_saved_path)

    log_filename = build_logfile(args)

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename),
                                logging.StreamHandler(sys.stdout)]
    )
    if args.model == 'tdtan' :
        from modules.tdtan import model_config as config
        config.update_config(args.cfg)
        
    elif args.model == 'ranet' :
        from modules.ranet import model_config as config 
        config.update_config(args.cfg)

    elif args.model == 'tmlga' :
        from modules.tmlga.config import cfg
        experiment_name = args.model_saved_path
        log_directory = os.path.join(experiment_name, "logs/")
        vis_directory = os.path.join(experiment_name, "visualization")
        
        if not (os.path.exists(log_directory) and os.path.exists(vis_directory)) :
            os.makedirs(log_directory)
            os.makedirs(vis_directory)

        cfg.merge_from_list(['EXPERIMENT_NAME', experiment_name, 'LOG_DIRECTORY', log_directory, "VISUALIZATION_DIRECTORY", vis_directory])
        cfg.merge_from_file(args.cfg)
    
    main(args, writer)
