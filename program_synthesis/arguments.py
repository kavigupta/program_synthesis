import argparse
import time


def get_arg_parser(title, mode):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--model_type', type=str, default='karel-lgrl-ref')
    parser.add_argument('--model_dir', type=str, default='models/%d' % int(time.time()))
    parser.add_argument('--dataset', type=str, default='karel')
    parser.add_argument('--dataset_max_size', type=int, default=0)
    parser.add_argument('--dataset_max_code_length', type=int, default=0)
    parser.add_argument('--dataset_filter_code_length', type=int, default=0)
    parser.add_argument('--dataset_bucket', action='store_true', default=False)
    parser.add_argument('--vocab_min_freq', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)#128
    parser.add_argument('--load-sync', action='store_true')

    parser.add_argument(
        '--pretrained', type=str, default='', 
        help='Use format "encoder:path/to/checkpoint,decoder:path/to/other/checkpoint"')

    if mode == 'train':
        train_group = parser.add_argument_group('train')
        train_group.add_argument('--save_every_n', type=int, default=100)
        train_group.add_argument('--keep_every_n', type=int, default=10000)#10000000
        train_group.add_argument('--debug_every_n', type=int, default=1000)#20
        train_group.add_argument('--eval_every_n', type=int, default=10000000)#1000
        train_group.add_argument('--eval_n_steps', type=int, default=50)
        train_group.add_argument('--log_interval', type=int, default=100)#20
        train_group.add_argument('--optimizer', type=str, default='sgd') #adam
        train_group.add_argument('--lr', type=float, default=1) #.001
        train_group.add_argument('--lr_decay_steps', type=int, default=100000)
        train_group.add_argument('--lr_decay_rate', type=float, default = 0.5)
        train_group.add_argument('--gradient-clip', type=float)
        train_group.add_argument('--n_warmup_steps', type=int, default=4000)
        train_group.add_argument('--num_epochs', type=int, default=50)
        train_group.add_argument('--num_units', type=int, default=100)
        train_group.add_argument('--num_placeholders', type=int, default=0)#100
        train_group.add_argument('--num-att-heads', type=int, default=8)
        train_group.add_argument('--bidirectional', action='store_true', default=False)
        train_group.add_argument('--read-code', dest='read_code', action='store_true', default=False)
        train_group.add_argument('--read-text', dest='read_text', action='store_true', default=True)
        train_group.add_argument('--skip-text', dest='read_text', action='store_false')
        train_group.add_argument('--read-io', dest='read_io', action='store_true', default=False)
        train_group.add_argument('--skip-io', dest='read_io', action='store_false')
        train_group.add_argument('--io-count', type=int, default=3)

        # REINFORCE.
        train_group.add_argument('--reinforce', action='store_true', default=False)
        train_group.add_argument('--max_rollout_length', type=int, default=1)
        train_group.add_argument('--replay_buffer_size', type=int, default=16384)
        train_group.add_argument('--erase_factor', type=float, default=0.01)
        train_group.add_argument('--num_episodes', type=int, default=10)
        train_group.add_argument('--num_training_steps', type=int, default=10)
        train_group.add_argument('--update_actor_epoch', type=int, default=10)

        train_group.add_argument(
            '--reinforce-step', type=int, default=0,
            help='Step after which start to use reinforce')
        train_group.add_argument(
            '--reinforce-beam-size', type=int, default=100,
            help='Size of beam to evalutate when using reinforce'
        )

        # REFINE.
        train_group.add_argument('--refine', action='store_true', default=False)
        train_group.add_argument(
            '--refine-beam', type=int, default=10,
            help='Beam size to use while decoding to generate candidate code for training the refinement model.')
        train_group.add_argument(
            '--refine-samples', type=int, default=100000,
            help='# Number of refinement training samples to keep in the buffer.')
        train_group.add_argument('--refine-min-items', type=int, default=128)
        train_group.add_argument(
            '--refine-frac', type=float, default=0.5, 
            help='Fraction of time we should sample refinement data for training.')
        train_group.add_argument(
            '--refine-warmup-steps', type=int, default=1000,
            help='Number of steps we should train before we sample any code to generate the refinement dataset.')
        train_group.add_argument(
            '--refine-sample-frac', type=float, default=0.1,
            help='Fraction of batches for which we should sample code to add to the refinement data for training.')

        train_group.add_argument('--karel-trace-enc', default='none') #lstm
        train_group.add_argument('--karel-code-enc', default='default')
        train_group.add_argument('--karel-refine-dec', default='edit') #default
        train_group.add_argument('--karel-trace-usage', default='memory')
        train_group.add_argument('--karel-code-usage', default='memory')

    elif mode == 'eval':
        eval_group = parser.add_argument_group('eval')
        eval_group.add_argument('--tag', type=str, default='')
        eval_group.add_argument('--example-id', type=int, default=None)
        eval_group.add_argument('--step', type=int, default=None)
        eval_group.add_argument('--refine-iters', type=int, default=1)
        eval_group.add_argument('--eval-train', action='store_true', default=False)
        eval_group.add_argument('--hide-example-info', action='store_true', default=False)
        eval_group.add_argument('--report-path')
        eval_group.add_argument('--eval-final', action='store_true')

    infer_group = parser.add_argument_group('infer')
    infer_group.add_argument('--max_decoder_length', type=int, default=100)
    infer_group.add_argument('--max_beam_trees', type=int, default=1)#100
    infer_group.add_argument('--max_beam_iter', type=int, default=1000)
    infer_group.add_argument('--max_eval_trials', type=int)
    infer_group.add_argument('--min_prob_threshold', type=float, default=1e-5)
    infer_group.add_argument('--search-bfs', action='store_true', default=True)
    infer_group.add_argument('--karel-mutate-ref', action='store_true', default=True)
    infer_group.add_argument('--karel-mutate-n-dist', default='1,2,3')

    runtime_group = parser.add_argument_group('runtime')
    runtime_group.add_argument(
        '--restore-map-to-cpu', action='store_true', default=False)

    return parser


def backport_default_args(args):
    """Backport default args."""
    backport = {
        "restore_map_to_cpu": False,
        "keep_every_n": 10000000,
        "read_text": True,
        "read_io": False,
        "io_count": 3,
        "refine": False,
        "read_code": False,
        "optimizer": "adam",
        "dataset_filter_code_length": 0,
        "karel_trace_usage": "memory",
        "karel_code_usage": "memory",
        "karel_refine_dec": "edit",
    }
    for key, value in backport.items():
        if not hasattr(args, key):
            setattr(args, key, value)
