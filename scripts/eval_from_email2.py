import glob
import json
import os
import os.path
import time
import argparse


def valid_checkpoints():
    for logdir in glob.glob('logdirs/*', recursive=True):
        if "logdirs/baseline_model" in logdir:
            continue
        numbers = []
        for ckpt in sorted(glob.glob(logdir + '/checkpoint-????????')):
            try:
                time_delta = time.time() - os.path.getmtime(ckpt)
            except FileNotFoundError:
                continue
            if time_delta < 5 * 60:
                # ignore really recent checkpoints (last 5 minutes)
                # these are deleted by the time an evaluation can be run :(
                continue
            ckpt_number = int(ckpt[-8:])
            cm100 = ckpt_number - 100
            if (cm100 % 50000 != 0 and cm100 % 10000 == 0) or ckpt_number < 1000:
                continue
            numbers.append(ckpt_number)
        numbers.sort(reverse=True)
        for idx, ckpt_number in enumerate(numbers):
            yield (logdir, ckpt_number), (idx, len(numbers))


def valid_modes_and_params():
    for mode in 'train', 'eval', 'real':
        if mode in {'train', 'eval'}:
            params = '1', '0,1', '0,0,1'
            for param in params:
                yield (mode, param, param), 'always'
        elif mode == 'real':
            for model in 'nearai', 'nearai32':
                yield (mode, (model, ''), model), 'always'
                for limit in 1, 5, 10, 25:
                    for strategy in 'greedy', 'best_first':
                        when = 'always' if limit <= 10 and strategy == 'greedy' else 'sometimes'
                        yield (mode, (model, (strategy, limit)), "{},,{},,{}".format(model, strategy, limit)), when
        else:
            assert False


def already_executed(output_path):
    done = False
    if os.path.exists(output_path):
        with open(output_path) as f:
            res = json.loads(next(f))
        if res.get('done', res.get('total', 0) >= 2500):
            done = True
    return done


def main(args):
    low_priority = []
    for (logdir, ckpt_number), (index_last, num_checkpoints) in valid_checkpoints():
        for (mode, param, render_param), when in valid_modes_and_params():
            output_path = "{logdir}/report-dev-m{dist}-{step}-{mode}.jsonl".format(logdir=logdir,
                                                                                   dist=render_param,
                                                                                   step=ckpt_number, mode=mode)
            if already_executed(output_path):
                continue

            command = ('python -u program_synthesis/eval.py --model_type karel-lgrl-ref --evaluate-on-all '
                       '--dataset karel --max_beam_trees 64 --step {step} '
                       '--model_dir {logdir} '
                       '--batch_size {batch_size} '
                       '--report-path {output_path} '
                       '--hide-example-info ').format(
                batch_size=16 if args.cpu else 64,
                step=ckpt_number, logdir=logdir,
                output_path=output_path
            )

            if mode == 'real':
                model_data, search_param = param
                command += '--karel-file-ref-val baseline/{}-val.json'.format(model_data)
                if search_param != '':
                    command += ' --iterative-search {} --iterative-search-step-limit {}'.format(*search_param)
            else:
                command += '--karel-mutate-ref --karel-mutate-n-dist {dist} '.format(dist=param)

            command += ' '
            if mode == 'train':
                command += '--eval-train --limit 2500'

            command += ' '

            if args.cpu:
                command += '--restore-map-to-cpu --no-cuda '

            if num_checkpoints < 5 or index_last % (num_checkpoints // 5) == 0:
#                 import sys
#                 print(index_last, num_checkpoints, file=sys.stderr)
                print(command)
            else:
                low_priority.append(command)
    for command in low_priority[:args.num_low_priority]:
        print(command)


parser = argparse.ArgumentParser()
parser.add_argument('--num-low-priority', type=int, default=0)
parser.add_argument('--cpu', action='store_true')

main(parser.parse_args())
