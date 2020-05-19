import glob
import json
import os
import sys
import os.path
import time
import argparse

from collections import Counter


def valid_checkpoints():
    for logdir in glob.glob('logdirs/*', recursive=True) + glob.glob('logdirs-overfit/*', recursive=True):
        if "logdirs/baseline_model" in logdir:
            continue
        if "many-mutations" in logdir:
            continue
        if "finetuned-rl-1e-5," in logdir:
            continue
        if "finetuned-rl-1e-5-use-heldout," in logdir:
            continue
        short_name = logdir.split("/")[-1].split(",")[0]
        if "-" in short_name:
            *stem, number = short_name.split("-")
            if stem == ['vanilla'] or stem == ['aggregate', 'with', 'io']:
                if number in "123":
                    continue
        numbers = []
        for ckpt in sorted(glob.glob(logdir + '/checkpoint-????????')):
            try:
                time_delta = time.time() - os.path.getmtime(ckpt)
            except FileNotFoundError:
                continue
            if time_delta < 60 * 60:
                # ignore really recent checkpoints (last hour)
                # these are deleted by the time an evaluation can be run :(
                continue
            ckpt_number = int(ckpt[-8:])
            cm100 = ckpt_number - 100
            cm1000 = ckpt_number - 1000
            interval = 25000 if logdir.startswith("logdirs-overfit") else 10000
            if (cm100 % interval != 0 and cm1000 % interval != 0) or ckpt_number < 1000:
                continue
            numbers.append(ckpt_number)
        numbers.sort(reverse=True)
        for idx, ckpt_number in enumerate(numbers):
            yield (logdir, ckpt_number), (idx, len(numbers)), "overfit" in logdir.split("/")[0]


def valid_modes_and_params():
    for mode in 'train', 'eval', 'real', 'realtrain':
        if mode in {'train', 'eval'}:
            params = '1', '0,1', '0,0,1'
            for param in params:
                yield (mode, param, param), 'always', ''
        elif mode in {'real', 'realtrain'}:
            for model in 'nearai', 'nearai32':
                yield (mode, (model, ''), model), 'always', ''
                for limit in 1, 5, 10, 25:
                    for strategy in 'greedy', 'best_first':
                        if strategy == 'greedy' and limit == 1:
                            continue # greedy and best first are exactly the same for 1 step
                        when = 'always' if strategy == 'best_first' else 'sometimes'
                        for extra in '', '--iterative-search-start-with-beams':
                            if extra != '':
                                if strategy != 'best_first':
                                    continue
                                if model == 'nearai':
                                    continue
                            render_extra = '' if extra == '' else ',,start-with-beams'
                            for overfit_model, overfit_cmd in overfit_models_to_use():
                                if overfit_model == '':
                                    render_extra_with = render_extra
                                    extra_with = extra
                                else:
                                    if strategy != 'best_first' or limit != 25 or model != "nearai32":
                                        continue
                                    render_extra_with = render_extra + ',,overfit=' + overfit_model
                                    extra_with = extra + " " + overfit_cmd
                                yield (mode, (model, (strategy, limit)), "{},,{},,{}{}".format(model, strategy, limit, render_extra_with)), when, extra_with
        else:
            assert False

def overfit_models_to_use():
    models = [
        #(2000, "overfit-vanilla-slow-split"),
        (51_000, "overfit-awoi-all-beams-balanced-load-debugger"),
        #(2000, "overfit-aggregate-with-io-slow-split")
    ]
    yield '', ''
    for step, model in models:
        [logdir] = glob.glob("logdirs-overfit/{},*".format(model))
        param = 'dataset="karel",step={},model_dir="{}"'.format(step, logdir)
        cmd = "--iterative-search-use-overfit-model '{}'".format(param)
        yield model.replace("-", "_"), cmd


def already_executed(output_path):
    done = False
    if os.path.exists(output_path):
        with open(output_path) as f:
            try:
                res = json.loads(next(f))
            except StopIteration:
                return False
        if res.get('done', res.get('total', 0) >= 2500):
            done = True
    return done


def main(args):
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 16 if args.cpu else 64
    by_priority = []
    for (logdir, ckpt_number), (index_last, num_checkpoints), is_overfit_model in valid_checkpoints():
        for (mode, param, render_param), when, extra_args in valid_modes_and_params():
            if is_overfit_model:
                if mode != 'real' and mode != 'realtrain':
                    continue
                if param[1] != '':
                    continue
                if param[0] == 'nearai':
                    continue
            if mode == 'realtrain':
                if not is_overfit_model:
                    continue

            output_path = "{logdir}/report-dev-m{dist}-{step}-{mode}.jsonl".format(logdir=logdir,
                                                                                   dist=render_param,
                                                                                   step=ckpt_number, mode=mode)
            if already_executed(output_path):
                continue

            command = ('python -u program_synthesis/eval.py --model_type {model_type} --evaluate-on-all '
                       '--dataset karel --max_beam_trees 64 --step {step} '
                       '--model_dir {logdir} '
                       '--batch_size {batch_size} '
                       '--report-path {output_path} '
                       '--hide-example-info ').format(
                model_type='karel-lgrl-overfit' if is_overfit_model else 'karel-lgrl-ref',
                batch_size=batch_size,
                step=ckpt_number, logdir=logdir,
                output_path=output_path
            )

            command += extra_args + ' '

            if mode == 'real' or mode == 'realtrain':
                model_data, search_param = param
                command += '--karel-file-ref-val baseline/{}-val.json'.format(model_data)
                if mode == 'realtrain':
                    # 0.91 to avoid overlap with 0-0.9
                    command += ' --karel-file-ref-train baseline/{}-train-only-val-segment.json:start=0.91'.format(model_data)
                if search_param != '':
                    command += ' --iterative-search {} --iterative-search-step-limit {}'.format(*search_param)
            else:
                command += '--karel-mutate-ref --karel-mutate-n-dist {dist} '.format(dist=param)

            command += ' '
            if mode == 'train' or mode == 'realtrain':
                command += '--eval-train --limit 2500'

            command += ' '

            if args.cpu:
                command += '--restore-map-to-cpu --no-cuda '

            if is_overfit_model:
                priority = 1
            elif index_last == 0:
                priority = 11
            elif num_checkpoints < 5 or index_last % (num_checkpoints // 5) == 0:
                priority = 21
            else:
                priority = 100
            if when == 'always':
                priority -= 1
            if "overfit=" in command:
                priority -= 1
            by_priority.append((priority, command))
    by_priority.sort()
    print_classes(by_priority)
    assert args.max_commands is None or args.priority is None, "cannot specify both a maximal number of commands and a maximal priority"
    if args.max_commands is not None:
        by_priority = by_priority[:args.max_commands]
    elif args.priority is not None:
        by_priority = [(x, y) for x, y in by_priority if x <= args.priority]
    print_classes(by_priority)

    for _, command in by_priority:
        print(command)

def print_classes(by_priority):
    for clas, count in sorted(Counter(x for x, _ in by_priority).items()):
        print(clas, count, file=sys.stderr)
    print(file=sys.stderr)

parser = argparse.ArgumentParser()
parser.add_argument('--max-commands', type=int, default=None)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--batch-size')
parser.add_argument('--priority', type=int, default=None)

main(parser.parse_args())
