import glob
import hashlib
import json
import os
import sys
import os.path
import time
import argparse
import shlex
import re

from collections import Counter


def valid_checkpoints():
    ensemble_names = set()
    for logdir in glob.glob('logdirs/*', recursive=True) + glob.glob('logdirs-overfit/*', recursive=True):
        if "logdirs/baseline_model" in logdir:
            continue
        if "finetuned-rl-1e-5," in logdir:
            continue
        if "finetuned-rl-1e-5-use-heldout," in logdir:
            continue
        if "vanilla-real-nearai-finetuned-1e-5," in logdir:
            continue
        if "aggregate-with-io-real-nearai-finetuned-1e-5," in logdir:
            continue
        if "finetuned-agg" in logdir:
            continue
        short_name = logdir.split("/")[-1].split(",")[0]
        ensemble_regex = "^(vanilla|aggregate-with-io)-[123]($|,|-)"
        if re.match(ensemble_regex, short_name):
            ensemble_names.add(re.sub(ensemble_regex, "\\1-#\\2", short_name))
            if short_name not in {"vanilla-%s" % b for b in "1"}:
                continue
        elif "123" in short_name:
            continue

        numbers = get_checkpoint_numbers(logdir)

        yield from valid_checkpoints_for_logdir(logdir, numbers)

    for name in ensemble_names:
        if "old" not in name:
            continue
        logdir = "logdirs/" + name + ",*"
        actual_logdirs = [x for number in "123" for x in glob.glob(logdir.replace("#", number))]
        checkpoints_per = [set(get_checkpoint_numbers(logdir)) for logdir in actual_logdirs]
        common_checkpoints = checkpoints_per[0]
        for checkpoints in checkpoints_per[1:]:
            common_checkpoints &= checkpoints
        common_checkpoints = sorted(common_checkpoints)
        yield from valid_checkpoints_for_logdir(logdir, common_checkpoints)


def valid_checkpoints_for_logdir(logdir, numbers):
    if len([x for x in numbers if is_multiple(x, 25000)]) >= 10:
        numbers = [x for x in numbers if is_multiple(x, 25000)]
    else:
        numbers = [x for x in numbers if is_multiple(x, 10000)]
    numbers.sort(reverse=True)

    for chunk in chunked(enumerate(numbers), len(numbers) // 5):
        for idx, ckpt_number in chunk:
            yield (logdir, ckpt_number), (idx, len(numbers), chunk), "overfit" in logdir.split("/")[0]


def chunked(lst, count):
    count = max(count, 1)
    out = []
    for x in lst:
        out.append(x)
        if len(out) == count:
            yield out
            out = []
    if out:
        yield out


def get_checkpoint_numbers(logdir):
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
        if not is_multiple(ckpt_number, 5000) or ckpt_number <= 1000:
            continue
        numbers.append(ckpt_number)
    return numbers


def is_multiple(ckpt_number, interval):
    return (ckpt_number - 100) % interval == 0 or (ckpt_number - 1000) % interval == 0


def valid_modes_and_params():
    for mode in 'train', 'eval', 'real', 'realtrain':
        if mode in {'train', 'eval'}:
            params = '1', '0,1', '0,0,1', '0,0,0,1', '0,0,0,0,1'
            for param in params:
                yield (mode, param, param), 'always', ''
        elif mode in {'real', 'realtrain'}:
            for model in 'nearai', 'nearai32', "egnps64":
                yield (mode, (model, ''), model), 'always', ''
                for limit in 1, 5, 10, 25, 50, 100:
                    for strategy in 'greedy', 'best_first':
                        if strategy == 'greedy' and limit == 1:
                            continue # greedy and best first are exactly the same for 1 step
                        when = 'always' if strategy == 'best_first' else 'sometimes'
                        for extra in '', '--iterative-search-start-with-beams':
                            if extra != '':
#                                 if strategy != 'best_first':
#                                     continue
                                if model == 'nearai':
                                    continue
                            else:
                                if model == "nearai32":
                                    continue
                            render_extra = '' if extra == '' else ',,start-with-beams'
                            for overfit_model, overfit_cmd in overfit_models_to_use():
                                if overfit_model == '':
                                    render_extra_with = render_extra
                                    extra_with = extra
                                else:
                                    if strategy != 'best_first' or limit != 25 or model == "nearai":
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


def compute_batch_size(is_vanilla, count, max_gpu_memory):
    # gpu_usage_in_mb = m * count * batch_size + b
    if is_vanilla:
        m = (2703 - 1721) / (48 - 24)
        b = 1721 - 24 * m
    else:
        m = (5585 - 4057) / (36 - 24)
        b = 4057 - 24 * m + 1000 # just add a bit extra in case
    # 0.9 for wiggle room
    batch_size = (max_gpu_memory * 0.9 - b * count) / m
    return int(batch_size)


def main(args):
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    elif args.cpu:
        batch_size = 16
    else:
        assert args.max_gpu_memory
        batch_size = None
    by_priority = []
    planned = set()
    for (logdir, ckpt_number), (index_last, num_checkpoints, chunk), is_overfit_model in valid_checkpoints():
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

            output_path_pattern = "{logdir}/report-dev-m{dist}-%s-{mode}.jsonl".format(
                logdir=logdir.replace("#", "123").replace("*", "dovetail"),
                dist=render_param, mode=mode)
            output_path = output_path_pattern % ckpt_number
            if already_executed(output_path):
                continue

            if batch_size is None:
                used_batch_size = compute_batch_size(logdir.split("/")[-1].startswith("vanilla"), 3 if '#' in logdir else 1, args.max_gpu_memory)
            else:
                used_batch_size = batch_size
            command = ('python -u program_synthesis/eval.py --model_type {model_type} --evaluate-on-all '
                       '--dataset karel --max_beam_trees 64 --step {step} '
                       '--model_dir {logdir} '
                       '--batch_size {batch_size} '
                       '--report-path {output_path} '
                       '--hide-example-info ').format(
                model_type='karel-lgrl-overfit' if is_overfit_model else 'karel-lgrl-ref',
                batch_size=used_batch_size,
                step=ckpt_number, logdir=shlex.quote(logdir),
                output_path=output_path
            )

            command += extra_args + ' '
            model_data = search_param = ''
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

            if '#' in logdir:
                command += ' --ensemble-parameters 1 2 3 --ensemble-mode dovetail'

            command += ' '

            if args.cpu:
                command += '--restore-map-to-cpu --no-cuda '

            if mode == "real" and "many-mutations" in logdir:
                continue
            if is_overfit_model:
                continue # given up on these
            elif index_last == 0 and is_multiple(ckpt_number, 25000):
                priority = 11
            elif not any(already_executed(output_path_pattern % other_step) or output_path_pattern % other_step in planned for _, other_step in chunk):
                if "finetuned" in logdir:
                    priority = 16
                else:
                    priority = 21
            else:
                priority = 100
            if when == 'always':
                priority -= 1
            if mode == 'eval':
                # really fast, might as well do it immediately
                priority -= 17
            if "overfit=" in command:
                continue # given up on these
            if '#' in logdir:
                # don't deprioritize these too much we want to see the results
                priority += 20
            if search_param and search_param[1] >= 50:
                priority += 30
            planned.add(output_path)
            by_priority.append((priority, command))
    by_priority.sort()
    print_classes(by_priority)
    by_priority = filter_hash(by_priority, args.bin_count, args.bins, args.seed)
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

def hashnum(s, seed):
    mat = re.match(r".*--report-path\s*(\S*).*", s)
    assert mat
    s = mat.group(1)
    m = hashlib.sha256()
    m.update(str((s, seed)).encode('utf-8'))
    return int(m.hexdigest(), 16)

def filter_hash(by_priority, bin_count, bins, seed):
    if bin_count is None:
        assert bins is None
        return by_priority
    bins = [int(x) for x in bins.split(",")]
    assert all(0 <= x < bin_count for x in bins)
    by_priority = [(x, y) for x, y in by_priority if hashnum(y, seed) % bin_count in bins]
    print("after hash filtering", file=sys.stderr)
    print_classes(by_priority)
    return by_priority

parser = argparse.ArgumentParser()
parser.add_argument('--max-commands', type=int, default=None)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--max-gpu-memory', default=None, type=int, help="in MB")
parser.add_argument('--batch-size')
parser.add_argument('--priority', type=int, default=None)
parser.add_argument('--bin-count', type=int, default=None)
parser.add_argument('--bins', type=str, default=None)
parser.add_argument('--seed', type=str, default='')

main(parser.parse_args())
