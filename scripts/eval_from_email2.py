import glob
import json
import os
import os.path
import time


def valid_checkpoints():
    for logdir in glob.glob('logdirs/**/*', recursive=True):
        if "baseline_model" in logdir:
            continue
        for ckpt in sorted(glob.glob(logdir + '/checkpoint-????????')):
            time_delta = time.time() - os.path.getmtime(ckpt)
            if time_delta < 5 * 60:
                # ignore really recent checkpoints (last 5 minutes)
                # these are deleted by the time an evaluation can be run :(
                continue
            ckpt_number = int(ckpt[-8:])
            cm100 = ckpt_number - 100
            if (cm100 % 50000 != 0 and cm100 % 10000 == 0) or ckpt_number < 1000:
                continue
            yield logdir, ckpt_number


def valid_modes_and_params():
    for mode in 'train', 'eval', 'real':
        if mode in {'train', 'eval'}:
            params = '1', '0,1', '0,0,1'
            for param in params:
                yield mode, param, param
        elif mode == 'real':
            yield mode, '', ''
            for limit in 1, 5, 10:
                yield mode, ('greedy', limit), "greedy,,{}".format(limit)
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


def main():
    for logdir, ckpt_number in valid_checkpoints():
        for mode, param, render_param in valid_modes_and_params():
            output_path = "{logdir}/report-dev-m{dist}-{step}-{mode}.jsonl".format(logdir=logdir,
                                                                                   dist=render_param,
                                                                                   step=ckpt_number, mode=mode)
            if already_executed(output_path):
                continue

            command = ('python -u program_synthesis/eval.py --model_type karel-lgrl-ref '
                       '--dataset karel --max_beam_trees 64 --step {step} '
                       '--model_dir {logdir} '
                       '--batch_size 64 '
                       '--report-path {output_path} '
                       '--hide-example-info ').format(
                step=ckpt_number, logdir=logdir,
                output_path=output_path
            )

            if mode == 'real':
                command += '--karel-file-ref-val ../nearai/logdirs/baseline_model/on-val.json'
                if param != '':
                    command += ' --iterative-search {} --iterative-search-step-limit {}'.format(*param)
            else:
                command += '--karel-mutate-ref --karel-mutate-n-dist {dist} '.format(dist=param)

            command += ' '
            if mode == 'train':
                command += '--eval-train --limit 2500'

            command += ' '

            print(command)


main()
