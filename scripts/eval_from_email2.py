import glob, re, os, os.path, json, time

import sys

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
        for dist in ('1', '0,1', '0,0,1', ''):
            for on_train in True, False:
                if on_train and dist == '':
                    continue
                mode = 'train' if on_train else ('eval' if dist else 'real')
                extra_args = '--eval-train --limit 2500' if on_train else ''
                output_path = "{logdir}/report-dev-m{dist}-{step}-{mode}.jsonl".format(logdir=logdir, dist=dist, step=ckpt_number, mode=mode)
                if os.path.exists(output_path):
                    with open(output_path) as f:
                        res = json.loads(next(f))
                    if res.get('done', res.get('total', 0) >= 2500):
                        # print(output_path, file=sys.stderr)
                        continue
                if dist:
                    mutate_or_ref = '--karel-mutate-ref --karel-mutate-n-dist {dist} '.format(dist=dist)
                else:
                    mutate_or_ref = '--karel-file-ref-val ../nearai/logdirs/baseline_model/on-val.json'

                print(
                    ('python -u program_synthesis/eval.py --model_type karel-lgrl-ref '
                     '--dataset karel --max_beam_trees 64 --step {step} '
                     '{mutate_or_ref} '
                     '--model_dir {logdir} '
                     '--batch_size 64 '
                     '--report-path {output_path} '
                     '--hide-example-info '
                     '{extra_args} '
                     # '--restore-map-to-cpu'
                    ).format(
                        step=ckpt_number, logdir=logdir,
                        extra_args=extra_args,
                        output_path=output_path,
                        mutate_or_ref = mutate_or_ref
                    )
                )

