
import fire
import glob
import os
import json

EVAL_COMMAND = "python -u program_synthesis/eval.py --model_type karel-lgrl-ref --dataset karel --evaluate-on-all --max_beam_trees {beam_size} --step {step} --model_dir {logdir} --batch_size {batch_size} --hide-example-info --karel-file-ref-{train_or_val} baseline/{dataset}-{segment}.json --run-predict --predict-path {predict_path} {maybe_eval_train}"

TRAIN_OUTPUT_DIRECTORY = "train-results"

def combine_prediction_results(results_per):
    """
    Gets all the correct beams for each example and collects them into a dictionary from guid to correct beams
    """
    combined = {}
    assert len({len(x) for x in results_per}) == 1
    for datums in zip(*results_per):
        [guid] = {datum['guid'] for datum in datums}
        valid_programs = []
        for model_idx, datum in enumerate(datums):
            valid_idxs = [idx for idx, val in enumerate(datum['beams_correct']) if val['total'] == val['correct']]
            valid_programs += [(model_idx, datum['beams'][idx]) for idx in valid_idxs]
        assert guid not in combined, guid
        combined[guid] = valid_programs
    return combined

def run_model(*, model_id, step, beam_size, batch_size, dataset, segment, just_print_commands):

    predict_path="{dir}/{segment},{model_id},{step},{beam_size}.json".format(
        dir=TRAIN_OUTPUT_DIRECTORY,
        model_id=model_id,
        segment=segment,
        step=step,
        beam_size=beam_size
    )
    if not os.path.exists(predict_path):
        [logdir] = glob.glob("logdirs/{model_id},*".format(model_id=model_id))

        try:
            os.mkdir(TRAIN_OUTPUT_DIRECTORY)
        except FileExistsError:
            pass

        command_to_run = EVAL_COMMAND.format(
            beam_size=beam_size,
            step=step,
            logdir=logdir,
            batch_size=batch_size,
            dataset=dataset,
            segment=segment,
            predict_path=predict_path,
            train_or_val="train" if segment.startswith("train") else "val",
            maybe_eval_train="--eval-train" if segment.startswith("train") else ""
        )
        print(command_to_run)
        if just_print_commands:
            return None
        os.system(command_to_run)

    with open(predict_path) as f:
        return json.load(f)

def get_combined_results(*, model_id_pattern, num_model, steps, segment, dataset, output_file, beam_size=64, batch_size=16, just_print_commands=False):
    assert not os.path.exists(output_file), "output file must not exist"
    if isinstance(steps, int):
        steps = [steps] * num_model
    results_per = [
        run_model(
            model_id = model_id_pattern.replace("#", str(1 + number)),
            step=step,
            beam_size=beam_size,
            batch_size=batch_size,
            dataset=dataset,
            segment=segment,
            just_print_commands=just_print_commands
        )
        for number, step in enumerate(steps)
    ]
    combined = combine_prediction_results(results_per)
    with open(output_file, "w") as f:
        json.dump(combined, f)

if __name__ == '__main__':
    fire.Fire(get_combined_results)
