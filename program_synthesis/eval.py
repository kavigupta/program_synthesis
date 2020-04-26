import argparse
import functools
import sys
import os
import json

import torch

from tools import evaluation
from datasets import executor
import datasets
import tools

import models
import arguments

from tools.iterative_search import IterativeSearch, Strategy, TimeLimitStrategy


def evaluate(args):
    print("Evaluation:")
    print("\tModel type: %s\n\tModel path: %s" % (args.model_type, args.model_dir))
    tools.restore_args(args)
    arguments.backport_default_args(args)
    datasets.set_vocab(args)
    m = models.get_model(args)
    if args.eval_final:
        eval_dataset = datasets.get_eval_final_dataset(args, m)
    elif args.eval_train:
        eval_dataset, _ = datasets.get_dataset(args, m, eval_on_train=True)
    else:
        eval_dataset = datasets.get_eval_dataset(args, m)
    if m.last_step == 0:
        raise ValueError('Attempting to evaluate on untrained model')
    m.model.eval()
    current_executor = executor.get_executor(args)()
    if args.example_id is not None:
        eval_dataset.data = [eval_dataset.task[args.example_id]]

    inference = m.inference

    if args.iterative_search is not None:
        inference = IterativeSearch(inference, TimeLimitStrategy.limit(Strategy.get(args.iterative_search), args.iterative_search_step_limit), current_executor,
                                    args.karel_trace_enc != 'none', m.batch_processor(for_eval=True), start_with_beams=args.iterative_search_start_with_beams)
    if args.run_predict:
        evaluation.run_predict(eval_dataset, inference, current_executor.execute, args.predict_path, evaluate_on_all=args.evaluate_on_all)
    else:
        evaluation.run_eval(
            args.tag, eval_dataset, inference,
            current_executor.execute, not args.hide_example_info,
            args.report_path,
            limit=args.limit,
            evaluate_on_all=args.evaluate_on_all)


if __name__ == "__main__":
    parser = arguments.get_arg_parser('Evaluating Text2Code', 'eval')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.model_type or (not args.model_dir and args.model_type != 'search'):
        raise ValueError("Specify model_dir and model_type")
    if not args.tag:
        args.tag = args.model_type
    evaluate(args)
