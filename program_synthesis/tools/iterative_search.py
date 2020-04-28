from abc import ABC, abstractmethod

from collections import defaultdict
from itertools import count

from datasets.executor import evaluate_code
from datasets.karel import mutation

from models.base import InferenceResult


class IterativeSearch:
    def __init__(self, original_inference, init_strategy, executor, add_trace, batch_processor, start_with_beams,
                 time_limit):
        self.original_inference = original_inference
        self.init_strategy = init_strategy
        self.executor = executor
        self.add_trace = add_trace
        self.batch_processor = batch_processor
        # whether to start with the beams from the original model
        self.start_with_beams = start_with_beams
        self.time_limit = time_limit

    def __call__(self, batch):
        strategies = [self.init_strategy(item) for item in batch.orig_examples]
        done = [False] * len(batch.orig_examples)
        finalized_candidates = [[] for _ in range(len(batch.orig_examples))]
        attempts = [[] for _ in range(len(batch.orig_examples))]
        index_mapping = {i: i for i in range(len(batch.orig_examples))}  # mapping from indices in batch/strategies to indices in done
        num_inferences = 0
        for iteration_idx in count():
            if iteration_idx == 0 and self.start_with_beams:
                results = [example.ref_beams for example in batch.orig_examples]
                assert not any(x is None for x in results), "the original examples must contain a full list of reference beams"
            else:
                results = [result.info['candidates'] for result in self.original_inference(batch)]
                num_inferences += 1
            decisions = [strategy.decide(result, lambda code: self.test_results(code, example)) for
                         strategy, result, example in zip(strategies, results, batch.orig_examples)]
            new_index_mapping = {}
            new_strategies = []
            new_wrong_code = []
            new_batch = []
            for idx, decision in enumerate(decisions):
                if decision[0] == 'accept' or num_inferences == self.time_limit:
                    finalized_candidates[index_mapping[idx]].append(decision[1])
                    done[index_mapping[idx]] = True
                elif decision[0] == 'expand':
                    attempts[index_mapping[idx]].append(decision[1])
                    new_wrong_code.append(decision[1])
                    new_batch.append(batch.orig_examples[idx])
                    new_strategies.append(strategies[idx])
                    new_index_mapping[len(new_wrong_code) - 1] = index_mapping[idx]
                else:
                    raise ValueError(
                        "Invalid decision: {}. The first element must be either 'accept' or 'expand' but was {}".format(
                            decision, decision[0]))
            if all(done):
                break
            index_mapping = new_index_mapping
            strategies = new_strategies
            batch = self.update_wrong_code_and_pack(new_batch, new_wrong_code)

        assert all(len(candidates) == 1 for candidates in finalized_candidates)
        best_code = [candidates[0] for candidates in finalized_candidates]

        return [InferenceResult(code_sequence=code, info={'candidates' : [code], 'expanded' : [expanded]})
                   for code, expanded in zip(best_code, attempts)]

    def update_wrong_code_and_pack(self, new_examples, new_wrong_code):
        assert new_examples
        assert len(new_examples) == len(new_wrong_code)
        updated_examples = []
        for example, code in zip(new_examples, new_wrong_code):
            updated_examples.append(
                mutation.add_incorrect_code(example, tuple(code), self.add_trace, self.executor, check_ref_example=False))
        return self.batch_processor(updated_examples)

    def test_results(self, code, example):
        return evaluate_code(code, example.schema.args, example.input_tests, self.executor.execute)


class Strategy(ABC):
    @abstractmethod
    def decide(self, candidates, evaluate):
        pass

    @staticmethod
    def get(descriptor):
        if ":" not in descriptor:
            descriptor += ":"
        start, *rest = descriptor.split(":")
        kwargs = eval("dict({})".format(":".join(*rest)))
        return {
            'greedy': lambda: GreedyStrategy,
            'best_first': lambda: BestFirstSearch
        }[start](**kwargs)

def valid(considered_program, result):
    if not considered_program:
        return False
    if result['syntax-error'] > 0:
        return False
    return True


class GreedyStrategy(Strategy):
    def __init__(self, item):
        self.seen = set()
        del item  # no need

    def decide(self, candidates, evaluate):
        unseen = []
        for considered in candidates:
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if not valid(considered, res):
                continue
            if res['correct'] == res['total']:
                self.seen.add(considered)
                return 'accept', considered
            unseen.append((res['correct'], considered))
        if not unseen:
            self.seen.add(candidates[0])
            return 'accept', candidates[0]
        unseen.sort(reverse=True)
        self.seen.add(unseen[0][1])
        return 'expand', unseen[0][1]


class BestFirstSearch(Strategy):
    def __init__(self, item):
        self.seen = set()
        self.by_number_correct = defaultdict(list)

    def decide(self, candidates, evaluate):
        for considered in candidates:
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if not valid(considered, res):
                continue
            self.seen.add(considered)
            assert res['total'] == 5
            self.by_number_correct[res['correct']].append(considered)

        for n_correct in sorted(self.by_number_correct, reverse=True):
            if self.by_number_correct[n_correct]:
                decision = 'accept' if n_correct == 5 else 'expand'
                return decision, self.by_number_correct[n_correct].pop(0)

        return 'accept', tuple(candidates[0])
