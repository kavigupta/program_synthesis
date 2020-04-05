from abc import ABC, abstractmethod

from datasets.executor import evaluate_code
from datasets.karel import mutation

from models.base import InferenceResult


class IterativeSearch:
    def __init__(self, original_inference, init_strategy, executor, add_trace, batch_processor):
        self.original_inference = original_inference
        self.init_strategy = init_strategy
        self.executor = executor
        self.add_trace = add_trace
        self.batch_processor = batch_processor

    def __call__(self, batch):
        strategies = [self.init_strategy(item) for item in batch.orig_examples]
        done = [False] * len(batch.orig_examples)
        attempts = [[] for _ in range(len(batch.orig_examples))]
        index_mapping = {i: i for i in range(len(batch.orig_examples))}  # mapping from indices in batch/strategies to indices in done
        while True:
            results = self.original_inference(batch)
            decisions = [strategy.decide(result, lambda code: self.test_results(code, example)) for
                         strategy, result, example in zip(strategies, results, batch.orig_examples)]
            new_index_mapping = {}
            new_strategies = []
            new_wrong_code = []
            new_batch = []
            for idx, decision in enumerate(decisions):
                if decision[0] == 'accept':
                    done[index_mapping[idx]] = decision[1]
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
        return [InferenceResult(code_sequence=code, info={'candidates' : [code], 'expanded' : [expanded]})
                   for code, expanded in zip(done, attempts)]

    def update_wrong_code_and_pack(self, new_examples, new_wrong_code):
        assert new_examples
        assert len(new_examples) == len(new_wrong_code)
        updated_examples = []
        for example, code in zip(new_examples, new_wrong_code):
            updated_examples.append(
                mutation.add_incorrect_code(example, tuple(code), self.add_trace, self.executor, check_ref_example=False))
        return self.batch_processor(updated_examples)

    def test_results(self, code, example):
        return evaluate_code(code, example.schema.args, example.tests, self.executor.execute)


class Strategy(ABC):
    @abstractmethod
    def decide(self, inference_result, evaluate):
        pass

    @staticmethod
    def get(descriptor):
        if ":" not in descriptor:
            descriptor += ":"
        start, *rest = descriptor.split(":")
        kwargs = eval("dict({})".format(":".join(*rest)))
        return {
            'greedy': lambda: GreedyStrategy
        }[start](**kwargs)


class TimeLimitStrategy(Strategy):

    @staticmethod
    def limit(init_strategy, limit):
        return lambda example: TimeLimitStrategy(init_strategy(example), limit)

    def __init__(self, strategy, limit):
        self.strategy = strategy
        self.limit = limit
        self.step = 0

    def decide(self, inference_result, evaluate):
        self.step += 1
        assert self.step <= self.limit
        decision, node = self.strategy.decide(inference_result, evaluate)
        if self.step == self.limit:
            return 'accept', node
        return decision, node


class GreedyStrategy(Strategy):
    def __init__(self, item):
        self.seen = set()
        del item  # no need

    def decide(self, inference_result, evaluate):
        unseen = []
        for considered in inference_result.info['candidates']:
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if res['correct'] == res['total']:
                return 'accept', considered
            unseen.append((res['correct'], considered))
        if not unseen:
            return 'accept', inference_result.info['candidates'][0]
        unseen.sort(reverse=True)
        self.seen.add(unseen[0][1])
        return 'expand', unseen[0][1]
