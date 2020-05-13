from abc import ABC, abstractmethod

from collections import defaultdict
from itertools import count

from datasets.executor import evaluate_code
from datasets.karel import mutation

from models.base import InferenceResult

from datasets.karel.utils import chunked
import numpy as np

class IterativeSearch:
    def __init__(self, original_inference, init_strategy, executor, add_trace, batch_processor, start_with_beams,
                 time_limit, overfit_model):
        self.original_inference = original_inference
        self.init_strategy = init_strategy
        self.executor = executor
        self.add_trace = add_trace
        self.batch_processor = batch_processor
        self.overfit_model = overfit_model
        # whether to start with the beams from the original model
        self.start_with_beams = start_with_beams
        self.time_limit = time_limit

    def __call__(self, batch):
        original_batch = batch

        strategies = [self.init_strategy(item) for item in batch.orig_examples]
        done = [False] * len(batch.orig_examples)
        finalized_candidates = [[] for _ in range(len(batch.orig_examples))]
        attempts = [[] for _ in range(len(batch.orig_examples))]
        index_mapping = {i: i for i in
                         range(len(batch.orig_examples))}  # mapping from indices in batch/strategies to indices in done
        num_inferences = 0
        for iteration_idx in count():
            if iteration_idx == 0 and self.start_with_beams:
                results = [example.ref_beams for example in batch.orig_examples]
                assert not any(
                    x is None for x in results), "the original examples must contain a full list of reference beams"
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
                give_up = num_inferences == self.time_limit and not finalized_candidates[index_mapping[idx]]
                if decision[0] == 'accept' or give_up:
                    finalized_candidates[index_mapping[idx]].append(decision[1])

                if decision[0] != 'expand' and self.overfit_model is None:
                    # only mark something done if the overfit model is not in effect,
                    # otherwise we keep going until we run out of time
                    done[index_mapping[idx]] = True
                else:
                    attempts[index_mapping[idx]].append(decision[1])
                    new_wrong_code.append(decision[1])
                    new_batch.append(batch.orig_examples[idx])
                    new_strategies.append(strategies[idx])
                    new_index_mapping[len(new_wrong_code) - 1] = index_mapping[idx]

            if all(done) or num_inferences == self.time_limit:
                break
            index_mapping = new_index_mapping
            strategies = new_strategies
            batch = self.update_wrong_code_and_pack(new_batch, new_wrong_code)

        if self.overfit_model is None:
            assert all(len(candidates) == 1 for candidates in finalized_candidates)
            best_code = [candidates[0] for candidates in finalized_candidates]
        else:
            best_code = self.select_best_code_per(original_batch, finalized_candidates)

        return [InferenceResult(code_sequence=code, info={'candidates': [code], 'expanded': [expanded]})
                for code, expanded in zip(best_code, attempts)]

    def update_wrong_code_and_pack(self, new_examples, new_wrong_code, add_trace=None, batch_processor=None):
        if add_trace is None:
            # this case is when we are using the forward model not the overfit model which may
            # need the trace
            add_trace = self.add_trace
        if batch_processor is None:
            batch_processor = self.batch_processor
        assert new_examples
        assert len(new_examples) == len(new_wrong_code)
        updated_examples = []
        for example, code in zip(new_examples, new_wrong_code):
            updated_examples.append(
                mutation.add_incorrect_code(example, tuple(code), add_trace, self.executor,
                                            check_ref_example=False))
        return batch_processor(updated_examples)

    def test_results(self, code, example):
        return evaluate_code(code, example.schema.args, example.input_tests, self.executor.execute)

    def run_overfit_model(self, items):
        egs, cands = zip(*items)
        batch = self.update_wrong_code_and_pack(egs, cands, add_trace=True,
                                                batch_processor=self.overfit_model.batch_processor(for_eval=True))
        return self.overfit_model.inference(batch)

    def select_best_code_per(self, original_batch, finalized_candidates):
        flattened_candidates = []
        flattened_examples = []
        indices_per_original = []
        for example, candidates in zip(original_batch.orig_examples, finalized_candidates):
            indices_per_original.append([])
            for candidate in candidates:
                indices_per_original[-1].append(len(flattened_examples))
                flattened_examples.append(example)
                flattened_candidates.append(candidate)

        results = []
        for items in chunked(zip(flattened_examples, flattened_candidates), len(original_batch.orig_examples)):
            results += self.run_overfit_model(items).cpu().detach().numpy().tolist()
        best_code = []
        for idxs, candidates in zip(indices_per_original, finalized_candidates):
            best_idx = max(range(len(candidates)), key=lambda i: results[idxs[i]])
            best_code.append(candidates[best_idx])
        return best_code


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
            'best_first': lambda: BestFirstSearch,
            'diverse': lambda: DiversitySearch
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


class DiversitySearch(Strategy):
    """
    Add a new alternative to BestFirstStrategy that takes into
    account semantic diversity, that is it breaks ties by 
    picking programs that pass different sets of test cases 
    from ones expanded in the past. see res[‘individual’] 

    """
    def __init__(self, item, past=[]):
        self.seen = set()
        self.by_number_correct = defaultdict(list)
        self.individual_correct = defaultdict(list)
        # should it contain beam number and res['individual']?

    def calculate_key(self, choose_idx, types_correct, corr_idx, corr_idx_compare):
        """
        Calculat
        """
        set1_elem = choose_idx[types_correct[corr_idx]]
        set2_elem = choose_idx[types_correct[corr_idx_compare]]
        diff1 = len(np.setdiff1d(set1_elem, set2_elem))
        diff2 = len(np.setdiff1d(set2_elem, set1_elem))
        key = int(diff1+diff2)
        return key

    def diversify_decision(self, code, correct):
        # check which of the 5 tests are passed
        choose_idx = np.arange(len(correct[0][1]))
        types_correct = [np.array(corr[1])>0 for corr in correct]
        
        # if only 1 beam return that
        if len(types_correct)==1:
            return 'expand', code.pop(0)

        # compare using the set difference and pick the one that is most different
        order = defaultdict(list)
        for corr_idx, corr in enumerate(correct):
            for corr_idx_compare, corr_compare in enumerate(correct):
                if corr_idx_compare<=corr_idx:
                    continue
                key = self.calculate_key(choose_idx, types_correct, corr_idx, corr_idx_compare)
                order[key].append(corr_idx)
        # remove duplicates
        for key in order.keys():
            order[key] = list(set(order[key]))
        best_choice_idx = np.array(list(order.keys())).max()
        
        return 'expand', code[np.random.choice(order[best_choice_idx],1)[0]]

    def decide(self, candidates, evaluate):
        for candidate_idx, considered in enumerate(candidates):
            considered = tuple(considered)
            if considered in self.seen:
                continue
            res = evaluate(considered)
            if not valid(considered, res):
                continue
            self.seen.add(considered)
            assert res['total'] == 5
            self.by_number_correct[res['correct']].append(considered)
            self.individual_correct[res['correct']].append([candidate_idx,res['individual']])

        for n_correct in sorted(self.by_number_correct, reverse=True):
            if self.by_number_correct[n_correct]:
                decision = 'accept' if n_correct == 5 else 'expand'
                if decision == 'expand':
                    self.diversify_decision(self.by_number_correct[n_correct], self.individual_correct[n_correct])
                else:
                    return decision, self.by_number_correct[n_correct].pop(0)

        return 'accept', tuple(candidates[0])