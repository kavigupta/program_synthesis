import collections
import copy
import itertools
import struct
import json

import numpy as np

from ..dataset import executor
from . import parser_for_synthesis
from ..karel import KarelForSynthesisParser, KarelSyntaxError

# Tree structure
# - run: body
# - if: cond, body
# - ifElse: cond, ifBody, elseBody
# - while: cond, body
# - repeat: times, body
# - not: cond



def masked_uniform(choices, i):
    prob = np.full(choices, 1. / (choices - 1))
    prob[i] = 0
    return prob


def choose(rng, sequence, p=None):
    # Using rng.choice directly on sequence sometimes leads to undesirable
    # effects, as that entails converting sequence into a numpy.ndarray.
    return sequence[rng.choice(len(sequence), p=p)]


CONDS = [{
    'type': t
}
    for t in ('frontIsClear', 'leftIsClear', 'rightIsClear',
              'markersPresent', 'noMarkersPresent')]
# no not for markersPresent and noMarkersPresent
CONDS.extend({'type': 'not', 'cond': cond} for cond in CONDS[:3])
CONDS_MASKED_PROBS = {
    n: masked_uniform(len(CONDS), i)
    for i, n in enumerate(('frontIsClear', 'leftIsClear', 'rightIsClear', 'markersPresent',
                           'noMarkersPresent', 'notfrontIsClear', 'notleftIsClear',
                           'notrightIsClear'))
}

ACTION_NAMES = ('move', 'turnLeft', 'turnRight', 'putMarker', 'pickMarker')


def get_action_name_id(action_name):
    return ACTION_NAMES.index(action_name)


ACTIONS_MASKED_PROBS = {
    n: masked_uniform(len(ACTION_NAMES), i)
    for i, n in enumerate(ACTION_NAMES)
}
ACTIONS = [{
    'type': t
} for t in ACTION_NAMES]
ACTIONS_DICT = dict(zip(ACTION_NAMES, ACTIONS))

REPEAT_COUNTS = [{'type': 'count', 'value': i} for i in range(2, 11)]
REPEAT_MASKED_PROBS = [None, None] + [masked_uniform(len(REPEAT_COUNTS), i) for
                                      i in range(len(REPEAT_COUNTS))]

BLOCK_TYPE = ['if', 'while', 'repeat']


def get_block_type_id(block_type):
    return BLOCK_TYPE.index(block_type)


def random_singular_block(rng):
    type_ = rng.choice(BLOCK_TYPE)
    if type_ == 'repeat':
        return {'type': type_, 'times': rng.choice(REPEAT_COUNTS)}
    else:
        return {'type': type_, 'cond': rng.choice(CONDS)}


# operations:
# - Add action
ADD_ACTION = 0
# - Remove action
REMOVE_ACTION = 1
# - Replace action
REPLACE_ACTION = 2
# - Unwrap if/ifElse/while/repeat
UNWRAP_BLOCK = 3
# - Wrap with if/while/repeat
WRAP_BLOCK = 4
# - Wrap with ifElse
WRAP_IFELSE = 5
# - Change condition in if/ifElse/while
REPLACE_COND = 6  # Not implemented
# - Switch between if/while
SWITCH_IF_WHILE = 7  # Not implemented


class Operation:
    # TODO: Move operations into this class

    @staticmethod
    def total():  # Implemented operations
        return 6


DEFAULT_PROBS = np.array([1, 1, 1, 1, .25, .75, 1, 1], dtype=float)

BodyInfo = collections.namedtuple('BodyInfo', ['node', 'type', 'elems'])


class TreeIndex(object):
    def __init__(self, tree):
        self.action_locs = []
        self.cond_locs = []
        self.all_bodies = []
        self.unwrappables = []
        self.all_if_whiles = []

        queue = collections.deque([(tree, (None, None))])
        while queue:
            node, address = queue.popleft()
            if node['type'] == 'ifElse':
                bodies = [BodyInfo(node, 'ifElse-if', node['ifBody']),
                          BodyInfo(node, 'ifElse-else', node['elseBody'])]
                self.unwrappables.append(address)
            elif 'body' in node:
                bodies = [BodyInfo(node, node['type'], node['body'])]
                if address[0]:
                    self.unwrappables.append(address)
            else:
                bodies = []
                self.action_locs.append(address)
            if 'cond' in node or 'times' in node:
                self.cond_locs.append(node)
            if node['type'] in ('if', 'while'):
                self.all_if_whiles.append(node)

            for body in bodies:
                for i, child in enumerate(body.elems):
                    queue.append((child, (body.elems, i)))
            self.all_bodies.extend(bodies)

        self.add_locs = [(body.elems, i)
                         for body in self.all_bodies for i in range(
                len(body) + 1)]
        self.remove_locs = [x for x in self.action_locs if len(x[0]) > 1]

    def count_actions(self):
        # wrap_block_choices: (n + 1) choose 2 for each len(body)
        # wrap_ifelse_choices: (n + 1) choose 3 for each len(body)
        wrap_block_choices = np.array(
            [len(body.elems) for body in self.all_bodies], dtype=float)
        wrap_ifelse_choices = wrap_block_choices.copy()
        wrap_block_choices *= (wrap_block_choices + 1)
        wrap_block_choices /= 2
        wrap_ifelse_choices *= (wrap_ifelse_choices + 1) * (
                wrap_ifelse_choices - 1)
        wrap_ifelse_choices /= 6


def mutate(tree, probs=None, rng=None):
    if probs is None:
        probs = DEFAULT_PROBS.copy()
    if rng is None:
        rng = np.random.RandomState()

    assert len(probs) == 8
    assert tree['type'] == 'run'

    tree_index = TreeIndex(tree)

    # wrap_block_choices: (n + 1) choose 2 for each len(body)
    # wrap_ifelse_choices: (n + 1) choose 3 for each len(body)
    wrap_block_choices = np.array(
        [len(body.elems) for body in tree_index.all_bodies], dtype=float)
    wrap_ifelse_choices = wrap_block_choices.copy()
    wrap_block_choices *= (wrap_block_choices + 1)
    wrap_block_choices /= 2
    wrap_ifelse_choices *= (wrap_ifelse_choices + 1) * (
            wrap_ifelse_choices - 1)
    wrap_ifelse_choices /= 6

    probs[ADD_ACTION] *= len(tree_index.add_locs)
    probs[REMOVE_ACTION] *= len(tree_index.remove_locs)
    probs[REPLACE_ACTION] *= len(tree_index.action_locs)
    probs[UNWRAP_BLOCK] *= len(tree_index.unwrappables)
    probs[WRAP_BLOCK] *= sum(wrap_block_choices)
    probs[WRAP_IFELSE] *= sum(wrap_ifelse_choices)
    probs[REPLACE_COND] *= len(tree_index.cond_locs)
    probs[SWITCH_IF_WHILE] *= len(tree_index.all_if_whiles)
    probs_sum = np.sum(probs)
    if probs_sum == 0:
        raise Exception('No mutation possible')
    probs /= probs_sum

    choice = rng.choice(8, p=probs)
    if choice == ADD_ACTION:
        body, i = choose(rng, tree_index.add_locs)
        body.insert(i, rng.choice(ACTIONS))
    elif choice == REMOVE_ACTION:
        body, i = choose(rng, tree_index.remove_locs)
        del body[i]
    elif choice == REPLACE_ACTION:
        body, i = choose(rng, tree_index.action_locs)
        body[i] = choose(rng, ACTIONS,
                         p=ACTIONS_MASKED_PROBS[body[i]['type']])
    elif choice == UNWRAP_BLOCK:
        body, i = choose(rng, tree_index.unwrappables)
        block = body[i]
        del body[i]
        body[i:i] = block.get('body', [])
        body[i:i] = block.get('elseBody', [])
        body[i:i] = block.get('ifBody', [])
    elif choice == WRAP_BLOCK:
        wrap_block_choices /= np.sum(wrap_block_choices)
        body = choose(rng, tree_index.all_bodies, p=wrap_block_choices).elems
        bounds = list(itertools.combinations(range(len(body) + 1), 2))
        left, right = choose(rng, bounds)
        subseq = body[left:right]
        del body[left:right]
        new_block = random_singular_block(rng)
        new_block['body'] = subseq
        body.insert(left, new_block)
    elif choice == WRAP_IFELSE:
        wrap_ifelse_choices /= np.sum(wrap_ifelse_choices)
        body = choose(rng, tree_index.all_bodies, p=wrap_ifelse_choices).elems
        bounds = list(itertools.combinations(range(len(body) + 1), 3))
        left, mid, right = choose(rng, bounds)
        if_body = body[left:mid]
        else_body = body[mid:right]
        del body[left:right]
        new_block = {
            'type': 'ifElse',
            'cond': rng.choice(CONDS),
            'ifBody': if_body,
            'elseBody': else_body
        }
        body.insert(left, new_block)
    elif choice == REPLACE_COND:
        node = choose(rng, tree_index.cond_locs)
        if 'cond' in node:
            node['cond'] = rng.choice(
                CONDS,
                p=CONDS_MASKED_PROBS[node['cond']['type'] + node['cond'].get(
                    'cond', {}).get('type', '')])
        elif 'repeat' in node:
            node['repeat'] = rng.choice(
                REPEAT_COUNTS,
                p=REPEAT_MASKED_PROBS[node['repeat']['times']['value']])
    elif choice == SWITCH_IF_WHILE:
        node = choose(rng, tree_index.all_if_whiles)
        node['type'] = {'if': 'while', 'while': 'if'}[node['type']]

    return tree


def mutate_n(tree, count, probs=None, rng=None, allow_in_place=False):
    if rng is None:
        rng = np.random.RandomState()
    if count == 1:
        if allow_in_place:
            return mutate(tree, probs, rng)
        return mutate(copy.deepcopy(tree), probs, rng)

    previous_seqs = set([parser_for_synthesis.tree_to_tokens(tree)])
    for i in range(count):
        found = False
        for _ in range(1000):
            tree = copy.deepcopy(tree)
            mutate(tree, probs, rng)
            new_seq = parser_for_synthesis.tree_to_tokens(tree)
            if new_seq not in previous_seqs:
                previous_seqs.add(new_seq)
                found = True
                break
        if not found:
            raise Exception('Rejection sampling failed')
    return tree


def add_incorrect_code(karel_example, new_code, add_trace, executor, check_ref_example=True, code_is_correct=None, beams=None):
    from ..dataset import KarelExample
    if check_ref_example:
        assert karel_example.ref_example is None
    # TODO: Get the real trace
    new_tests = []
    if add_trace:
        for ex in karel_example.input_tests:
            result = executor.execute(new_code, None, ex['input'],
                                      record_trace=True, strict=True)
            new_ex = dict(ex)
            new_ex['trace'] = result.trace
            new_tests.append(new_ex)
    karel_example.ref_example = KarelExample(
        idx=None,
        guid=None,
        code_sequence=new_code,
        input_tests=new_tests,
        tests=karel_example.tests,
        code_is_correct=code_is_correct)
    karel_example.ref_beams = beams
    return karel_example


class KarelExampleMutator(object):
    def __init__(self, n_dist, rng_fixed, add_trace, probs=None):
        self.n_dist = n_dist / np.sum(n_dist)
        self.rng_fixed = rng_fixed
        self.add_trace = add_trace
        self.probs = probs

        self.rng = np.random.RandomState()
        self.parser = parser_for_synthesis.KarelForSynthesisParser(
            build_tree=True)
        self.executor = executor.KarelExecutor(action_limit=250)

    def __call__(self, karel_example):
        return add_incorrect_code(karel_example, self.mutate_code(karel_example), self.add_trace, self.executor)

    def mutate_code(self, karel_example):
        tree = self.parser.parse(karel_example.code_sequence)
        if self.rng_fixed:
            self.rng.seed(int(karel_example.guid[:8], base=16))
        n = self.rng.choice(len(self.n_dist), p=self.n_dist) + 1
        new_tree = mutate_n(tree, n, self.probs, self.rng, allow_in_place=True)
        new_code = parser_for_synthesis.tree_to_tokens(new_tree)
        return new_code


class KarelOutputRefExampleMutator(object):
    def __init__(self, to_be_used_indices, ref_code, code_is_correct, beams, add_trace):
        """
        Represents a list of reference outputs, one per example in the training data.

        This is used to represent incorrect programs, but it can also be used to represent
            potentially overfit programs.

        Arguments:
            to_be_used_indices: a list of indices of the code examples to be used

            ref_code: a list of tuples, each of which represents a program to be used.
                Each of these examples must correspond to the equivalent index in
                to_be_used_indices

            code_is_correct: a list of whether each code example is correct

            beams: a list of all the beams (including ref_code) for each code example

            add_trace: whether to add the execution trace when modifying a program
        """
        self.add_trace = add_trace
        self.executor = executor.KarelExecutor(action_limit=250)
        self.to_be_used_indices = to_be_used_indices
        self.beams = beams
        self.code_is_correct = code_is_correct
        self.ref_code = ref_code

    @classmethod
    def from_path(cls, karel_ref_file_train, add_trace, mode='debugger', for_eval=False):
        """
        Get a mutator from the given file.

        Arguments:
            karel_ref_file_train: the file to get the data from
            add_trace: whether or not to add the traces
            mode: which mode to load the data in.
                If 'debugger', load all examples which do not pass all 5 test cases
                If 'overfit-check', load all examples which pass 5 test cases, and
                    ensure an equal number pass the held out test or not.
            for_eval: whether the given dataset will be used for evaluation or not. If so, do not require a 50/50 split
                in some cases
        """
        if karel_ref_file_train is None:
            return None

        with open(karel_ref_file_train) as f:
            examples = json.load(f)

        parser = KarelForSynthesisParser()

        def is_valid_syntax(x):
            try:
                parser.parse(tuple(x['output']), debug=False)
            except KarelSyntaxError:
                return False
            return True

        def passes_given_tests(x):
            return x['passes_given_tests']

        can_be_used = {
            'debugger': lambda x: not passes_given_tests(x) and is_valid_syntax(x),
            'overfit-check': lambda x: passes_given_tests(x) and is_valid_syntax(x)
        }[mode]

        to_be_used_idx = [i for i, x in enumerate(examples) if can_be_used(x)]
        if mode == 'overfit-check' and not for_eval:
            to_be_used_idx = equal_halves(to_be_used_idx, lambda x: examples[x]['is_correct'])
        negative_examples = [tuple(examples[i]['output']) for i in to_be_used_idx]
        code_is_correct = [examples[i]['is_correct'] for i in to_be_used_idx]
        # get each of the beams. If not found the output is the only beam
        beams = [examples[i].get('beams', [examples[i]['output']]) for i in to_be_used_idx]
        return cls(to_be_used_idx, negative_examples, code_is_correct, beams, add_trace)

    def filter_index(self, index):
        return [index[i] for i in self.to_be_used_indices]

    def __call__(self, idx, karel_example):
        assert self.ref_code[idx]
        result = add_incorrect_code(karel_example, self.ref_code[idx], self.add_trace, self.executor,
                                    code_is_correct=self.code_is_correct[idx], beams=self.beams[idx])
        assert result.ref_example.code_sequence
        return result


def equal_halves(items, predicate, seed=0):
    rng = np.random.RandomState(seed)
    pos_idx = []
    neg_idx = []
    for i, item in enumerate(items):
        if predicate(item):
            pos_idx.append(i)
        else:
            neg_idx.append(i)
    if len(pos_idx) < len(neg_idx):
        rng.shuffle(neg_idx)
        neg_idx = neg_idx[:len(pos_idx)]
    else:
        rng.shuffle(pos_idx)
        pos_idx = pos_idx[:len(neg_idx)]
    all_idx = sorted(pos_idx + neg_idx)
    return [items[i] for i in all_idx]


# Definition of Action Parameters

Action = collections.namedtuple('Action', ['id', 'parameters'])

ActionAddParameters = collections.namedtuple('ActionAddParameters', ['location', 'token'])
ActionRemoveParameters = collections.namedtuple('ActionRemoveParameters', ['location'])
ActionReplaceParameters = collections.namedtuple('ActionReplaceParameters', ['location', 'token'])
ActionUnwrapBlockParameters = collections.namedtuple('ActionUnwrapBlockParameters', ['location'])
ActionWrapBlockParameters = collections.namedtuple('ActionWrapBlockParameters',
                                                   ['block_type', 'cond_id', 'start', 'end'])
ActionWrapIfElseParameters = collections.namedtuple('ActionWrapIfElseParameters',
                                                    ['cond_id', 'if_start', 'else_start', 'end'])

# Obsolete notes
# ==============
# Actions: move, turnLeft, turnRight, putMarker, pickMarker
# Conditions: frontIsClear, leftIsClear, rightIsClear, markersPresent (+ not)
# Atoms:
# - actions (5)
# - if: pick cond (8) and pick action (5) = 40
# - ifElse: pick cond (8) and pick ifBody (5) and elseBody(5) = 200
#   if nots not allowed, then 100
# - while: pick cond (8) and pick action (5) = 40
# - repeat: pick times (9: 2..10) and body (5)
