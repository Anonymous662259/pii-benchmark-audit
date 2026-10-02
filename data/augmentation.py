import random
from typing import List

from gliner2.training.data import InputExample

from config.hyperparams import AUG_FACTOR, AUG_RATIO, AUG_SEED

_aug_rng = random.Random(AUG_SEED)


def shuffle_entity_order(example: InputExample) -> InputExample:
    """Return a copy of example with entity keys randomly shuffled.

    Text and span offsets are unchanged; only the dict key order changes.
    Goal: reduce model bias toward label position in the prompt.
    """
    keys = list(example.entities.keys())
    _aug_rng.shuffle(keys)
    shuffled = {k: example.entities[k] for k in keys}
    kwargs = {"text": example.text, "entities": shuffled}
    for attr in ("classifications", "structures", "relations"):
        v = getattr(example, attr, None)
        if v:
            kwargs[attr] = v
    # Preserve exact char offsets; per-label value-list order is unchanged
    # (only key order in the outer dict changes), so the positional
    # correspondence between entities[label][i] and entity_char_spans[label][i]
    # remains valid.
    ecs = getattr(example, "entity_char_spans", None)
    if ecs is not None:
        kwargs["entity_char_spans"] = ecs
    return InputExample(**kwargs)


def augment_train_examples(
    examples: List[InputExample],
    aug_ratio: float = AUG_RATIO,
    aug_factor: int = AUG_FACTOR,
) -> List[InputExample]:
    """Augment training set with label-order shuffled copies.

    Args:
        examples: Original training examples.
        aug_ratio: Fraction of examples to augment.
        aug_factor: Number of augmented copies per selected example.

    Returns:
        Shuffled list of original + augmented examples.
    """
    n_before = len(examples)
    n_to_aug = int(round(n_before * aug_ratio))
    aug_idx = set(_aug_rng.sample(range(n_before), n_to_aug)) if n_to_aug > 0 else set()

    augmented = []
    for i, ex in enumerate(examples):
        augmented.append(ex)
        if i in aug_idx:
            for _ in range(aug_factor):
                augmented.append(shuffle_entity_order(ex))
    _aug_rng.shuffle(augmented)

    print(
        f"Label-order shuffle augment: {n_before} → {len(augmented)} train examples "
        f"(AUG_RATIO={aug_ratio}, AUG_FACTOR={aug_factor}, augmented {n_to_aug}/{n_before} examples)"
    )
    return augmented
