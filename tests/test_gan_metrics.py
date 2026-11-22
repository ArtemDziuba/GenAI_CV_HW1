"""Analytical MMD reductions and misleading-diversity counterexamples."""

import math

import pytest
import torch

from gan.metrics import class_statistics, polynomial_mmd


@pytest.mark.parametrize("block_size", [1, 2, 10])
def test_mmd_matches_full_unbiased_gram_estimator(block_size):
    real = torch.tensor([[1., 2.], [2., 0.], [-1., 1.]], dtype=torch.float64)
    fake = torch.tensor([[0., 0.], [3., 2.]], dtype=torch.float64)
    rr = (real @ real.T / 2 + 1).pow(3)
    ff = (fake @ fake.T / 2 + 1).pow(3)
    rf = (real @ fake.T / 2 + 1).pow(3)
    expected = (rr.sum() - rr.diagonal().sum()) / 6 + (ff.sum() - ff.diagonal().sum()) / 2 - 2 * rf.mean()
    assert polynomial_mmd(real, fake, block_size=block_size) == pytest.approx(expected.item())
    assert polynomial_mmd(fake, real, block_size=block_size) == pytest.approx(expected.item())


def test_negative_unbiased_mmd_is_not_clipped():
    values = torch.tensor([[0.], [1.]])
    assert polynomial_mmd(values, values) == pytest.approx(-3.5)
    assert polynomial_mmd(torch.zeros(3, 2), torch.zeros(4, 2)) == 0


@pytest.mark.parametrize("real,fake", [(torch.zeros(1, 2), torch.zeros(2, 2)),
                                       (torch.zeros(2, 2), torch.zeros(2, 3)),
                                       (torch.full((2, 2), float('nan')), torch.zeros(2, 2))])
def test_mmd_rejects_invalid_features(real, fake):
    with pytest.raises(ValueError):
        polynomial_mmd(real, fake)


def test_unrecognizable_predictions_do_not_become_confident_diversity():
    probabilities = torch.full((10, 10), .1 - .001 / 9)
    probabilities.diagonal().fill_(.101)
    metrics = class_statistics(probabilities, coverage_min_count=1)
    assert metrics['class_coverage'] == 10
    assert metrics['class_entropy'] == pytest.approx(math.log(10))
    assert metrics['confident_class_coverage'] == 0 and metrics['confident_fraction'] == 0
    assert metrics['confident_class_entropy'] is None
    assert metrics['low_confidence_fraction'] == 1


def test_balanced_confident_predictions_and_conditional_accuracy():
    labels = torch.arange(10).repeat_interleave(5)
    probabilities = torch.nn.functional.one_hot(labels, 10).float()
    metrics = class_statistics(probabilities, requested_labels=labels)
    assert metrics['class_coverage'] == metrics['confident_class_coverage'] == 10
    assert metrics['normalized_confident_class_entropy'] == pytest.approx(1)
    assert metrics['conditional_accuracy'] == metrics['confident_conditional_accuracy'] == 1
    collapsed = class_statistics(probabilities[:5])
    assert collapsed['class_coverage'] == 1 and collapsed['class_entropy'] == 0
