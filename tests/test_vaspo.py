import math
from types import SimpleNamespace
import pytest
import torch
from _vaspo_fusion_shim import compute_vaspo_fused_advantage as fuse, compute_policy_loss_vaspo as loss


def fixture():
    return torch.tensor([[2., 2., -2., 0., 1., 99.]]), torch.tensor([[10., -10., 10., 10., 0., float('nan')]]), torch.zeros(1, 6), torch.tensor([[1, 1, 1, 1, 1, 0]])


def test_paper_formula_and_padding():
    a, teacher, old, mask = fixture()
    a.requires_grad_(); teacher.requires_grad_()
    out, d = fuse(a, teacher, old, mask)
    torch.testing.assert_close(out, torch.tensor([[2.4, 1.6, -2., 0., 1., 0.]]))
    assert d['applied_token_mask'].tolist() == [[True, True, False, False, False, False]]
    assert not out.requires_grad
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('width', [0., .2, .5])
def test_clip_width_ablation(width):
    a, t, old, m = fixture()
    out, _ = fuse(a, t, old, m, epsilon_w=width)
    torch.testing.assert_close(out[0, :2], torch.tensor([2 * (1 + width), 2 * (1 - width)]))


def test_mask_removal_preserves_positive_gate():
    a, t, old, m = fixture()
    out, d = fuse(a, t, old, m, token_mask_enabled=False)
    assert d['applied_token_mask'].tolist() == [[True, True, False, False, True, False]]
    assert out[0, 2] == -2


def test_current_denominator():
    a, t, old, m = fixture()
    current = torch.nan_to_num(t)
    out, _ = fuse(a, t, old, m, student_log_probs=current, student_prob_source='current')
    torch.testing.assert_close(out, torch.where(m.bool(), a, 0.))
    with pytest.raises(ValueError):
        fuse(a, t, old, m, student_prob_source='current')
    with pytest.raises(ValueError):
        fuse(a, t, old, m, student_log_probs=current, student_prob_source='current', whiten_advantages=True)


def test_whitening_full_valid_batch():
    a, t, old, m = fixture()
    out, d = fuse(a, t, old, m, whiten_advantages=True)
    raw = d['fused_raw'][m.bool()]
    expected = (raw - raw.mean()) / (raw.var() + 1e-8).sqrt()
    torch.testing.assert_close(out[m.bool()], expected)
    assert out[0, -1] == 0


def test_loss_response_means_and_gradients():
    # Different token counts expose accidental token-global/path-independent means.
    old = torch.zeros(2, 3)
    current = torch.zeros(2, 3, requires_grad=True)
    anchor = torch.tensor([[1., 2., 3.], [4., 0., 0.]])
    fused = torch.tensor([[1.2, 2., 3.], [4.8, 0., 0.]])
    valid = torch.tensor([[1, 1, 1], [1, 0, 0]])
    selected = torch.tensor([[1, 0, 0], [1, 0, 0]])
    result, _ = loss(old, current, fused, valid, config=SimpleNamespace(policy_loss={'epsilon': .2}),
                     vaspo_anchor_advantages=anchor, vaspo_token_mask=selected)
    torch.testing.assert_close(result, torch.tensor(-((1.2 + 2 + 3) / 3 + 4.8) / 2))
    result.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[1, 1:].count_nonzero() == 0


def test_loss_clipping_and_rollout_correction():
    old = torch.zeros(1, 2)
    current = torch.full((1, 2), math.log(2))
    adv = torch.tensor([[2., -2.]])
    result, _ = loss(old, current, adv, torch.ones(1, 2), config=SimpleNamespace(policy_loss={'epsilon': .2}),
                     vaspo_anchor_advantages=adv, vaspo_token_mask=torch.ones(1, 2), rollout_is_weights=torch.tensor([[2., 1.]]))
    torch.testing.assert_close(result, torch.tensor(-.4))


@pytest.mark.parametrize('kwargs', [{'epsilon_w':1.}, {'tanh_tau':0.}, {'fusion_mode':'add'}])
def test_invalid_and_removed_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        fuse(*fixture(), **kwargs)
