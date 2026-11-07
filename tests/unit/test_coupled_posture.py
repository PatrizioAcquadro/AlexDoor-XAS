"""Essential numerical contracts; native qualification remains independent."""

import pytest
import torch

from alexdoor_xas.kinematics.posture import (
    coupled_margin_preference,
    joint_command_position_response,
)


def test_constrained_response_matches_one_step_momentum_with_passive_follower():
    # One driven coordinate and a passive mimic follower move together. The
    # native effective mass already includes motor armature; damping is distinct.
    mass = torch.diag(torch.tensor([2., 3.], dtype=torch.float64))[None]
    basis = torch.ones(2, 1, dtype=torch.float64)
    response = joint_command_position_response(
        mass, basis, torch.tensor([0., .05], dtype=torch.float64),
        torch.tensor([40., 0.], dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
        [0], .1, 1,
    )
    torch.testing.assert_close(response, torch.tensor([[[.1**2*40/(5+.1*.05)]]],
                                                       dtype=torch.float64))


def preference_inputs(dtype=torch.float64):
    jacobian = torch.eye(6, 7, dtype=dtype)[None]
    jacobian[0, 0, 6] = 1
    positions = torch.zeros(1, 7, dtype=dtype)
    positions[0, 0], positions[0, 6] = .8, -.8
    previous = positions.clone()
    # These loaded target coordinates exceed measured support. They remain
    # valid actuator targets, and must not be clamped to empirical support.
    previous[0, 0], previous[0, 6] = 1.2, -1.2
    primary = previous.clone()
    primary[0, 1] += .001
    return dict(
        jacobian=jacobian, positions=positions, measured_velocity=torch.zeros_like(positions),
        primary_target=primary, previous_target=previous,
        limits=torch.tensor([[[-2., 2.]]*7], dtype=dtype),
        velocities=torch.ones_like(positions)*.45, dt=1/60,
        support=torch.tensor([[[-1., 1.]]*7], dtype=dtype),
        stopping_reserve=torch.ones_like(positions)*.01,
        position_response=torch.diag(torch.tensor([.1, .2, .3, .4, .5, .6, .7],
                                                  dtype=torch.float64))[None],
        gain=2.,
    )


def test_inward_preference_improves_coupled_faces_and_preserves_primary_and_loaded_targets():
    values = preference_inputs()
    snapshot = {key: value.clone() for key, value in values.items() if torch.is_tensor(value)}
    target, info = coupled_margin_preference(**values)
    repeated, _ = coupled_margin_preference(**values)
    assert torch.equal(target, repeated)
    assert all(torch.equal(values[key], value) for key, value in snapshot.items())
    assert target[0, 0] < values['primary_target'][0, 0]
    assert target[0, 6] > values['primary_target'][0, 6]
    assert target[0, 0] > 1 and target[0, 6] < -1
    assert info['score_after'][0] > info['score_before'][0]
    torch.testing.assert_close(
        values['jacobian'] @ (target-values['primary_target']).unsqueeze(-1),
        torch.zeros(1, 6, 1, dtype=target.dtype), atol=1e-14, rtol=0,
    )
    assert bool(((target-values['previous_target']).abs() <= .45/60).all())
    assert info['predicted_score_is_not_an_admission_gate']


def test_negative_empirical_score_is_not_a_new_refusal_and_zero_gain_is_exact():
    values = preference_inputs()
    values['positions'][0, 0] = 1.01
    target, info = coupled_margin_preference(**values)
    assert info['score_before'][0] < 0 and torch.isfinite(target).all()
    values['gain'] = 0.
    target, _ = coupled_margin_preference(**values)
    assert torch.equal(target, values['primary_target'])
    values['gain'] = float('nan')
    with pytest.raises(ValueError, match='gain_or_dt'):
        coupled_margin_preference(**values)


def test_existing_primary_float32_rate_representation_does_not_create_refusal():
    values = preference_inputs(torch.float32)
    upper = values['previous_target'] + values['velocities']*values['dt']
    values['primary_target'][0, 0] = torch.nextafter(upper[0, 0], torch.tensor(torch.inf))
    target, _ = coupled_margin_preference(**values)
    assert bool((target <= torch.maximum(upper, values['primary_target'])).all())
    lower = values['previous_target'] - values['velocities']*values['dt']
    assert bool((target >= torch.minimum(lower, values['primary_target'])).all())
