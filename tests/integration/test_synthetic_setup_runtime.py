"""Runtime-specific checks separated from portable behavioral tests."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def test_urdf_chain_jacobian_matches_finite_difference():
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT

    from alexdoor_xas.kinematics.purdue_chain import PurdueChain

    urdf = (
        REPOSITORY_ROOT
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    chain = PurdueChain(urdf, "cpu")
    q = torch.tensor([[-0.5, -0.5, 0.2, -1.0, 0.1, 0.2, 0.1]])
    pose, jac = chain.forward(q)
    for index in range(7):
        plus, minus = q.clone(), q.clone()
        plus[0, index] += 0.001
        minus[0, index] -= 0.001
        tp, _ = chain.forward(plus)
        tm, _ = chain.forward(minus)
        linear = (tp[0, :3, 3] - tm[0, :3, 3]).numpy() / 0.002
        angular = (
            Rotation.from_matrix((tp[0, :3, :3] @ tm[0, :3, :3].T).numpy()).as_rotvec() / 0.002
        )
        np.testing.assert_allclose(jac[0, :, index], np.r_[linear, angular], atol=2e-4)
    recovered, pe, re = chain.solve(pose[:, :3, 3], pose[:, :3, :3], q + 0.05)
    assert pe.item() < 1e-4 and re.item() < 1e-3
