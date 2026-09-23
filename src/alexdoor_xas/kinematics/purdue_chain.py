"""URDF-derived batched arm kinematics for synthetic expert screening only."""

import xml.etree.ElementTree as ET

import numpy as np
import torch

from alexdoor_xas.assets.purdue import ARM_JOINTS, derive_push_geometry, origin_matrix


class PurdueChain:
    def __init__(self, urdf, device="cuda:0"):
        self.device = device
        root = ET.parse(urdf).getroot()
        parents = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], "RIGHT_GRIPPER_Y_LINK"
        while link in parents:
            joint = parents[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain, limits = [], {}
        for joint in reversed(chain):
            name = joint.get("name")
            index = ARM_JOINTS.index(name) if name in ARM_JOINTS else None
            if index is None and joint.get("type") != "fixed":
                raise ValueError(f"Unexpected active ancestor {name}")
            axis = (
                np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                if index is not None
                else np.zeros(3)
            )
            x, y, z = axis
            cross = self.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            self.chain.append(
                (self.array(origin_matrix(joint.find("origin"))), index, self.array(axis), cross)
            )
            if index is not None:
                limits[index] = [float(joint.find("limit").get(k)) for k in ("lower", "upper")]
        self.limits = self.array([limits[i] for i in range(7)])
        geometry = derive_push_geometry(urdf)
        tip = np.eye(4)
        tip[:3, 3] = geometry.translation
        self.tip = self.array(geometry.wrist_from_base @ tip)

    def array(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def forward(self, q):
        transform = torch.eye(4, device=self.device).expand(q.shape[0], 4, 4).clone()
        origins, axes = {}, {}
        for origin, index, axis, cross in self.chain:
            transform = transform @ origin
            if index is not None:
                origins[index] = transform[:, :3, 3].clone()
                axes[index] = transform[:, :3, :3] @ axis
                angle = q[:, index, None, None]
                rotation = (
                    torch.eye(3, device=self.device)
                    + angle.sin() * cross
                    + (1 - angle.cos()) * (cross @ cross)
                )
                motion = torch.eye(4, device=self.device).expand(q.shape[0], 4, 4).clone()
                motion[:, :3, :3] = rotation
                transform = transform @ motion
        transform = transform @ self.tip
        linear = [torch.linalg.cross(axes[i], transform[:, :3, 3] - origins[i]) for i in range(7)]
        jacobian = torch.cat(
            (torch.stack(linear, -1), torch.stack([axes[i] for i in range(7)], -1)), 1
        )
        return transform, jacobian

    def solve(self, position, rotation, initial, iterations=100):
        """Bounded local solve; failure is not proof of global unreachability."""
        q = initial.clone()
        eye = torch.eye(6, device=self.device)
        for _ in range(iterations):
            transform, jac = self.forward(q)
            # Local rotation log; multistart screening does not prove unreachability.
            relative = rotation @ transform[:, :3, :3].transpose(-1, -2)
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
                -1 + 1e-6, 1 - 1e-6
            )
            angle = cosine.acos()
            skew = torch.stack(
                (
                    relative[:, 2, 1] - relative[:, 1, 2],
                    relative[:, 0, 2] - relative[:, 2, 0],
                    relative[:, 1, 0] - relative[:, 0, 1],
                ),
                -1,
            )
            re = skew * (angle / (2 * angle.sin())).unsqueeze(-1)
            error = torch.cat((position - transform[:, :3, 3], re), -1)
            delta = jac.transpose(-1, -2) @ torch.linalg.solve(
                jac @ jac.transpose(-1, -2) + 0.01**2 * eye, error.unsqueeze(-1)
            )
            q = (q + delta.squeeze(-1).clamp(-0.15, 0.15)).clamp(
                self.limits[:, 0], self.limits[:, 1]
            )
        transform, _ = self.forward(q)
        pe = (position - transform[:, :3, 3]).norm(dim=-1)
        relative = rotation @ transform[:, :3, :3].transpose(-1, -2)
        re = (((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).acos()
        return q, pe, re
