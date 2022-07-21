"""
Originally, the code is taken from the official FLAME PyTorch repo:
https://github.com/soubhiksanyal/FLAME_PyTorch
"""
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lbs
from . import utils


class FlameMM(nn.Module):
    """
    Given FLAME parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks.
    """

    def __init__(
        self,
        flame_model_path: str,
        flame_lmk_embedding_path: str,
        n_shape: int = 100,
        n_exp: int = 50,
    ) -> None:
        super(FlameMM, self).__init__()

        flame_model = utils.load_flame(flame_model_path)

        self.dtype = torch.float32
        self.register_buffer(
            "faces_tensor",
            utils.to_tensor(
                utils.to_np(flame_model.f, dtype=np.int64), dtype=torch.long
            ),
        )
        # The vertices of the template model
        self.register_buffer(
            "v_template",
            utils.to_tensor(utils.to_np(flame_model.v_template), dtype=self.dtype),
        )
        # The shape components and expression
        shapedirs = utils.to_tensor(
            utils.to_np(flame_model.shapedirs), dtype=self.dtype
        )
        shapedirs = torch.cat(
            [
                shapedirs[:, :, :n_shape],
                shapedirs[:, :, 300 : 300 + n_exp],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer(
            "posedirs", utils.to_tensor(utils.to_np(posedirs), dtype=self.dtype)
        )
        self.register_buffer(
            "J_regressor",
            utils.to_tensor(utils.to_np(flame_model.J_regressor), dtype=self.dtype),
        )
        parents = utils.to_tensor(utils.to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights",
            utils.to_tensor(utils.to_np(flame_model.weights), dtype=self.dtype),
        )

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        )
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter(
            "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        )

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(
            flame_lmk_embedding_path, allow_pickle=True, encoding="latin1"
        )
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx",
            torch.tensor(lmk_embeddings["static_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.tensor(lmk_embeddings["static_lmk_bary_coords"], dtype=self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx",
            lmk_embeddings["dynamic_lmk_faces_idx"].clone().detach().to(torch.long),
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            lmk_embeddings["dynamic_lmk_bary_coords"].clone().detach().to(torch.long),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def seletec_3d68(self, vertices: torch.Tensor) -> torch.Tensor:
        landmarks3d = lbs.vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def forward(
        self,
        shape_params: torch.Tensor,
        expression_params: torch.Tensor,
        pose_params: Optional[torch.Tensor] = None,
        eye_pose_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks2d: N X number of landmarks X 3
            landmarks3d: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [
                pose_params[:, :3],
                self.neck_pose.expand(batch_size, -1),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            dim=1,
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs.lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(
            batch_size, -1, -1
        )

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = lbs.find_dynamic_lmk_idx_and_bcoords(
            vertices,
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = lbs.vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = vertices.shape[0]
        landmarks3d = lbs.vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        return vertices, landmarks2d, landmarks3d
