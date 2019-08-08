# -*- coding: utf-8 -*-
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Union

import torch.nn

from utils.utils import flatten

SUPPORTED_REDUCTIONS = [None, "mean"]

EPSILON = 1e-15


class DiceLoss(torch.nn.Module):
    """
    The Sørensen-Dice Loss.
    """

    def __init__(self, reduction: Union[None, str] = "mean"):
        super(DiceLoss, self).__init__()
        assert reduction in SUPPORTED_REDUCTIONS, "Reduction type not supported."
        self._reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
        """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            inputs (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The model prediction on which the loss has to be computed.
            targets (:obj:`torch.Tensor`) : A tensor of shape (B, C, ..). The ground truth.
            ignore_index (int): An index to ignore for computation.
        Returns:
            :obj:`torch.Tensor`: The Sørensen–Dice loss for each class or reduced according to reduction method.
        """

        assert inputs.size() == targets.size(), "'Inputs' and 'Targets' must have the same shape."

        inputs = flatten(inputs)
        targets = flatten(targets)

        targets = targets.float()

        # Compute per channel Dice Coefficient
        intersect = (inputs * targets).sum(-1)
        denominator = (inputs + targets).sum(-1)
        dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON))
        accuracy = 100 * (dice / 32)

        if weights is not None:
            if weights.requires_grad is not False:
                weights.requires_grad = False
            intersect = weights * intersect
            dice = 1.0 - (2.0 * intersect / denominator.clamp(min=EPSILON))

        return dice, accuracy


