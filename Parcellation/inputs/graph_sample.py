# -*- coding: utf-8 -*-
# Copyright 2019 SAMITorch Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# gtou may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch

"""
Class to describe a Graph Sample object:

"""
class GraphSample(object):

    def __init__(self, x=None, edge_idx=None, edge_wht=None, gt=None, xyz=None, face=None, age=None, sx=None, lab=None):
        """
        Graph_Sample initializer.
        Args:
            x --> (Tensor, optional): Node feature matrix: [num_nodes, num_node_features]
            edge_idx --> (LongTensor, optional): Graph connectivity XY matrix: [2, num_edges]
            edge_wgt --> (Tensor, optional): Edge weights/feature matrix [num_edges, num_edge_features].
                           The last dimension same as edge_idx.            
            gt --> (Tensor, optional): One-Hot_Encoded GT matrix:[num_nodes, num_classes]. 
                   For node classication/Graph segmentaiton task.
    
            xyz --> (Tensor, optional): Node represented in Euclidean coordinates: [num_nodes, 3]
            face --> Triangulated mesh face surface [num_nodes, 3]

            age --> (Tensor, optional): Brain age of the subject: [scalar]
            sx --> (Tensor, optional): Gender of the subject if available: [scalar]
            lab --> (Tensor, optional): One-Hot_Encoded GT matrix:[1, num_classes]
                    For graph classication task.

            
            !!!!  Need to implement this. !!!!
            
            If there are other infrormation from the brain surface. Add them as keys in the argument. 
        """

        self._x = x
        self._edge_idx = edge_idx
        self._edge_wht = edge_wht
        self._gt = gt
        self._xyz = xyz
        self._face = face
        self._age = age
        self._sx = sx
        self._lab = lab

    @staticmethod
    def from_dict(dictionary):
        data = GraphSample()
        for key, item in dictionary.items():
            data[key] = item
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)

    @property
    def keys(self):
        return [key for key in self.__dict__.keys() if self[key] is not None]

    def __len__(self):
        return len(self.keys)

    def __contains__(self, key):
        return key in self.keys

    def __iter__(self):
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if self[key] is not None:
                yield key, self[key]

    def cat_dim(self, key):
        return -1 if self[key].dtype == torch.long else 0
    @property
    def num_nodes(self):
        for key, item in self('x'):
            return item.size(self.cat_dim(key))
        if self._edge_idx is not None:
            return maybe_num_nodes(self._edge_idx)
        return None

    @property
    def x(self):
        """
        Returns the property of x.
        """
        return self._x

    @property
    def edge_idx(self):
        """
        Returns the property of edge_idx.
        """
        return self._edge_idx

    @property
    def edge_wht(self):
        """
        Returns the property of edge_wht.
        """
        return self._edge_wht

    @property
    def gt(self):
        """
        Returns the property of gt.
        """
        return self._gt

    @property
    def xyz(self):
        """
        Returns the property of xyz.
        """
        return self._xyz

    @property
    def face(self):
        """
        Returns the property of face.
        """
        return self._face

    @property
    def age(self):
        """
        Returns the property of age.
        """
        return self._age

    @property
    def sx(self):
        """
        Returns the property of sx.
        """
        return self._sx

    @property
    def lab(self):
        """
        Returns the property of lab.
        """
        return self._lab

    @x.setter
    def x(self, x):
        self._x = x

    @edge_idx.setter
    def edge_idx(self, edge_idx):
        self._edge_idx = edge_idx

    @edge_wht.setter
    def edge_wht(self, edge_wht):
        self._edge_wht = edge_wht

    @gt.setter
    def gt(self, gt):
        self._gt = gt

    @xyz.setter
    def xyz(self, xyz):
        self._xyz = xyz

    @face.setter
    def face(self, face):
        self._face = face

    @age.setter
    def age(self, age):
        self._age = age

    @sx.setter
    def sx(self, sx):
        self._sx = sx

    @lab.setter
    def lab(self, lab):
        self._lab = lab

    def apply(self, func, *keys):
        for key, item in self(*keys):
            self[key] = func(item)
        return self

    def contiguous(self, *keys):
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys):
        return self.apply(lambda x: x.to(device), *keys)

    def __repr__(self):
        info = ['{}={}'.format(key, list(item.size())) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))


    '''

    def update(self, sample):
        """
        Update an existing sample from another Sample.
        Args:
            sample (:obj:`samitorch.inputs.sample.Sample`): Takes the properties of this Sample to update an existing
                Sample.
        Returns:
            :obj:`samitorch.inputs.sample.Sample`: The updated Sample.
        """
        self._x = sample.x
        self._edge_idx = sample.edge_idx
        self._edge_wht = sample.edge_wht
        self._gt = sample.gt
        self._xyz = sample.xyz
        self._face = sample.face
        self._age = sample.age
        self._sx = sample.sx
        self._lab = sample.lab

        return self

    
    def unpack(self) -> tuple:
        """
        Unpack a Sample.
        Returns:
            tuple: A Tuple of elements representing the (X, y) properties of the Sample.
        """
        return self._x, self._y
    

    @classmethod
    def from_sample(cls, sample):
        """
        Create a new Sample from an existing Sample passed in parameter.
        Args:
            sample (:obj:`samitorch.inputs.sample.Sample`): A template Sample.
        Returns:
            :obj:`samitorch.inputs.sample.Sample`: A new Sample object with same properties as the one passed in
                parameter.
        """
        return cls(sample.x, sample.edge_idx, sample.edge_wht, sample.gt, sample.xyz, sample.face, sample.age,
                   sample.sx,
                   sample.lab)
                   
    '''
