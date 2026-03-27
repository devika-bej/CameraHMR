import numpy as np
import torch
import pickle
from smplx import SMPLX as SMPLX_
from smplx.utils import ModelOutput, SMPLOutput
from smplx.lbs import vertices2joints
from smplx import SMPLXLayer as SMPLXLayer_
from .vertex_ids import (
    VertexJointSelector,
    smplx_ids,
    smpl_to_openpose,
    smplx_to_openpose,
    smplx_lr_hand_inds,
    split2face_feet_hand_parts,
    mano_smplx_lhand_vertex_ids,
    mano_smplx_rhand_vertex_ids,
    mano_smplh_lhand_vertex_ids,
    mano_smplh_rhand_vertex_ids,
)
from .constants import smpl_to_openpose, SMPL_to_J19, SMPLX2SMPL, SMPL_MODEL_DIR, JOINT_MAP

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}


class SMPLX(SMPLX_):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)
        import smplx
        self.smpl = smplx.SMPL(model_path=SMPL_MODEL_DIR)
        self.smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32).repeat(1, 1, 1) #Todo provide batch size?
        J_regressor_extra = pickle.load(open(SMPL_to_J19,'rb'))
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))

        J_regressor_smplx2smpljoints_ = torch.mm(self.smpl.J_regressor, self.smplx2smpl[0])
        self.register_buffer('J_regressor_smplx2smpljoints', J_regressor_smplx2smpljoints_ )

        self.J_regressor_extra = torch.mm(self.J_regressor_extra, self.smplx2smpl[0])
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.register_buffer('mano_lhand_vertex_ids', mano_smplx_lhand_vertex_ids)
        self.register_buffer('mano_rhand_vertex_ids', mano_smplx_rhand_vertex_ids)
        self.ffh_joints_selector = VertexJointSelector(vertex_ids=smplx_ids, use_hands=True, use_feet_keypoints=True)


    def forward(self, *args, **kwargs):
        smplx_output = super(SMPLX, self).forward(*args, **kwargs)
        smplx_output_vertices, smplx_output_joints = smplx_output.vertices, smplx_output.joints
        smpl_24_joints = vertices2joints(self.J_regressor_smplx2smpljoints, smplx_output_vertices)
        smpl_output_extra_joints = smplx_output_joints[:,55:76]
        smpl_output_joints = torch.cat([smpl_24_joints, smpl_output_extra_joints],dim=1)
        joints = smpl_output_joints[:, self.joint_map, :]
        extra_joints = vertices2joints(self.J_regressor_extra, smplx_output_vertices)
        joints = torch.cat([joints, extra_joints, smplx_output_joints], dim=1)
        vert_mask = torch.ones(smplx_output.vertices.shape[1], dtype=torch.bool)

        vert_mask[self.mano_lhand_vertex_ids] = False
        vert_mask[self.mano_rhand_vertex_ids] = False

        output = SMPLOutput(vertices=smplx_output_vertices,
                            joints=joints)
        output.lhand_verts = torch.index_select(output.vertices, 1, self.mano_lhand_vertex_ids)
        output.rhand_verts = torch.index_select(output.vertices, 1, self.mano_rhand_vertex_ids)
        output.body_verts = output.vertices[:, vert_mask]

        ffh_joints = torch.cat(
            [
                self.ffh_joints_selector(output.vertices), # last 10 are the finger tips
                smplx_output.joints[:, 25: 55, :], # left and right hand joints
                smplx_output.joints[:, 20: 22, :] # left and right thumb joints
            ], 1)
        
        face_joints, feet_joints, hand_joints = split2face_feet_hand_parts(ffh_joints)
        
        output.body_joints = joints
        output.face_joints = face_joints
        output.feet_joints = feet_joints
        output.hand_joints = hand_joints
        return output

class SMPLXLayer(SMPLXLayer_):
    """ Extension of the official SMPL implementation to support more joints """
    def __init__(self, *args, **kwargs):
        super(SMPLXLayer, self).__init__(*args, **kwargs)
        # smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
        #                     7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        import smplx
        self.smpl = smplx.SMPL(model_path=SMPL_MODEL_DIR)
        self.smplx2smpl = pickle.load(open(SMPLX2SMPL, 'rb'))
        self.smplx2smpl = torch.tensor(self.smplx2smpl['matrix'][None], dtype=torch.float32).repeat(1, 1, 1) #Todo provide batch size?
        J_regressor_extra = pickle.load(open(SMPL_to_J19,'rb'))
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))

        J_regressor_smplx2smpljoints_ = torch.mm(self.smpl.J_regressor, self.smplx2smpl[0])
        self.register_buffer('J_regressor_smplx2smpljoints', J_regressor_smplx2smpljoints_ )
        self.J_regressor_extra = torch.mm(self.J_regressor_extra, self.smplx2smpl[0])
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
        self.register_buffer('mano_lhand_vertex_ids', mano_smplx_lhand_vertex_ids)
        self.register_buffer('mano_rhand_vertex_ids', mano_smplx_rhand_vertex_ids)
        self.ffh_joints_selector = VertexJointSelector(vertex_ids=smplx_ids, use_hands=True, use_feet_keypoints=True)

    def forward(self, *args, **kwargs):

        smplx_output = super(SMPLXLayer, self).forward(*args, **kwargs)
        smplx_output_vertices, smplx_output_joints = smplx_output.vertices, smplx_output.joints
        smpl_24_joints = vertices2joints(self.J_regressor_smplx2smpljoints, smplx_output_vertices)
        smpl_output_extra_joints = smplx_output_joints[:,55:76]
        smpl_output_joints = torch.cat([smpl_24_joints, smpl_output_extra_joints],dim=1)
        joints = smpl_output_joints[:, self.joint_map, :]
        extra_joints = vertices2joints(self.J_regressor_extra, smplx_output_vertices)
        joints = torch.cat([joints, extra_joints, smplx_output_joints], dim=1)
        vert_mask = torch.ones(smplx_output.vertices.shape[1], dtype=torch.bool)

        vert_mask[self.mano_lhand_vertex_ids] = False
        vert_mask[self.mano_rhand_vertex_ids] = False

        output = SMPLOutput(vertices=smplx_output_vertices,
                            joints=joints)
        output.lhand_verts = torch.index_select(output.vertices, 1, self.mano_lhand_vertex_ids)
        output.rhand_verts = torch.index_select(output.vertices, 1, self.mano_rhand_vertex_ids)
        output.body_verts = output.vertices[:, vert_mask]

        ffh_joints = torch.cat(
            [
                self.ffh_joints_selector(output.vertices), # last 10 are the finger tips
                smplx_output.joints[:, 25: 55, :], # left and right hand joints
                smplx_output.joints[:, 20: 22, :] # left and right thumb joints
            ], 1)
        
        face_joints, feet_joints, hand_joints = split2face_feet_hand_parts(ffh_joints)
        
        output.body_joints = smpl_24_joints
        output.face_joints = face_joints
        output.feet_joints = feet_joints
        output.hand_joints = hand_joints
        return output