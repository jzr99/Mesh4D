from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import point_cloud_utils as pcu
from Anymate.utils.loss_utils import chamfer_distance_with_average, cross_entropy_with_probs_batch, cos_loss, cos_loss_clamp
from ThirdParty.Rignet_utils.utils import get_skel
from ThirdParty.Rignet_utils.Rignet_loss import edit_dist, chamfer_dist, joint2bone_chamfer_dist, bone2bone_chamfer_dist
from scipy.optimize import linear_sum_assignment

def evaluate_joint(joints, joints_gt, threshold=1e-1):
    """
    joints: list of predicted joints: tensor of shape (n,joints_num,3)
    joints_gt: list of ground truth joints : tensor of shape (n,joints_num,3)
    """
    chamfer_loss_all = 0
    emd_loss_all = 0
    precision = 0
    recall = 0
    count = 0

    for i in tqdm(range(len(joints))):
        joint_predict = joints[i].cpu()
        joint_gt = joints_gt[i].cpu()
        distance_matrix = torch.cdist(joint_gt, joint_predict) # (n_gt, n_predict)
        n_gt,n_predict = distance_matrix.shape
        min_distance_pred = torch.min(distance_matrix, dim=0)
        min_distance_gt = torch.min(distance_matrix, dim=1)
        precision += torch.sum(min_distance_pred.values < threshold).item()/n_predict
        recall += torch.sum(min_distance_gt.values < threshold).item()/n_gt

        chamfer_loss_all += chamfer_distance_with_average(joint_predict.unsqueeze(0), joint_gt.unsqueeze(0))
        joint_predict = joint_predict.numpy().astype(np.float64)
        joint_gt = joint_gt.numpy().astype(np.float64)
        emd,_ = pcu.earth_movers_distance(joint_predict, joint_gt)
        emd_loss_all += emd

        count += 1
    
    print('------------------------------------')
    print('Evaluation results for joint:')
    print('chamfer_loss:', chamfer_loss_all/count)
    print('emd_loss:', emd_loss_all/count)
    print('precision:', precision/count)
    print('recall:', recall/count)
    print('count:', count)
    print('------------------------------------')
    return chamfer_loss_all/count, emd_loss_all/count, precision/count, recall/count

def evaluate_connectivity(conns, conns_gt, joints_gt, vox_list):
    
    """
    conns: list of predicted connections probability: tensor of shape (n,joints_num,joints_num)
    conns_gt: list of ground truth connections: tensor of shape (n,joints_num,joints_num)
    """
    
    precision_all = 0
    recall_all = 0
    cross_entropy_all = 0
    bone2bone_dist_con = 0
    count = 0
    for i in tqdm(range(len(conns))):

        conn_predict = conns[i].cpu().numpy()
        conn_gt = conns_gt[i].cpu().numpy()
        joints = joints_gt[i].cpu().numpy()
        vox = vox_list[i]
        
        cross_entropy_all += cross_entropy_with_probs_batch(torch.from_numpy(conn_predict).unsqueeze(0), torch.from_numpy(conn_gt).unsqueeze(0), reduction='mean')
        # consider to add tree edit distance (need joint and vox information)
        pred_skel, parent_matrix = get_skel(joints, conn_predict, vox=vox)
        gt_skel, parent_matrix = get_skel(joints, conn_gt, vox=vox)
        bone2bone_dist_con += bone2bone_chamfer_dist(pred_skel, gt_skel)
        
        conn_predict = np.argmax(conn_predict, axis=1)
        conn_gt = np.argmax(conn_gt, axis=1)
        connection_matrix_pre = torch.zeros((len(conn_predict),len(conn_predict)))
        connection_matrix_gt = torch.zeros((len(conn_predict),len(conn_predict)))
        
        for i in range(len(conn_predict)):
            connection_matrix_pre[i][conn_predict[i]] = 1
            connection_matrix_pre[conn_predict[i]][i] = 1
            connection_matrix_gt[i][conn_gt[i]] = 1
            connection_matrix_gt[conn_gt[i]][i] = 1

        TP = 0
        FP = 0
        FN = 0
        FP = 0

        for i in range(len(conn_predict)):
            if connection_matrix_gt[i][conn_predict[i]] == 1:
                TP += 1
            if connection_matrix_gt[i][conn_predict[i]] == 0:
                FP += 1
            if connection_matrix_pre[i][conn_gt[i]] == 0:
                FN += 1
            
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        
        precision_all += precision
        recall_all += recall
        count+=1
    print('------------------------------------')
    print('Evaluation results for connectivity:')
    print('precision:',precision_all/count)
    print('recall:',recall_all/count)
    print('cross_entropy:',cross_entropy_all/count)
    print('bone2bone_dist_con:',bone2bone_dist_con/count)
    print('count:',count)
    print('------------------------------------')
    return precision_all/count, recall_all/count

def evaluate_skinning(skins, skins_gt, threshold=5e-2):
    """
    skins: list of predicted skinning weights: tensor of shape (n,vertices_num, bones_num)
    skins_gt: list of ground truth skinning weights: tensor of shape (n,vertices_num, bones_num)
    """
    cs_loss = 0
    ce_loss = 0
    cs_loss_clamp = 0
    count = 0
    L1_loss = 0
    precision = 0
    recall = 0
    mean_l1_dist = 0
    
    for i in tqdm(range(len(skins))):
        skin_predict = skins[i].cpu().unsqueeze(0)
        skin_gt = skins_gt[i].cpu().unsqueeze(0)
        
        precision_one = 0
        recall_one = 0
            
        ce_loss += cross_entropy_with_probs_batch(skin_predict, skin_gt)
        cs_loss += cos_loss(skin_predict, skin_gt)
        cs_loss_clamp += cos_loss_clamp(skin_predict, skin_gt)
        L1_loss += F.l1_loss(skin_predict, skin_gt)
        skin_predict = skin_predict[0].cpu().detach().numpy()
        skin_gt = skin_gt[0].cpu().detach().numpy()
        mean_l1_dist += np.sum(np.abs(skin_predict - skin_gt )) / len(skin_predict)
        
        for i in range(len(skin_predict)):
            influencial_bone_predict = skin_predict[i] >=threshold
            influencial_bone_gt = skin_gt[i] >=threshold
            influencial_bone_correct = influencial_bone_predict*influencial_bone_gt
            
            if np.sum(influencial_bone_predict)==0 or np.sum(influencial_bone_gt)==0:
                continue
            precision_one += np.sum(influencial_bone_correct)/np.sum(influencial_bone_predict)
            recall_one += np.sum(influencial_bone_correct)/np.sum(influencial_bone_gt)
            
        precision += precision_one/len(skin_predict)
        recall += recall_one/len(skin_predict)
        count +=1
    
    print('------------------------------------')
    print('Evaluation results for skinning:')
    print('cos loss: ', cs_loss/count)
    print('ce loss: ', ce_loss/count)
    print('cs_loss_clamp: ', cs_loss_clamp/count)
    print('L1 loss: ', L1_loss/count)
    print('mean_l1_dist: ', mean_l1_dist/count)
    print('precision: ', precision/count)
    print('recall: ', recall/count)
    print('count: ', count)
    print('------------------------------------')
     
def evaluate_skeleton(joints,joints_gt,conns,conns_gt,vox_list,fs_threshold=0.2):
    
    """
    joints: list of predicted joints: tensor of shape (n,joints_num,3)
    joints_gt: list of ground truth joints : tensor of shape (n,joints_num,3)
    conns: list of predicted connections probability: tensor of shape (n,joints_num,joints_num)
    conns_gt: list of ground truth connections: tensor of shape (n,joints_num,joints_num)
    vox_list: list of voxel: (n,88,88,88)
    """
    
    data_count = 0
    chamfer_score = 0
    j2b_chamfer_joint = 0
    bone2bone_dist_joint = 0
    edit_distance_joint = 0
    joint_IoU_total = 0
    joint_precision_total = 0
    joint_recall_total = 0

    for i in tqdm(range(len(joints))):
        joint_predict = joints[i].cpu().numpy()
        joint_gt = joints_gt[i].cpu().numpy()
        conn_predict = conns[i].cpu().numpy()
        conn_gt = conns_gt[i].cpu().numpy()
        vox = vox_list[i]
        
        # add shape diameter after we have vertex and faces
        # shape_diameter = get_shape_diameter(mesh, points, parent_index[:,0])

        dist_matrix = np.sqrt(np.sum((joint_predict[np.newaxis, ...] - joint_gt[:, np.newaxis, :]) ** 2, axis=2))
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        # fs_threshold = shape_diameter[row_ind]
        joint_IoU = 2 * np.sum(dist_matrix[row_ind, col_ind] < fs_threshold) / (len(joint_predict) + len(joint_gt))
        joint_IoU_total += joint_IoU
        joint_precision = np.sum(dist_matrix[row_ind, col_ind] < fs_threshold) / len(joint_predict)
        joint_precision_total += joint_precision
        joint_recall = np.sum(dist_matrix[row_ind, col_ind] < fs_threshold) / len(joint_gt)
        joint_recall_total += joint_recall
        
        pred_skel_joint,parent_matrix = get_skel(joint_predict,conn_predict,vox=vox)
        gt_skel, parent_matrix = get_skel(joint_gt,conn_gt,vox=vox)
        chamfer_score += chamfer_dist(joint_predict, joint_gt)
        j2b_chamfer_joint += joint2bone_chamfer_dist(pred_skel_joint, gt_skel)
        bone2bone_dist_joint += bone2bone_chamfer_dist(pred_skel_joint, gt_skel)
        edit_distance_joint += edit_dist(pred_skel_joint, gt_skel)
        data_count+=1

    print('------------------------------------')
    print('Evaluation results for skeleton:')
    print('chamfer_score:', chamfer_score/data_count)
    print('j2b_chamfer_joint:', j2b_chamfer_joint/data_count)
    print('bone2bone_dist_joint:', bone2bone_dist_joint/data_count)
    print('joint_IoU:', joint_IoU_total/data_count)
    print('joint_precision:', joint_precision_total/data_count)
    print('joint_recall:', joint_recall_total/data_count)
    print('------------------------------------')