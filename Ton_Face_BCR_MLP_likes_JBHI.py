# Copyright (c) [2012]-[2021] MJZ@.

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from timm.models import *
from timm.models import create_model

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import datetime
from load_data import get_FaceDL_CON, get_FaceDN_CON, get_TongueDL_CON, get_TongueDN_CON
from load_data import get_FaceConstitution, get_TonConstitution

from torch.optim import lr_scheduler
import time
import copy
from sklearn.metrics import roc_auc_score, roc_curve, auc
from FocalLoss import AsymmetricLoss, FocalLoss, myBFLoss, wDice_loss, ASL_Loss, myGLoss, meanGLoss, BinaryFocalLoss
from sklearn.metrics import roc_auc_score, roc_curve, auc

from Ev_mAPs import mAP_eval, AP_partial, AverageMeter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
# parser.add_argument('--dataset', type=str, default='multilabel_tongue_dataset_ALSwavemlp_1029', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
parser.add_argument('--dataset', type=str, default='multilabel_face_dataset_BCE_HWmlp_1229', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98

parser.add_argument('--b', type=int, default=64, help='batch size')
parser.add_argument('--img-size', type=int, default=224, metavar='N', help='Image patch size (default: None => model default)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0     # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomCrop(args.img_size, padding=(args.img_size//8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset=='cifar10':
    # print('Please use cifar10 or cifar100 dataset.')
    # ---------------- 加载新的数据集 -----------------
    root = "./cifar10"
    args.num_classes = 10
    DATASET_NAME = "cifar10"
    # trainloader, val, testloader = data_loader(root, DATASET_NAME, batch_size=args.b)
else:   # 多标签分类数据集加载
    # ================== face data ================
    # train_ds, test_ds = get_FaceDiseasePosition(transform_train, transform_test)
    # train_ds, test_ds = get_FaceDiseaseSymptom(transform_train, transform_test)
    
    # train_ds, test_ds = get_FaceConstitution(transform_train, transform_test)
    train_ds, test_ds = get_TonConstitution(transform_train, transform_test)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=args.b, shuffle=True,num_workers=8)
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=args.b, num_workers=8) 
    
    # ================ tongue data ================
    # train_ds, test_ds = get_TongueDiseasePosition(transform_train, transform_test)
    # train_ds, test_ds = get_TongueDiseaseSymptom(transform_train, transform_test)

    # trainloader = torch.utils.data.DataLoader(train_ds, batch_size=args.b, shuffle=True,num_workers=8)
    # testloader = torch.utils.data.DataLoader(test_ds, batch_size=args.b, num_workers=8)  
    
    # ============= Face & tongue 联合数据DN &DL 2023.08.24========
    # train_ds, test_ds = get_FaceDL_CON(transform_train, transform_test)
    # train_ds, test_ds = get_FaceDN_CON(transform_train, transform_test)

    # train_ds, test_ds = get_TongueDL_CON(transform_train, transform_test)
    # train_ds, test_ds = get_TongueDN_CON(transform_train, transform_test)


    # trainloader = torch.utils.data.DataLoader(train_ds, batch_size=args.b, shuffle=True,num_workers=8)
    # testloader = torch.utils.data.DataLoader(test_ds, batch_size=args.b, num_workers=8)    
  

# print(f'learning rate:{args.lr}, weight decay: {args.wd}')

# === Training position and disease of mutil-label classes task ===
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # print(model)
    Sigmoid_fun = nn.Sigmoid()
    since = time.time()
    best_acc = 0.81  # 开始保存模型的Acc,face-->0.81;ton-->0.80
    best_map = 0.65  # 开始保存模型的mAP,face-->0.65;ton-->0.41
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = criterion.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # 每训练一个epoch，验证一下网络模型
        for phase in ['train', 'test']:
            running_loss = 0.0
            running_Acc = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0
            running_mAP = 0.0
            running_AUC = 0.0
            result_list = []
            label_list = []
            test_APs = []
            test_F1 = []

            if phase == 'train':
                scheduler.step()    # 学习率更新方式
                model.train()       # 调用模型训练
                # result_list = []
                # label_list = []

                # 依次获取所有图像，参与模型训练或测试
                for inputs, labels in trainloader:
                    # if use_gpu:     # 判断是否使用gpu
                    #     inputs = inputs.cuda()
                    #     labels = labels.cuda()
                    # else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()   # 梯度清零
                    outputs = model(inputs).to(device) # 网络前向运行
                    # print(outputs)
                    loss = criterion(Sigmoid_fun(outputs), labels)   # 计算Loss值
                    # loss = criterion(outputs, labels)   # 计算Loss值

                    # result = nn.Sigmoid()(outputs).cpu().detach().numpy().tolist()  # 经过sigmoid得到各类别概率
                    # target = labels.cpu().detach().numpy().tolist()
                    # for k in range(len(inputs)):
                    #     result_list.append({"scores": result[k]})
                    #     label_list.append({"scores": target[k]})
                    # mAP = calculate_mAP (result=result_list, ann_path=label_list, mode=1)

                    # 模型预测结果准确率的函数
                    precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)
                    micro_Acc = calculate_acuracy_mode_zero(Sigmoid_fun(outputs), labels)
                    
                    running_Acc += micro_Acc
                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
                    # running_mAP += mAP
                    
                    loss.backward()     # 反传梯度
                    optimizer.step()    # 更新权重

                    running_loss += loss.item() * inputs.size(0)    # 计算一个epoch的loss值和准确率
            else:
                with torch.no_grad():   # 取消验证阶段的梯度
                    model.eval()
                    cal_mAPs(testloader, model)
                    for inputs, labels in testloader:
                        # if use_gpu:     # 判断是否使用gpu
                        #     inputs = inputs.cuda()
                        #     labels = labels.cuda()
                        # else:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs).to(device) # 网络前向运行
                        # print(outputs)
                        pred = Sigmoid_fun(outputs).to(device)

                        result = nn.Sigmoid()(outputs).cpu().detach().numpy().tolist()  # 经过sigmoid得到各类别概率
                        target = labels.cpu().detach().numpy().tolist() # 真实标签
                        for k in range(len(inputs)):
                            result_list.append({"scores": result[k]})
                            label_list.append({"scores": target[k]})

                        mAP, APs = calculate_mAP(result=result_list, ann_path=label_list, mode=2)
                        # Auc = calculate_AUC(pred,labels)

                        # BCELoss的输入（1、网络模型的输出必须经过sigmoid；2、标签必须是float类型的tensor）
                        # loss = criterion(outputs, labels)  # 计算Loss
                        loss = criterion(pred, labels) # 计算Loss
                        # 计算each epoch的loss值和准确率
                        running_loss += loss.item() * inputs.size(0)

                        # 模型预测结果准确率的函数
                        micro_Acc = calculate_acuracy_mode_zero(Sigmoid_fun(outputs), labels)
                        precision, recall = calculate_acuracy_mode_one(Sigmoid_fun(outputs), labels)

                        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                        running_Acc += micro_Acc
                        running_precision += precision
                        running_recall += recall
                        batch_num += 1
                        # running_mAP += mAP # 病性病位识别
                        # running_AUC += Auc
                        running_mAP = mAP   # 面部体质辨识
                        test_APs = APs      # 每个体质的准确率
                        test_F1 = f1_score
                    
            # ------------ 计算Loss和准确率的均值 -------------
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f} '.format(phase, epoch_loss))
            epoch_Acc = running_Acc / batch_num
            print('{} Acc: {:.4f} '.format(phase, epoch_Acc))
            epoch_precision = running_precision / batch_num
            print('{} Precision: {:.4f} '.format(phase, epoch_precision))
            epoch_recall = running_recall / batch_num
            print('{} Recall: {:.4f} '.format(phase, epoch_recall))
            # epoch_mAP = running_mAP / batch_num # 病性病位
            epoch_mAP = running_mAP # 体质
            epoch_APs = test_APs
            print('{} mAP: {:.4f} '.format(phase, epoch_mAP))
            print('APs:', epoch_APs)
            print('F1:', test_F1)
            # epoch_AUC = running_AUC / batch_num
            # print('{} AUC: {:.4f} '.format(phase, epoch_AUC))

            if phase == 'train':
                total_train_acc.append(epoch_Acc)
                total_train_loss.append(epoch_loss)
                # total_train_mAP.append(epoch_mAP)
                print("total_trACC:",total_train_acc)
            else:
                total_test_acc.append(epoch_Acc)
                total_test_loss.append(epoch_loss)
                recall_test.append(epoch_recall)
                precision_test.append(epoch_precision)
                total_test_mAP.append(epoch_mAP)
                print("total_teACC:",total_test_acc)

        # if phase == 'test' and epoch_Acc > best_acc:
        if phase == 'test' and epoch_mAP > best_map:    # 增加02
            best_acc = epoch_Acc
            best_map = epoch_mAP    # 增加0305
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Saving model...')
            state = {
                'net': net.state_dict(),
                'acc': epoch_Acc,
                'map': epoch_mAP,   # 增加0305
                'epoch': epoch,
            }
            if not os.path.isdir(f'checkpoint_{args.dataset}_{args.model}'):    # 数据集
                os.mkdir(f'checkpoint_{args.dataset}_{args.model}')
            # torch.save(state, f'./checkpoint_{args.dataset}_{args.model}/ckpt_{epoch_Acc*100}.pth')
            # torch.save(state, f'./checkpoint_{args.dataset}_{args.model}/ckpt_{epoch_Acc*100}.pkl')
            torch.save(state, f'./checkpoint_{args.dataset}_{args.model}/ckpt_{epoch_Acc*100:.2f}_{epoch_mAP:.2f}.pkl')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val mAP: {:4f}'.format(best_map))
    
    model.load_state_dict(best_model_wts)   # 网络导入最好的网络权重

    return model

# 计算准确率Acc，直接计算预测正确的标签数
def calculate_acuracy_mode_zero(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测该类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th   # 网络输出的结果model_pred与0.5比较，超过0.5判为true
    pred_result = pred_result.float()        # 将true、false改为1、0 ，此为预测结果
    
    compare = pred_result.eq(labels).float() # 比较预测与实际是否相同，相同为true，不同为false，改为1、0
    # print(compare.shape)
    num = torch.sum(compare)    # 判断正确的标签个数（num为tensor格式）
    # print(num)
    ma_l, ma_w = labels.shape   # 标签size为(64,12) = (batch_sieze, label)
    # print(labels.shape)
    ma_size = ma_l * ma_w       # 总数为64*12 = batch_sieze * label
    # print(ma_size)
    micro_Acc = num.item() / ma_size  # Acc准确率，即预测正确的标签数，微平均
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0.
    
    return micro_Acc
    
# 计算精准率和召回率，方式1
def calculate_acuracy_mode_one(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    eps = 1e-8
    pred_result = model_pred > accuracy_th  # 网络输出的结果model_pred与0.5比较，超过0.5判为true
    pred_result = pred_result.float()       # 将true、false改为1、0 ，此为预测结果

    # print(labels)
    # print(pred_result)
    # print(pred)

    pred_one_num = torch.sum(pred_result)   # TP+FP
    # print(pred_one_num)

    if pred_one_num == 0:
        return 0., 0.
    
    target_one_num = torch.sum(labels)      # TP+FN
    true_predict_num = torch.sum(pred_result * labels)  # TP
    # print(target_one_num)
    # print(true_predict_num)
    
    precision = true_predict_num / pred_one_num # 模型预测的结果中有多少个是正确的
    recall = true_predict_num / target_one_num  # 模型预测正确的结果中，占所有真实标签的数量
        
    # pred_thresh = (model_pred > accuracy_th)*1.0
    # tp = torch.sum((pred_thresh == labels)*(labels==1.0), dim=0)
    # fp = torch.sum((pred_thresh != labels)*(labels==0.0), dim=0)
    # fn = torch.sum((pred_thresh != labels)*(labels==1.0), dim=0)
    # recall = tp / (fn + tp + eps)
    # precision = tp / (fp + tp + eps)

    f1_score = 2 * precision * recall / (precision + recall + eps)

    return precision.item(), recall.item()#, f1_score.item()


# ========================计算mAP指标=====================
def calculate_mAP (result, ann_path, mode):
    max_mAP = 0.0
    # classes = ['0','1','2','3','4','5','6','7','8','9','10','11']  # 12个类/14个类
    # classes = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13']  # 12个类/14个类
    classes = ['0','1','2','3','4','5','6','7','8'] # 9类体质

    aps = np.zeros(len(classes), dtype=np.float64)

    pred_json= result      # 预测标签值
    ann_json = ann_path    # 实际标签

    for i, _ in enumerate(classes):     # 列举多个类别
        ap = json_map(i, pred_json, ann_json)
        aps[i] = ap
    mAP = np.mean(aps)
    # print("AP:",aps)    # 输出每个标签的准确率,自动叠加至测试样本数量

    if mode == 1:    # training stage
        print("mAP: {:4f}".format(mAP))
    elif mode == 2:  # testing stage
        if mAP > max_mAP:
            max_mAP = mAP
            # print("mAP: {:4f}".format(mAP), "max_mAP: {:4f}".format(max_mAP))
        else:
            print("mAP: {:4f}".format(mAP))

    return 100 * mAP.item(), 100 * aps

def json_map(cls_id, pred_json, ann_json):  # 对每个标签类cls_id，所有图片求准确率
    assert len(ann_json) == len(pred_json)
    num = len(ann_json)     # batch_size的图片数量
    predict = np.zeros(num, dtype=np.float64)
    target = np.zeros(num, dtype=np.float64)

    for i in range(num):    # 对每一张图片
        predict[i] = pred_json[i]["scores"][cls_id] # 预测标签值
        target[i] = ann_json[i]["scores"][cls_id]   # 实际标签

    tmp = np.argsort(-predict)  # 取预测概率从大到小排序的索引
    # print(tmp)
    target = target[tmp]        # 按预测概率从大到小排序
    # print("target:",target)

    index_id = 1
    if index_id ==0:
    #=========== method one ===========
        pre, obj = 0.0, 0.0
        esp = 1e-8
        for i in range(num):    # 设有M个正样本，则有M个label为1，取M个recall值,从0开始
            if target[i] == 1:  # 每当label为1，就是这个recall值下的precision最大的时候，取M个这个最大pre求平均即ap值
                obj += 1.0      # 0~M
                pre += obj / (i+1)  # 最大precision
        pre += pre              # Sum of the Max precision 
        # print("obj:",obj)
        # print("pre:", pre)
        AP = pre / (obj+ esp)   # 最大pre求平均

    elif index_id ==1:
    #=========== method two ===========
        pos_count = 1e-8
        total_count = 0.0
        pre_at_i = 0.0
        for i in range(num):
            label = target[i]
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                pre_at_i += pos_count / total_count
        # print("obj:", obj)
        # print("pre:", pre_at_i)
        pre_at_i /= pos_count   # 最大pre求平均 
        AP = pre_at_i

    return AP

# ==================== 计算AUC ====================
def calculate_AUC(label, preds):
    fpr = np.zeros((preds.shape[1]))
    tpr = np.zeros((preds.shape[1]))
    auc_score = np.zeros((preds.shape[1]))
    roc_auc = np.zeros((preds.shape[1]))

    for k in range(preds.shape[1]):
        scores = preds[:, k]
        target = label[:, k]

        fpr, tpr, _ = roc_curve(target, scores)
        # print(fpr.shape, tpr.shape)

        roc_auc[k] = auc(fpr, tpr)
        auc_score[k] = roc_auc_score(target, scores)
     
    return auc_score.mean(), roc_auc.mean(), roc_auc

# ============== add parameters ==================
def cal_mAPs(val_loader, model):
    Sig = torch.nn.Sigmoid()

    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    acc = []

    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()
        num_classes = output.shape[-1]

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(0.5).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)
        mic_acc = (tp.sum().float() + tn.sum().float()) / (num_classes*count) * 100.0

    print('==================================================================')
    print(' * P_C {:.4f} R_C {:.4f} F_C {:.4f} P_O {:.4f} R_O {:.4f} F_O {:.4f} Acc {:.4f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o, mic_acc))

    _, auc, auc_ap = calculate_AUC(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("auc:",auc*100,"auc_ap:",auc_ap*100)

    AP_score1, mAP_score1, macro_AP = AP_partial(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("AP_score:",AP_score1*100,"mAP score:", mAP_score1,"macro_AP:",macro_AP )

    return

if __name__ == '__main__':

    dataset_sizes = {'train': train_ds.__len__(), 'test': test_ds.__len__()}
    # use_gpu = torch.cuda.is_available()
    
    model_id = [6]  # 补充不同损失的对比实验，20240705

    
    for id in model_id: # 加载模型id
    # =================== create model =======================
        print("========id:=======",id)
        print('==> Building model..')
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
        parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
        # parser.add_argument('--dataset', type=str, default='multilabel_face_con_BGL200_0305', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='tonguePhoto_Dis04_dataset', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='multilabel_tongue_dataset_ALSwavemlp_1029', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='multilabel_Face_con_meanBGL0.8_0.1_200_0518', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='multilabel_Ton_con_meanBGL0.5_0.1_200_0804', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='multilabel_Face_con_meanBGL1_0.1_200_0803', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # ==== 补充Loss对比实验，20240705 ====
        # parser.add_argument('--dataset', type=str, default='multilabel_Face_con_Losses_20240705', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        parser.add_argument('--dataset', type=str, default='multilabel_Ton_con_Losses_20240705', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98

        # ======= 2023.08.23舌象DN/DL数据集测试 ========
        # parser.add_argument('--dataset', type=str, default='multilabel_Ton_DN_BCGL1_0.2_200_0825', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='multilabel_Face_DN_BCGL1_0.05_200_0827', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98
        # parser.add_argument('--dataset', type=str, default='multilabel_Ton_con_BCE_DFTmultichaos_200_240114', help='facePhoto or tonguePhoto') # ChestXRay,Chinese-Herbs98

        parser.add_argument('--b', type=int, default=64, help='batch size')
        parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        parser.add_argument('--pretrained', action='store_true', default=False, help='Start with pretrained version of specified network (if avail)')
        parser.add_argument('--num-classes', type=int, default=9, metavar='N', help='number of label classes (default: 1000)')
        if id == 1:
            parser.add_argument('--model', default='CycleMLP_B1', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 2:
            parser.add_argument('--model', default='ActivexTiny', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 3:
            parser.add_argument('--model', default='WaveMLP_Tor', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 4:
            parser.add_argument('--model', default='vip_s7', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 5:
            parser.add_argument('--model', default='ChaosMLP_T', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
        elif id == 6:
            parser.add_argument('--model', default='ChaosMLP_T_none', type=str, metavar='MODEL', help='Name of model to train (default: "countception"')
 
        parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.0)')
        parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT', help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
        parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: None)')
        parser.add_argument('--drop-block', type=float, default=None, metavar='PCT', help='Drop block rate (default: None)')
        parser.add_argument('--gp', default=None, type=str, metavar='POOL', help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
        parser.add_argument('--img-size', type=int, default=224, metavar='N', help='Image patch size (default: None => model default)')
        parser.add_argument('--bn-tf', action='store_true', default=False, help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
        parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override (if not None)')
        parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')
        parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH', help='Initialize model from this checkpoint (default: none)')
        # Transfer learning
        parser.add_argument('--transfer-learning', default=False, help='Enable transfer learning')
        parser.add_argument('--transfer-model', type=str, default=None, help='Path to pretrained model for transfer learning')
        parser.add_argument('--transfer-ratio', type=float, default=0.01, help='lr ratio between classifier and backbone in transfer learning')
        parser.add_argument('--seed', type=int, default=42, metavar='S',help='random seed (default: 42)')
        # PartialSelective Loss
        parser.add_argument('--clip', type=float, default=0)
        parser.add_argument('--gamma_pos', type=float, default=0)
        parser.add_argument('--gamma_neg', type=float, default=1)
        parser.add_argument('--gamma_unann', type=float, default=2)
        parser.add_argument('--alpha_pos', type=float, default=1)
        parser.add_argument('--alpha_neg', type=float, default=1)
        parser.add_argument('--alpha_unann', type=float, default=1)
        parser.add_argument('--prior_path', type=str, default=None)
        parser.add_argument('--partial_loss_mode', type=str, default="negative")

        args = parser.parse_args()

        net = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect, 
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            # bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            checkpoint_path=args.initial_checkpoint,
            img_size=args.img_size)


        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        parameters = net.parameters()
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)

        model = net
        # ============= 模型参数计算 ============
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)

        # if use_gpu:
        #     model = model.cuda()
        # else:
        model = model.to(device)
        # loss_id = 0   # 12.22 Loss改进设计;8-->G=1,lamda=0.1,0.2
        # loss_id_con = [1,2,9,3]    # 补充Loss的对比实验，20240705
        loss_id_con = [3]
        for loss_id in loss_id_con:
            print("====loss_id_name:====",loss_id)
            # ===== 增加多标签Loss对比实验，202040705 ====
            if loss_id == 0:
                criterion = nn.BCELoss()     # 定义损失函数, BCE Loss
            elif loss_id == 1:
                criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05) # ASL,改进Focal loss,2-1
            elif loss_id == 2:
                criterion = FocalLoss()      # 没有参数alpha, gamma=2
            elif loss_id == 3:
                criterion = myBFLoss()       # 改进BCE Loss, our
            elif loss_id == 4:
                criterion = wDice_loss()     # 改进Dice_loss, wight = 0 
            elif loss_id == 5:
                criterion = wDice_loss(wight = 1) 
            elif loss_id == 6:
                criterion = myGLoss()    
            elif loss_id == 7:
                criterion = meanGLoss()     # lamda=0.01,G=1(04.28，完成);G=0.01(完成，04.26)；G=0.5（04.29，完成）；G=0.1（05.01，完成）
            elif loss_id == 8:
                criterion = meanGLoss()     # G=1,lamda=0.01(完成，04.26)；lamda=0.05（05.03，完成）；lamda=0.1（05.01，完成）;G=1,lamda=0.1(0824)
                                            # G=1,lamda=0.2(08.25); G=1,lamda=0.05(08.27)
            elif loss_id == 9:
                criterion = BinaryFocalLoss()   # 补充了FL权重alpha = 0.25; 

            # 数据可视化
            total_test_acc = []
            total_train_acc = []

            total_train_loss = []
            total_test_loss = []

            total_test_mAP = []

            recall_test = []
            precision_test = []

            train_model(model, criterion, optimizer, scheduler, num_epochs = 200)
            
            print("total_trLoss:",total_train_loss)
            print("total_teLoss:",total_test_loss)
            print("total_mAP",total_test_mAP)
            print("total_recall",recall_test)
            print("total_precison",precision_test)
