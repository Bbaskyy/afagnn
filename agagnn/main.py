import torch
import utils
import time
import numpy as np
import torch.nn.functional as F
import scipy.io
from tnse import visualize
from model import AFAGNN


if __name__ == '__main__':
    dataset_name = 'squirrel'
    # dataset_name = 'Cora'

    heter_dataset = ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']
    homo_dataset = ['Cora', 'Citeseer', 'Pubmed']
    num_hidden = 32
    dropout = 0.5
    eps = 0.3
    layer_num = 2
    lr = 0.01
    weight_decay = 5e-5
    max_epoch = 500
    patience = 200


    re_generate_train_val_test = True
    split_by_label_flag = True
    if dataset_name in ['chameleon', 'cornell', 'texas']:
        split_by_label_flag = False



    if dataset_name in heter_dataset:
        data, num_features, num_classes = utils.load_heter_data(dataset_name)
    elif dataset_name in homo_dataset:
        dataset = utils.load_homo_data(dataset_name)
        data = dataset[0]
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    else:
        print("We do not have {} dataset right now.".format(dataset_name))

    utils.set_seed(15)

    if re_generate_train_val_test:
        idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.6, 0.2, 0.2, 15, split_by_label_flag)
    else:
        if dataset_name in heter_dataset:
            idx_train, idx_val, idx_test = utils.split_nodes(data.y, 0.6, 0.2, 0.2, 15, split_by_label_flag)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')


    data = data.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)



    net = AFAGCN(data, num_features, num_hidden, num_classes, dropout, eps, layer_num)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0



    for epoch in range(max_epoch):
        if epoch >= 3:
            t0 = time.time()
        net.train()
        logp,fea = net(data.x)


        cla_loss = F.nll_loss(logp[idx_train], data.y[idx_train])
        loss = cla_loss
        train_acc = utils.accuracy(logp[idx_train], data.y[idx_train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp,fea = net(data.x)
        test_acc = utils.accuracy(logp[idx_test], data.y[idx_test])
        loss_val = F.nll_loss(logp[idx_val], data.y[idx_val]).item()
        val_acc = utils.accuracy(logp[idx_val], data.y[idx_val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= patience and dataset_name in homo_dataset:
            print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
            epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

    # net.save_g_to_mat(f'{dataset_name}_kernel.mat') #保存edge

    # visualize(fea.cpu().detach().numpy(),data.cpu().y.detach().numpy(),f"{dataset_name}")



    def disedge(edge_index,node_labels):

        same_class_edges = []
        different_class_edges = []

        for i in range(edge_index.size(1)):
            src = edge_index[0][i]
            dst = edge_index[1][i]

            if node_labels[src] == node_labels[dst]:
                same_class_edges.append(i)
            else:
                different_class_edges.append(i)

        kernel = scipy.io.loadmat(f'{dataset_name}_kernel.mat')  # 假设原始数据保存在 original_data.mat 中
        same_class_kernel = {}  # 存储同类边数据的字典
        different_class_kernel= {}  # 存储异类边数据的字典

        for key, value in kernel.items():
            same_class_kernel[key] = value[same_class_edges] if isinstance(value, np.ndarray) else value
            different_class_kernel[key] = value[different_class_edges] if isinstance(value, np.ndarray) else value


        # 保存分割后的数据为两个.mat文件
        scipy.io.savemat(f'{dataset_name}_same_class_kernel.mat', same_class_kernel)  # 保存同类边数据为 same_class_data.mat
        scipy.io.savemat(f'{dataset_name}_different_class_data.mat', different_class_kernel)  # 保存异类边数据为 different_class_data.mat


    # disedge(data.edge_index,data.y)

    # if dataset_name in homo_dataset or 'syn' in dataset_name:
    #     los.sort(key=lambda x: x[1])
    #     print(los)
    #     acc = los[0][-1]
    #     print(acc)
    # else:
    #     los.sort(key=lambda x: -x[2])
    #     acc = los[0][-1]
    #     print(acc)
