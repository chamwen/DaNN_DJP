# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 19:18
# @Author  : wenzhang
# @File    : main_DaNN_DJP.py

import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import djp_mmd, data_loader, DaNN
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

# para of the network
LEARNING_RATE = 0.001  # 0.001
DROPOUT = 0.5
N_EPOCH = 100
BATCH_SIZE = [64, 64]  # bathsize of source and target domain

# para of the loss function
# accommodate small values of MMD gradient compared to NNs for each iteration
GAMMA = 1000  # 1000 more weight to transferability
SIGMA = 1  # default 1


# MMD, JMMD, JPMMD, DJP-MMD
def mmd_loss(x_src, y_src, x_tar, y_pseudo, mmd_type):
    if mmd_type == 'mmd':
        return djp_mmd.rbf_mmd(x_src, x_tar, SIGMA)
    elif mmd_type == 'jmmd':
        return djp_mmd.rbf_jmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'jpmmd':
        return djp_mmd.rbf_jpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'djpmmd':
        return djp_mmd.rbf_djpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)


def model_train(model, optimizer, epoch, data_src, data_tar, y_pse, mmd_type):
    tmp_train_loss = 0
    correct = 0
    batch_j = 0
    criterion = nn.CrossEntropyLoss()
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))

    # print('***********', len(list_src), len(list_tar))
    for batch_id, (x_src, y_src) in enumerate(data_src):
        optimizer.zero_grad()
        x_src, y_src = x_src.detach().view(-1, 28 * 28).to(DEVICE), y_src.to(DEVICE)
        _, (x_tar, y_tar) = list_tar[batch_j]
        x_tar = x_tar.view(-1, 28 * 28).to(DEVICE)
        model.train()
        ypred, x_src_mmd, x_tar_mmd = model(x_src, x_tar)

        # print('x_src: ', x_src.shape, '\t x_tar', x_tar.shape)  # both torch.Size([64, 784])
        loss_ce = criterion(ypred, y_src)
        loss_mmd = mmd_loss(x_src_mmd, y_src, x_tar_mmd, y_pse[batch_id, :], mmd_type)
        pred = ypred.detach().max(1)[1]  # get the index of the max log-probability

        # get pseudo labels of the target
        model.eval()
        pred_pse, _, _ = model(x_tar, x_tar)
        y_pse[batch_id, :] = pred_pse.detach().max(1)[1]

        # get training loss
        correct += pred.eq(y_src.detach().view_as(pred)).cpu().sum()
        loss = loss_ce + GAMMA * loss_mmd

        # error backward
        loss.backward()
        optimizer.step()
        tmp_train_loss += loss.detach()

    tmp_train_loss /= len(data_src)
    tmp_train_acc = correct * 100. / len(data_src.dataset)
    train_loss = tmp_train_loss.detach().cpu().numpy()
    train_acc = tmp_train_acc.numpy()

    tim = time.strftime("%H:%M:%S", time.localtime())
    res_e = '{:s}, epoch: {}/{}, train loss: {:.4f}, train acc: {:.4f}'.format(
        tim, epoch, N_EPOCH, tmp_train_loss, tmp_train_acc)
    tqdm.write(res_e)
    return train_acc, train_loss, model


def model_test(model, data_tar, epoch):
    tmp_test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with tr.no_grad():
        for batch_id, (x_tar, y_tar) in enumerate(data_tar):
            x_tar, y_tar = x_tar.view(-1, 28 * 28).to(DEVICE), y_tar.to(DEVICE)
            model.eval()
            ypred, _, _ = model(x_tar, x_tar)
            loss = criterion(ypred, y_tar)
            pred = ypred.detach().max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(y_tar.detach().view_as(pred)).cpu().sum()
            tmp_test_loss += loss.detach()

        tmp_test_loss /= len(data_tar)
        tmp_test_acc = correct * 100. / len(data_tar.dataset)
        test_loss = tmp_test_loss.detach().cpu().numpy()
        test_acc = tmp_test_acc.numpy()

        res = 'test loss: {:.4f}, test acc: {:.4f}'.format(tmp_test_loss, tmp_test_acc)
    tqdm.write(res)
    return test_acc, test_loss


def main():
    rootdir = "/mnt/xxx/dataset/office_caltech_10/"
    tr.manual_seed(1)
    domain_str = ['webcam', 'dslr']
    X_s = data_loader.load_train(root_dir=rootdir, domain=domain_str[0], batch_size=BATCH_SIZE[0])
    X_t = data_loader.load_test(root_dir=rootdir, domain=domain_str[1], batch_size=BATCH_SIZE[1])

    # train and test
    start = time.time()
    mmd_type = ['mmd', 'jmmd', 'jpmmd', 'djpmmd']
    for mt in mmd_type:
        print('-' * 10 + domain_str[0] + ' -->  ' + domain_str[1] + '-' * 10)
        print('MMD loss type: ' + mt + '\n')
        acc, loss = {}, {}
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        y_pse = tr.zeros(14, 64).long().cuda()

        mdl = DaNN.DaNN(n_input=28 * 28, n_hidden=256, n_class=10)
        mdl = mdl.to(DEVICE)

        # optimization
        opt_Adam = optim.Adam(mdl.parameters(), lr=LEARNING_RATE)

        for ep in tqdm(range(1, N_EPOCH + 1)):
            tmp_train_acc, tmp_train_loss, mdl = \
                model_train(model=mdl, optimizer=opt_Adam, epoch=ep, data_src=X_s, data_tar=X_t, y_pse=y_pse,
                            mmd_type=mt)
            tmp_test_acc, tmp_test_loss = model_test(mdl, X_t, ep)
            train_acc.append(tmp_train_acc)
            test_acc.append(tmp_test_acc)
            train_loss.append(tmp_train_loss)
            test_loss.append(tmp_test_loss)
        acc['train'], acc['test'] = train_acc, test_acc
        loss['train'], loss['test'] = train_loss, test_loss

        # visualize
        plt.plot(acc['train'], label='train-' + mt)
        plt.plot(acc['test'], label='test-' + mt, ls='--')

    plt.title(domain_str[0] + ' to ' + domain_str[1])
    plt.xticks(np.linspace(1, N_EPOCH, num=5, dtype=np.int8))
    plt.xlim(1, N_EPOCH)
    plt.ylim(0, 100)
    plt.legend(loc='upper right')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig(domain_str[0] + '_' + domain_str[1] + "_acc.jpg")
    plt.close()

    # time and save model
    end = time.time()
    print("Total run time: %.2f" % float(end - start))


if __name__ == '__main__':
    main()
