import traceback

import torch
from torch import nn


def loss_function_old(X, y, model):
    try:
        if torch.cuda.is_available():
            model = model.cuda()
        predictY = model(X)

        criterion = nn.L1Loss()
        loss = criterion(predictY, y)

        # predictY = predictY.reshape(-1)
        # loss = math.sqrt(((predictY - y.reshape(-1)) ** 2).mean())
        # loss.backward()

    except Exception as e:
        traceback.print_exc()
        predictY = predictY.reshape(-1)

        loss = ((predictY - y.reshape(-1)) ** 2).mean()
        # print("loss="+str(loss))
    return loss


def loss_function(X, y, model):
    """

    :param X:
    :param y:
    :param model:
    :return: loss的值，并且转换为cpu上的
    """
    global predictY
    try:
        if torch.cuda.is_available():
            model = model.cuda()
        predictY = model(X)

        criterion = nn.L1Loss()
        loss = criterion(predictY, y)

        # predictY = predictY.reshape(-1)
        # loss = math.sqrt(((predictY - y.reshape(-1)) ** 2).mean())
        # loss.backward()

    except Exception as e:
        traceback.print_exc()
        print(e)
        predictY = predictY.reshape(-1)

        loss = ((predictY - y.reshape(-1)) ** 2).mean()
    return loss.data.cpu()

