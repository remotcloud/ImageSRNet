from matplotlib import pyplot as plt
import numpy as np
import os
orignalY = np.loadtxt('../../Dataset/impedance.csv', dtype=float, delimiter=',')
orignalY=np.delete(orignalY,-1,axis=1)
y = np.loadtxt('../../Result/CLGP/SplitLessImage/predictImp/17.txt', dtype=float, delimiter=',')
# 图片保存的路径
figpath=r"../../Result/CGP/SplitLessImage/reDrawFig"
# 图片的文件名
figName = str(17) + ".png"

# 画图并保存
# plt.subplot(3 ,1 ,1)
# plt.imshow(orignalY, vmin=np.min(orignalY), vmax=np.max(orignalY))
# plt.subplot(3 ,1 ,2)
# # y = bestIndividual(X)
# plt.imshow(y, vmin=np.min(orignalY), vmax=np.max(orignalY))
# plt.subplot(3, 1, 3)
# # 现在的
# # temp_y = (orignalY -y)
# # 原来的
# temp_y = (np.abs((orignalY -y)))
# c=plt.imshow(temp_y, vmin=np.min(orignalY), vmax=np.max(orignalY))
# plt.colorbar(c)

# 取二者中的最大的数值
max=np.max(orignalY)
if max < np.max(y):
    max=np.max(y)

plt.subplot(3 ,1 ,1)
fig0=plt.imshow(orignalY, vmin=0, vmax=max)
plt.subplot(3 ,1 ,2)
# y = bestIndividual(X)
fig1=plt.imshow(y, vmin=0, vmax=max)
plt.subplot(3, 1, 3)
# 现在的
# temp_y = (orignalY -y)
# 原来的
temp_y = (np.abs((orignalY -y)))
fig2=plt.imshow(temp_y, vmin=0, vmax=max)
# plt.colorbar(fig1, ax=[fig0,fig1,fig2])
# 保存图片

if not os.path.exists(figpath):
    os.makedirs(figpath)
figName = os.path.join(figpath, figName)
plt.savefig(figName)
plt.close()