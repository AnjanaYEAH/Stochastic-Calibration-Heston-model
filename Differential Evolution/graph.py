import matplotlib.pyplot as plt

import numpy as np
# from mpl_toolkits import mplot3d
import math
import pandas as pd
# f = open("iterations.txt","r")
# iterations = f.read()

f = open("cost.txt", "r")
cost = f.read()
cost = cost.split()

f = open("kappa.txt", "r")
kappa = f.read()
kappa = kappa.split()

f = open("rho.txt", "r")
rho = f.read()
rho = rho.split()

f = open("sigma.txt", "r")
sigma = f.read()
sigma = sigma.split()

f = open("v0.txt", "r")
v0 = f.read()
v0 = v0.split()

f = open("vbar.txt", "r")
vbar = f.read()
vbar = vbar.split()

i = 0
cost1d = []
cost2d = []

kappa1d = []
kappa2d = []

sigma1d = []
sigma2d = []

rho1d = []
rho2d = []

v01d = []
v02d = []

vbar1d = []
vbar2d = []


# for i in range(1, len(cost)):
#     if cost[i] == "[":
#         cost2d.append(cost1d)
#         cost1d = []
#
#         kappa2d.append(kappa1d)
#         kappa1d = []
#
#         rho2d.append(rho1d)
#         rho1d = []
#
#         v02d.append(v01d)
#         v01d = []
#
#         vbar2d.append(vbar1d)
#         vbar1d = []
#
#         sigma2d.append(sigma1d)
#         sigma1d = []
#         continue
#     if cost[i] == " ":
#         continue
#     if cost[i] == "]":
#         continue
#     cost1d.append(math.log10(float(cost[i])))
#     kappa1d.append(float(kappa[i]))
#     rho1d.append(float(rho[i]))
#     v01d.append(float(v0[i]))
#     vbar1d.append(float(vbar[i]))
#     sigma1d.append(float(sigma[i]))
#
#
#
# iterations2d = []
# for i in range(1, len(cost2d) + 1):
#     iterations2d.append([i] * len(cost2d[0]))
# # print(cost2d)
# # print(len(cost2d))
# # print(iterations2d)
# iterations2d = np.array(iterations2d)
# cost2d  = np.array(cost2d)
#
# kapp2d = np.array(kappa2d)
# fig = plt.figure(figsize=[8, 6])
# ax = plt.axes(projection='3d')
# ax.plot_surface(iterations2d, kappa2d, cost2d,cmap='viridis', edgecolor='none')
# ax.set_xlabel("iteration number")
# ax.set_ylabel("kappa")
# ax.set_zlabel("log10(cost)")
# plt.savefig("kappaimg.png", bbox_inches='tight')
#
#
# rho2d = np.array(rho2d)
# fig = plt.figure(figsize=[8, 6])
# ax = plt.axes(projection='3d')
# ax.plot_surface(iterations2d, rho2d, cost2d,cmap='viridis', edgecolor='none')
# ax.set_xlabel("iteration number")
# ax.set_ylabel("rho")
# ax.set_zlabel("log10(cost)")
# plt.savefig("rhoimg.png", bbox_inches='tight')
#
# v02d = np.array(v02d)
# fig = plt.figure(figsize=[8, 6])
# ax = plt.axes(projection='3d')
# ax.plot_surface(iterations2d, v02d, cost2d,cmap='viridis', edgecolor='none')
# ax.set_xlabel("iteration number")
# ax.set_ylabel("v0")
# ax.set_zlabel("log10(cost)")
# plt.savefig("v0img.png", bbox_inches='tight')
#
# vbar2d = np.array(vbar2d)
# fig = plt.figure(figsize=[8, 6])
# ax = plt.axes(projection='3d')
# ax.plot_surface(iterations2d, vbar2d, cost2d,cmap='viridis', edgecolor='none')
# ax.set_xlabel("iteration number")
# ax.set_ylabel("vbar")
# ax.set_zlabel("log10(cost)")
# plt.savefig("vbarimg.png", bbox_inches='tight')
#
# sigma2d = np.array(sigma2d)
# fig = plt.figure(figsize=[8, 6])
# ax = plt.axes(projection='3d')
# ax.plot_surface(iterations2d, sigma2d, cost2d,cmap='viridis', edgecolor='none')
# ax.set_xlabel("iteration number")
# ax.set_ylabel("sigma")
# ax.set_zlabel("log10(cost)")
# plt.savefig("sigmaimg.png", bbox_inches='tight')

# karr = [[0.9371, 0.8603, 0.8112, 0.7760, 0.7470, 0.7216, 0.6699, 0.6137],
#         [0.9956, 0.9868, 0.9728, 0.9588, 0.9464, 0.9358, 0.9175, 0.9025],
#         [1.0427, 1.0463, 1.0499, 1.0530, 1.0562, 1.0593, 1.0663, 1.0766],
#         [1.2287, 1.2399, 1.2485, 1.2659, 1.2646, 1.2715, 1.2859, 1.3046],
#         [1.3939, 1.4102, 1.4291, 1.4456, 1.4603, 1.4736, 1.5005, 1.5328]]
#
# tarr = [[0.119047619047619, 0.238095238095238,	0.357142857142857, 0.476190476190476,	0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143],
#         [0.119047619047619	,0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714	,1.07142857142857, 1.42857142857143],
#         [0.119047619047619, 	0.238095238095238,	0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143],
#         [0.119047619047619,	0.238095238095238,	0.357142857142857,	0.476190476190476	,0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143],
#         [0.119047619047619,	0.238095238095238	,0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143]]
#
# f = open("costdif.txt", "r")
# costdif = f.read()
# costdif = costdif.split()
# costdif1d =[float(costdif[0])]
# costdif2d = []
# for i in range(0, len(costdif)):
#     if i % 8 == 0:
#         costdif2d.append(costdif1d)
#         costdif1d = []
#
#     costdif1d.append(float(costdif[i]))
# costdif2d.append(costdif1d)
# costdif2d.pop(0)
#
# karr = np.array(karr)
# tarr = np.array(tarr)
# costdif2d = np.array(costdif2d)
#
# fig = plt.figure(figsize=[10, 8])
# ax = plt.axes(projection='3d')
# ax.scatter(tarr, karr, costdif2d,cmap='viridis', edgecolor='none', marker="o", color="blue")
# ax.set_title("Final parameters")
# ax.set_xlabel("T")
# ax.set_ylabel("K")
# ax.set_zlabel("C(theta*, K, T) - C(theta, K, T)")
# for i in range(0, len(karr)):
#     for j in range(0,len(karr[0])):
#         ax.plot([tarr[i][j], tarr[i][j]],[karr[i][j], karr[i][j]], [0, costdif2d[i][j]], color = 'blue')
# plt.savefig("costdif.png", bbox_inches='tight')

f = open("bestCost.txt", "r")
bestCost = f.read()
bestCost = bestCost.split()
bestCost.pop(0)

f = open("bestAgent.txt", "r")
bestAgent = f.read()
bestAgent = bestAgent.split()
bestAgent1d = []
bestAgent2d = []

for i in range(0, len(bestAgent)):
    if bestAgent[i] == "[":
        continue
    if bestAgent[i] == " ":
        continue
    if bestAgent[i] == "]":
        bestAgent2d.append(bestAgent1d)
        bestAgent1d = []
        continue
    bestAgent1d.append(math.log10(float(bestAgent[i])))
bestAgent2d.pop(0)
a = []
b = []
c = []
d = []
e = []
bestCostN = []
xvals = []

for i in range(0, len(bestAgent2d)):
    a.append(bestAgent2d[i][0])
    b.append(bestAgent2d[i][1])
    c.append(bestAgent2d[i][2])
    d.append(bestAgent2d[i][3])
    e.append(bestAgent2d[i][4])
    bestCostN.append(math.log10(float(bestCost[i])))
    xvals.append(float(i))
print(len(bestCostN))
print(len(xvals))


a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)
e = np.array(e)
bestCostN = np.array(bestCostN)
x = np.array(xvals)
df = pd.DataFrame({"iterationNumber": xvals, "cost": bestCostN, "|kappa - kappa*|/|kappa*|": a, "|rho - rho*|/|rho*|": b, "|sigma - sigma*|/|sigma*|": c, "|v0 - v0*|/|v0*|": d, "|vbar - vbar*|/|vbar*|": e})
plt.plot('iterationNumber', 'cost', data=df)
plt.plot('iterationNumber', '|kappa - kappa*|/|kappa*|', data=df)
plt.plot('iterationNumber', '|rho - rho*|/|rho*|', data=df)
plt.plot('iterationNumber', '|sigma - sigma*|/|sigma*|', data=df)
plt.plot('iterationNumber', '|v0 - v0*|/|v0*|', data=df)
plt.plot('iterationNumber', '|vbar - vbar*|/|vbar*|', data=df)


plt.xlabel("Iteration number")
plt.ylabel("log10(*)")
plt.legend(loc="lower left")
plt.show()
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
#
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
