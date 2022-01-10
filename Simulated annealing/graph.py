import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from mpl_toolkits import mplot3d
f = open("iterations.txt","r")
iterations = f.read()
iterations = iterations.split(" ")

f = open("cost.txt", "r")
cost = f.read()
cost = cost.split(" ")

f = open("kappa.txt", "r")
kappa = f.read()
kappa = kappa.split(" ")

f = open("rho.txt", "r")
rho = f.read()
rho = rho.split(" ")

f = open("sigma.txt", "r")
sigma = f.read()
sigma = sigma.split(" ")

f = open("v0.txt", "r")
v0 = f.read()
v0 = v0.split(" ")

f = open("vbar.txt", "r")
vbar = f.read()
vbar = vbar.split(" ")

f = open("costdifinit.txt", "r")
costdifini = f.read()
costdifini = costdifini.split()

f = open("costdif.txt", "r")
costdif = f.read()
costdif = costdif.split()

iterationsN = []
costN = []
kappaN = []
rhoN = []
sigmaN = []
v0N = []
vbarN = []

for i in range(0, len(iterations)):
    if iterations[i] != '':
        iterationsN.append(int(iterations[i]))
        costN.append(math.log10(float(cost[i])))
        kappaN.append(math.log10(float(kappa[i])))
        rhoN.append(math.log10(float(rho[i])))
        sigmaN.append(math.log10(float(sigma[i])))
        v0N.append(math.log10(float(v0[i])))
        vbarN.append(math.log10(float(vbar[i])))

karr = [[0.9371, 0.8603, 0.8112, 0.7760, 0.7470, 0.7216, 0.6699, 0.6137],
        [0.9956, 0.9868, 0.9728, 0.9588, 0.9464, 0.9358, 0.9175, 0.9025],
        [1.0427, 1.0463, 1.0499, 1.0530, 1.0562, 1.0593, 1.0663, 1.0766],
        [1.2287, 1.2399, 1.2485, 1.2659, 1.2646, 1.2715, 1.2859, 1.3046],
        [1.3939, 1.4102, 1.4291, 1.4456, 1.4603, 1.4736, 1.5005, 1.5328]]

tarr = [[0.119047619047619, 0.238095238095238,	0.357142857142857, 0.476190476190476,	0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143],
        [0.119047619047619	,0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714	,1.07142857142857, 1.42857142857143],
        [0.119047619047619, 	0.238095238095238,	0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143],
        [0.119047619047619,	0.238095238095238,	0.357142857142857,	0.476190476190476	,0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143],
        [0.119047619047619,	0.238095238095238	,0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143]]
costdifini1d =[float(costdifini[0])]
costdifini2d = []
for i in range(0, len(costdifini)):
    if i % 8 == 0:
        costdifini2d.append(costdifini1d)
        costdifini1d = []

    costdifini1d.append(float(costdifini[i]))
costdifini2d.append(costdifini1d)
costdifini2d.pop(0)

costdifini2d = np.array(costdifini2d)
karr = np.array(karr)
tarr = np.array(tarr)
print(costdifini2d)
print(karr.shape)
fig = plt.figure(figsize=[10, 8])
ax = plt.axes(projection='3d')
ax.scatter(tarr, karr, costdifini2d,cmap='viridis', edgecolor='none', marker="o", color="blue")
ax.set_title("Initial parameters")
ax.set_xlabel("T")
ax.set_ylabel("K")
ax.set_zlabel("C(theta*, K, T) - C(theta, K, T)")
for i in range(0, len(karr)):
    for j in range(0,len(karr[0])):
        print("sdasd")
        ax.plot([tarr[i][j], tarr[i][j]],[karr[i][j], karr[i][j]], [0, costdifini2d[i][j]], color = 'blue')
plt.savefig("costdifinit.png", bbox_inches='tight')



costdif1d =[float(costdif[0])]
costdif2d = []
for i in range(0, len(costdif)):
    if i % 8 == 0:
        costdif2d.append(costdif1d)
        costdif1d = []

    costdif1d.append(float(costdif[i]))
costdif2d.append(costdif1d)
costdif2d.pop(0)
costdif2d = np.array(costdif2d)

fig = plt.figure(figsize=[10, 8])
ax = plt.axes(projection='3d')
ax.scatter(tarr, karr, costdif2d,cmap='viridis', edgecolor='none', marker="o", color="blue")
ax.set_title("Final parameters")
ax.set_xlabel("T")
ax.set_ylabel("K")
ax.set_zlabel("C(theta*, K, T) - C(theta, K, T)")
for i in range(0, len(karr)):
    for j in range(0,len(karr[0])):
        print("sdasd")
        ax.plot([tarr[i][j], tarr[i][j]],[karr[i][j], karr[i][j]], [0, costdif2d[i][j]], color = 'blue')
plt.savefig("costdif.png", bbox_inches='tight')




x = np.array(iterationsN)
y = np.array(costN)
z = np.array(kappaN)
a = np.array(rhoN)
b = np.array(sigmaN)
c = np.array(v0N)
d = np.array(vbarN)

df = pd.DataFrame({"iterationNumber": x, "cost": y, "|kappa - kappa*|/|kappa*|": z, "|rho - rho*|/|rho*|": a, "|sigma - sigma*|/|sigma*|": b, "|v0 - v0*|/|v0*|": c, "|vbar - vbar*|/|vbar*|": d})
# plt.plot(x, y, label="cost")
# plt.plot(x, z, label="|kappa - kappa*|/|kappa*|")
# plt.plot(x, a, label="|rho - rho*|/|rho*|")
# plt.plot(x, b, label="|sigma - sigma*|/|sigma*|")
# plt.plot(x, c, label="|v0 - v0*|/|v0*|")
# plt.plot(x, d, lable="|vbar - vbar*|/|vbar*|")
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

print("done")