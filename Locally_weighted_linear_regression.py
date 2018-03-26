# _*_ coding: utf-8 _*_  
import math
x = [(1, 0., 3), (1, 1., 3), (1, 3., 2), (1, 4., 4)]
y = [10, 12, 13, 21]
x0 = [1, 1, 1] # The pointer we need to predict
eps = 0.0001

alpha = 0.1
diff = [0, 0]
max_itor = 1000
error1 = 0
error0 = 0
cnt = 0
m = len(x)
weight = 0
u = 10

theta0 = 0
theta1 = 0
theta2 = 0

while True:
	cnt += 1

	for i in range(m):
		weight = math.exp(- ((x0[0] - x[i][0])**2 + (x0[1] - x[i][1])**2 + (x0[2] - x[i][2])**2) / (2*u))
		diff[0] = ((theta0 + theta1 * x[i][1] + theta2 * x[i][2]) - y[i]) * weight
		theta0 -= alpha * diff[0]
		theta1 -= alpha * diff[0] * x[i][1]
		theta2 -= alpha * diff[0] * x[i][2]
	error1 = 0
	for lp in range(m):
		error1 += (y[lp]-(theta0 + theta1 * x[lp][1] + theta2 * x[lp][2]))**2/2  
	if abs(error1 - error0) < eps:
		break
	else:
		error0 = error1
	print(' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f' % (theta0, theta1, theta2, error1))  
	print('Done: theta0 : %f, theta1 : %f, theta2 : %f' % (theta0, theta1, theta2))
	print('迭代次数: %d' % cnt) 