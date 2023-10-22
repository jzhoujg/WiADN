import matplotlib.pyplot as plt

# 准备数据
x = [1, 2, 3, 4, 5]  # x轴数据
y = [2, 4, 6, 8, 10]  # y轴数据

# 创建画布和子图
fig, ax = plt.subplots()

# 绘制折线图
ax.plot(x, y, marker='o')

# 设置标题和轴标签
ax.set_title('title')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 显示图形
plt.show()