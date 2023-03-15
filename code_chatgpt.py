import matplotlib.pyplot as plt

# 生成一些示例数据
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# 创建一个新的绘图
fig, ax = plt.subplots()

# 绘制数据
ax.plot(x, y)

# 去掉坐标轴的边框
for spine in ax.spines.values():
    spine.set_visible(False)

# 显示图形
plt.show()
