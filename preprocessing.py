import os
from collections import defaultdict
import numpy as np

# 纯净版
def xyz_2Dsplit(horizontalAxis: np.ndarray, verticalAxis: np.ndarray, rowH: float, colW: float,
                overlapH: float = 0, overlapW: float = 0, areaBrokenMerge: bool = True, ptBrokenMerge: int = None):
    """
    将二维坐标 (horizontalAxis, verticalAxis) 按照指定的行列大小 (rowH, colW) 进行分割，并允许设置重叠度 (overlapH, overlapW)，以及是否合并分割后不足尺寸的区域。

    Args:
        horizontalAxis: 代表水平方向的坐标（一维数组）
        verticalAxis: 代表垂直方向的坐标（一维数组）
        rowH: 垂直方向的块的高度（每个分块的垂直尺寸）
        colW: 水平方向的块的宽度（每个分块的水平尺寸）
        overlapH: 垂直方向块之间的重叠度，范围在 [0, 1) 之间
        overlapW: 水平方向块之间的重叠度，范围在 [0, 1) 之间
        areaBrokenMerge: 控制是否合并最后不足指定大小的分块。如果为 True，则将剩余的小块合并到前一块
        ptBrokenMerge: 如果提供，则当某个分块包含的点数少于此值时，尝试将其与相邻的分块合并
    Returns:
        split_idx:每个分块包含的点的索引 Location(H,W) & ID[]
    """
    split_idx = defaultdict(list)  # 用于存储每个分块的索引信息

    # 计算 X 轴相关参数
    maxX, minX = horizontalAxis.max(), horizontalAxis.min()
    rangeX = maxX - minX
    stepX = colW * (1 - overlapW)  # 每次步进的长度
    xNums, xExtra = divmod(rangeX, stepX)  # 分割块数与余数

    # 计算 Y 轴相关参数
    maxY, minY = verticalAxis.max(), verticalAxis.min()
    rangeY = maxY - minY
    stepY = rowH * (1 - overlapH)
    yNums, yExtra = divmod(rangeY, stepY)

    # 生成布尔掩码列表
    judgeXresult = [
        (minX + i * stepX <= horizontalAxis) & (horizontalAxis <= np.clip(minX + i * stepX + colW, minX, maxX))
        for i in range(int(xNums) + 1)]
    judgeYresult = [
        (minY + j * stepY <= verticalAxis) & (verticalAxis <= np.clip(minY + j * stepY + rowH, minY, maxY))
        for j in range(int(yNums) + 1)]

    # 使用 np.where 联合判定 X 和 Y
    for n, judgeY in enumerate(judgeYresult):  # 先遍历行
        for m, judgeX in enumerate(judgeXresult):  # 再遍历列
            valid_idx = np.where(judgeX & judgeY)[0]  # 提取符合条件的索引
            if valid_idx.size > 0:
                split_idx[(n, m)].extend(valid_idx.tolist())  # 存储索引信息

    # 如果启用了 areaBrokenMerge，则处理不足尺寸的分块
    if areaBrokenMerge:
        # 合并最右边的列
        if xExtra < colW:
            for n in range(len(judgeYresult)):
                if (n, int(xNums)) in split_idx and (n, int(xNums) - 1) in split_idx:
                    # 将最右边的列与前一列合并
                    split_idx[(n, int(xNums) - 1)].extend(split_idx.pop((n, int(xNums))))
        # 合并最下面的行
        if yExtra < rowH:
            for m in range(len(judgeXresult)):
                if (int(yNums), m) in split_idx and (int(yNums) - 1, m) in split_idx:
                    # 将最下面的行与上一行合并
                    split_idx[(int(yNums) - 1, m)].extend(split_idx.pop((int(yNums), m)))

    # 合并小于 ptBrokenMerge 的块
    if ptBrokenMerge is not None:
        for (rows, cols), pt_id in list(split_idx.items()):
            if len(pt_id) <= ptBrokenMerge:
                # 查找相邻块
                neighbor_blocks = [(rows + di, cols + dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]
                                   if (di != 0 or dj != 0) and (rows + di, cols + dj) in split_idx]
                if neighbor_blocks:
                    # 找到最近且点数最少的块
                    ni, nj = min(neighbor_blocks, key=lambda blk: len(split_idx[blk]))
                    split_idx[(ni, nj)].extend(split_idx.pop((rows, cols)))
    # 返回分割结果
    return split_idx

# 控制台输出分割点的数量分布
def xyz_2Dsplit_show(horizontalAxis: np.ndarray, verticalAxis: np.ndarray, rowH: float, colW: float, overlapH: float = 0, overlapW: float = 0, areaBrokenMerge: bool = True, ptBrokenMerge: int = None):
    print('')
    split_idx = defaultdict(list)  

    maxX, minX = horizontalAxis.max(), horizontalAxis.min()
    rangeX = maxX - minX
    stepX = colW * (1 - overlapW)  
    xNums, xExtra = divmod(rangeX, stepX)  

    maxY, minY = verticalAxis.max(), verticalAxis.min()
    rangeY = maxY - minY
    stepY = rowH * (1 - overlapH)
    yNums, yExtra = divmod(rangeY, stepY)
    print(f'总点数：{horizontalAxis.size}\n'
          f'高：{rangeY:0.2f} 宽：{rangeX:0.2f}\n'
          f'分割高：{rowH:0.2f} 分割宽：{colW:0.2f}\n'
          f'纵向重叠：{overlapH * 100}% 横向重叠：{overlapW * 100}%\n'
          f'分割为 {int(yNums) + 1}✖{int(xNums) + 1} 块(包含空白)\n')

    judgeXresult = [
        (minX + i * stepX <= horizontalAxis) & (horizontalAxis <= np.clip(minX + i * stepX + colW, minX, maxX))
        for i in range(int(xNums) + 1)]
    judgeYresult = [(minY + j * stepY <= verticalAxis) & (verticalAxis <= np.clip(minY + j * stepY + rowH, minY, maxY))
                    for j in range(int(yNums) + 1)]

    print('初步分块的点数分布')
    for n, judgeY in enumerate(judgeYresult):  
        for m, judgeX in enumerate(judgeXresult):  
            valid_idx = np.where(judgeX & judgeY)[0]  
            if valid_idx.size > 0:
                split_idx[(n, m)].extend(valid_idx.tolist())  
            print(f'{valid_idx.size:6d},', end='') if valid_idx.size > 0 else print(f'      ,', end='')
        print('')

    if areaBrokenMerge:
        print(f'\n开始(向左/上)合并不足{rowH}✖{colW}的块')
        if xExtra < colW:
            for n in range(len(judgeYresult)):
                if (n, int(xNums)) in split_idx and (n, int(xNums) - 1) in split_idx:
                    split_idx[(n, int(xNums) - 1)].extend(split_idx.pop((n, int(xNums))))
        if yExtra < rowH:
            for m in range(len(judgeXresult)):
                if (int(yNums), m) in split_idx and (int(yNums) - 1, m) in split_idx:
                    split_idx[(int(yNums) - 1, m)].extend(split_idx.pop((int(yNums), m)))

        print(f'合并后分块的点数分布:')
        for row in range(len(judgeYresult)):
            for col in range(len(judgeXresult)):
                if (row, col) in split_idx:
                    print(f'{len(split_idx[(row, col)]):6d},', end='')
                else:
                    print(f'      ,', end='')
            print('')  
    else:
        print(f'\n禁止(向左/上)合并不足{rowH}✖{colW}的块')

    if ptBrokenMerge is not None:
        print(f'\n开始合并点数小于{ptBrokenMerge}的分块:')
        for (rows, cols), pt_id in list(split_idx.items()):
            if len(pt_id) <= ptBrokenMerge:
                neighbor_blocks = [(rows + di, cols + dj) for di in [-1, 0, 1] for dj in [-1, 0, 1]
                                   if (di != 0 or dj != 0) and (rows + di, cols + dj) in split_idx]
                if neighbor_blocks:
                    ni, nj = min(neighbor_blocks, key=lambda blk: len(split_idx[blk]))
                    split_idx[(ni, nj)].extend(split_idx.pop((rows, cols)))
        print(f'合并后分块的点数分布:')
        for row in range(len(judgeYresult)):
            for col in range(len(judgeXresult)):
                if (row, col) in split_idx:
                    print(f'{len(split_idx[(row, col)]):6d},', end='')  
                else:
                    print(f'      ,', end='')  
            print('')  

    return split_idx
