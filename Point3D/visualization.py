import numpy as np
import open3d as o3d
from sklearn.metrics import confusion_matrix

label_colors = {
    0: [0.5, 0.5, 0.5],  #
    1: [1.0, 0.0, 0.0],  #
    2: [0.0, 1.0, 0.0],  #
    3: [1.0, 1.0, 0.0],  #
    4: [0.0, 0.0, 1.0],  #
}


def xyz_visual(pt_xyz: np, pt_colors: np = None, pt_labels: np = None, label2colors: dict = None):
    """
    可视化三维点云数据，支持基于点颜色或类别标签的渲染方式。

    Args:
        pt_xyz (numpy.ndarray): 形状为 (N, 3) 的数组，表示 N 个点的三维坐标。
        pt_colors (numpy.ndarray, optional): 形状为 (N, 3) 的数组，表示每个点的 RGB 颜色值，值域为 [0,1]。默认值为 None。
        pt_labels (numpy.ndarray, optional): 形状为 (N,) 的数组，表示每个点的类别标签。默认值为 None。
        label2colors (dict, optional): 字典，映射每个类别标签到相应的 RGB 颜色数组。如果 `pt_labels` 存在且未提供 `pt_colors`，则根据该映射上色。默认值为 None。

    Raises:
        ValueError: 如果既未提供 `pt_colors`，也未提供 `pt_labels` 和 `label2colors`，将抛出错误。

    Description:
        本函数根据输入的坐标、颜色或标签信息选择合适的点云渲染方式：

        1. 如果提供 `pt_colors`，则直接使用该颜色数组对点云上色。
        2. 如果 `pt_colors` 未提供，但存在 `pt_labels` 和 `label2colors`，则根据 `label2colors` 的映射为每个类别着色。
        3. 若 `pt_colors` 和 `label2colors` 均未提供，则为每个唯一的标签生成随机颜色进行渲染。
        4. 如果以上条件均未满足，将抛出 ValueError，提示用户必须提供有效的颜色或标签信息。

    Notes:
        该方法用于快速查看三维点云，并可通过标签进行分割、上色，从而帮助识别各类的点分布。

    """
    # 如果提供了颜色，则直接使用
    if pt_colors is not None:
        # 使用提供的 pt_colors 作为颜色
        colors = pt_colors
    elif pt_labels is not None:
        # 如果没有提供 pt_colors，且提供了标签
        if label2colors is not None:
            # 根据标签和 label2colors 映射生成颜色
            colors = np.array([label2colors[label] for label in pt_labels])
        else:
            # 如果没有 label2colors 映射，则为每个标签生成随机颜色
            unique_labels = np.unique(pt_labels)
            label2colors = {label: np.random.rand(3) for label in unique_labels}
            colors = np.array([label2colors[label] for label in pt_labels])
    else:
        # 如果既没有提供 pt_colors，也没有提供 pt_labels，则抛出异常
        raise ValueError("必须提供 颜色 或者 类别、映射颜色")
    # 创建点云对象并设置颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_xyz)  # 设置点云的坐标
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置点云的颜色

    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name="PointCloud", width=800, height=600)


def xyz_visual_difference(pt_xyz, label_gt, label_pred, Data_Graphics: bool = True, idx2Graphics: int = None):
    """
    可视化三维点云数据的预测与真实标签差异，自动选择输出或图形化方式。

    Args:
        - pt_xyz (ndarray): 点云坐标数组，形状为 (N, 3)，其中 N 表示点的数量。
        - label_gt (ndarray): 真实标签数组，与 pt_xyz 一一对应。
        - label_pred (ndarray): 预测标签数组，与 pt_xyz 一一对应。
        - Data_Graphics (bool): 可选参数，默认为 True，指示是否打印每个类别的混淆矩阵信息。
        - idx2Graphics (int): 可选参数，指定要可视化的类别的标签编号。如果为 None，则随机选取一个类别进行可视化。

    Raises:
        ValueError: 当 Data_Graphics 为 False 且没有指定 idx2Graphics 时，如果未找到类别标签，会抛出错误。

    Description:
        该函数根据输入参数执行以下功能：
        - 如果 Data_Graphics 为 True：计算并输出每个类别的混淆矩阵，显示预测标签与真实标签的匹配情况。
        - 如果 Data_Graphics 为 False：查找 idx2Graphics 指定的类别点云并调用 xyz_visual 函数进行可视化。如果未指定 idx2Graphics，则随机选择一个类别进行可视化。

    Note:
        该函数旨在帮助用户分析不同类别的预测精度以及可视化差异点云。
    """
    unique_classes = np.union1d(label_gt, label_pred)
    if Data_Graphics is True:
        for i in range(len(unique_classes)):
            print(f'GT-{unique_classes[i]} ', end='')
            c_m = confusion_matrix(label_gt, label_pred)  # 混淆矩阵
            print(f'{c_m[i]}')
    else:
        if idx2Graphics is None:
            idx = np.random.randint(0, len(unique_classes))
            idx2Graphics = unique_classes[idx]  # 生成随机类别
        for i in range(len(unique_classes)):
            label_gt_inds = (label_gt == unique_classes[i])  # 获取gt的某个类别的所有索引
            label_gt_pred = label_pred[label_gt_inds]  # gt的类别索引下的 pred类别
            if idx2Graphics == unique_classes[i]:
                print(f'当前生成的类别为: {idx2Graphics}')
                xyz_visual(pt_xyz=pt_xyz[label_gt_inds], pt_labels=label_gt_pred)
                break















