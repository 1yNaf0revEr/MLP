# Fashion-MNIST MLP Classifier 
从零实现三层 MLP 分类器，完成 Fashion-MNIST 服装 10 分类任务。

## 环境配置
Python 3.12依赖：numpy>=1.26、matplotlib>=3.8

# Conda 环境
conda create -n cv python=3.12

conda activate cv

pip install -r requirements.txt

## 数据集
Fashion-MNIST：10 类服装灰度图像，尺寸 28×28
预处理：展平为 784 维向量，归一化 [0,1]

类别：T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴

数据划分：训练集 / 验证集 / 测试集
首次运行自动下载至 data/ 目录，下载失败可手动放入原始压缩包。

## 项目结构
.
├── data.py        # 数据加载、预处理、批生成

├── layers.py      # 线性层/激活层（前向+反向）

├── loss.py        # Softmax 交叉熵损失

├── model.py       # 三层 MLP 模型

├── optimizer.py   # SGD + L2 正则

├── utils.py       # 评估、可视化、模型保存加载

├── train.py       # 训练流程

├── search.py      # 超参数搜索

├── test.py        # 测试集评估

├── main.py        # 主入口

├── data/
├── checkpoints/
└── search_results/

## 模型结构
标准三层 MLP：
Input → Linear → Activation → Linear → Activation → Linear

默认结构：784 → 256 → 128 → 10

## 运行方式
python main.py

## 输出结果
运行后自动生成：
- training_curves.png：训练/验证曲线
- confusion_matrix.png：测试集混淆矩阵
- first_layer_weights.png：首层权重可视化
- misclassified.png：分类错误样本
- checkpoints/final_best_model.npz：最优模型
- search_results/search_results.json：超参搜索记录

## 仓库与模型
GitHub 仓库：

模型权重下载：