# =============================================================
# 定义树结点，用于数据集中连续型数值的二分裂。
# =============================================================
class BinaryTreeNode:

    def __init__(self):
        self.targetValue = None
        self.splitFeature = None
        self.splitValue = None
        self.leftNode = None
        self.rightNode = None
        self.isLeaf = False
        self.val = None