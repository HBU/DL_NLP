import pandas as pd
version = pd.__version__
print(version)
# pandas 中的主要数据结构被实现为以下两类：
# DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
# Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。
