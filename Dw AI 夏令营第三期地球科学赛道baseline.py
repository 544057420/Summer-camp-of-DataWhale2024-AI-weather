#链接：https://www.modelscope.cn/datasets/Datawhale/AICamp_earth_baseline

#数据集下载

from modelscope.msdatasets import MsDataset

ds =  MsDataset.load('Datawhale/AICamp_earth_baseline')

#您可按需配置 subset_name、split，参照“快速使用”示例代码

###数据集: Datawhale/AICamp_earth_baseline 请确保 lfs 已经被正确安装

git lfs install

git clone https://www.modelscope.cn/datasets/Datawhale/AICamp_earth_baseline.git
