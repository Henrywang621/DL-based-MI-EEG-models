# Conda (4.6+) works
eval "$(conda shell.bash hook)"

conda activate tf-gpu
nohup python train_EEGNet_4_2.py --numepochs 300 --Patience4ES 30
nohup python train_EEGNet_8_2.py --numepochs 300 --Patience4ES 30
nohup python train_ETENet.py --numepochs 300 --Patience4ES 30  
nohup python train_EEGfusion.py --numepochs 300 --Patience4ES 30
nohup python train_Min2Net.py --numepochs 300 --Patience4ES 30
nohup python train_pCNN.py --numepochs 300 --Patience4ES 30
nohup python train_LSTM.py --numepochs 300 --Patience4ES 30
# nohup python train_PGCFMTL.py --numepochs 300 --Patience4ES 15
# nohup python train_SSNet.py --numepochs 300 --Patience4ES 10


conda activate torch37c
nohup python train_CLSTM.py --numepochs 300 --Patience4ES 30
conda activate torch37b6
nohup python train_ShallowConvNet.py --numepochs 300 --Patience4ES 30
nohup python train_DeepConvNet.py --numepochs 300 --Patience4ES 30
conda activate torch37c
nohup python -u train_SEFFNet.py --numepochs 300 --Patience4ES 30



