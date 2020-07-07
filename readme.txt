1. 最高準確率是以每5張取1張test，其餘4張當作train所得到。

2. 若要執行test或train，請開啟任意編輯器，環境配置如下
-CUDA : 10.2
-Pytorch : 1.1.0
-Torchvision : 0.2.1
-Pillow : 6.2.1
-Numpy : 1.18.1

3. 直接執行test,py或train.py即可看到結果。

4. 我們有針對utils.py增加內容，相關參數調整可直接透過utils.py進行調整。

5. 最佳模型權重位於 resNet101_Pretrain_100e_5for1_lr0.001/model-1.00-best_train_acc.pth。

6. 最佳結果之參數設定、訓練曲線及測試結果皆在resNet101_Pretrain_100e_5for1_lr0.001資料夾中。

