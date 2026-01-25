"""
快速使用指南
============

本脚本提供消融实验的快速启动方法。

基本用法：
---------

1. 查看所有实验配置：
   python ablation_config.py

2. 一键运行完整实验（训练+测试）：
   python run_ablation_study.py

3. 仅训练所有模型：
   python run_ablation_study.py --train

4. 仅测试已训练的模型：
   python run_ablation_study.py --test

5. 查看实验配置概览：
   python run_ablation_study.py --summary


高级用法：
---------

单独运行训练：
   python train_ablation.py

单独运行测试：
   python test_ablation.py

测试模型包装器：
   python model_wrapper.py

测试损失函数包装器：
   python loss_wrapper.py


预期输出：
---------

训练完成后会生成：
- checkpoints/model_*.pth：11个训练好的模型
- training_history.json：训练历史记录

测试完成后会生成：
- results/{experiment_name}/*.mat：测试结果文件
- test_results_summary.json：测试结果汇总


注意事项：
---------

1. 确保已正确配置 DAT-Net-Unsupervised-v2/config.py
2. 确保数据集路径正确
3. 预留足够的磁盘空间（模型+结果文件）
4. 如果使用GPU，确保显存充足
5. 完整训练可能需要数小时到数天


故障排除：
---------

如遇到导入错误，检查：
- DAT-Net 目录是否存在
- DAT-Net-Unsupervised-v2 目录是否存在
- Python路径是否正确

如遇到内存不足，调整：
- DAT-Net-Unsupervised-v2/config.py 中的 BATCH_SIZE

如需中断后继续，修改：
- ablation_config.py 中的 ABLATION_ORDER


联系方式：
---------

如有问题，请查看 README.md 获取详细文档。
"""

if __name__ == '__main__':
    print(__doc__)
