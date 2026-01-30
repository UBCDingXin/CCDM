# Improved Continuous Conditional Diffusion Model (iCCDM)

版本：iCCDM v0.5.2

Log:
1. 修复了bug,即trainer.py中,loss的计算应该基于targe_labels而不是图片真实的labels
2. 修复了UNet中，给定mask后不起作用的bug。
3. 训练和推理阶段，用不同的lambda_y来控制y2cov

- [x] Support Matrix Form EDM
- [x] Support y-dependent noise
- [x] Support adaptive vicinity 
- [x] Support auxiliary regression penalty (but not effective)
- [x] Support DiT
- [x] Support DMD2M
- [ ] Support Multi-GPU
