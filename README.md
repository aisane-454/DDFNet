1. Environment Setup

    1.1 Create Conda Environment

    conda create -n DDFNet python=3.10

    1.2 Activate Conda Environment

    conda activate DDFNet

2. Install Packages

    2.1 Torch and CUDA

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Lower versions of CUDA are also available.

    2.2 Other Dependencies

    pip install numpy scikit-learn tensorboardX tqdm requests

3. Train the Model

    # Note: The file 'data.npy' is missing in the current dataset folder. 'data.npy' stores the original data, which is not currently available due to privacy concerns. If you have any requirements, please contact me 'ltpang@stu.ecnu.edu.cn'.

    python train.py --dataset_name data_218_hypertension_longer --name hypertension_lr0.01_epoch60 --lr 0.01 --checkpoints_dir ./output/checkpoints --optimizer adam --epoch 60 --batch_size 16 --dataroot ./datasets/data_218_hypertension_longer --save_epoch_freq 20

5. Evaluate the Model
    # Note: The file 'data.npy' is missing in the current dataset folder. 'data.npy' stores the original data, which is not currently available due to privacy concerns. If you have any requirements, please contact me 'ltpang@stu.ecnu.edu.cn'.
   
    python eval.py --dataset_name data_218_hypertension_longer --name hypertension_lr0.01_epoch60 --lr 0.01 --checkpoints_dir ./output/checkpoints --optimizer adam --epoch 60 --batch_size 16 --dataroot ./datasets/data_218_hypertension_longer --save_epoch_freq 20 --load ./output/checkpoints/data_218_hypertension_longer/hypertension_lr0.001_epoch60/resnet18_focal_best.pth

