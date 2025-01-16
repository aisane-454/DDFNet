import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import HypertenseDataset
from utils.tools import setup_seed
from utils.visualizer import Visualizer
from utils.metric import Metrics
from utils.options import TrainOptions
from utils.logger import Logger
from utils.loss import BinaryFocalLoss
from models.fdlnet import mymodel
'''
# Example Command
python eval.py --dataset_name data_218_hypertension_longer --name hypertension_lr0.001_epoch60 --lr 0.001 --checkpoints_dir ./output/checkpoints --optimizer adam --epoch 60 --batch_size 16 --dataroot ./datasets/data_218_hypertension_longer --save_epoch_freq 20 --load ./output/checkpoints/data_218_hypertension_longer/hypertension_lr0.001_epoch60/resnet18_focal_best.pth

'''

logger = Logger()

args = TrainOptions(logger)
opts = args.parse()

os.environ["CUDA_VISIBLE_DEVICES"]='0'

DEVICE = "cuda"
LEARNING_RATE = opts.lr
BATCH_SIZE = opts.batch_size
NUM_EPOCH = opts.epoch
IMAGE_WIDTH, IMAGE_HEIGHT = [opts.image_size, opts.image_size]
PATCH_SIZE = opts.patch_size
NUM_WORKERS = 8
PIN_MEMORY = True
SEED = 3407

eval_transforms = transforms.Compose([
    transforms.ToTensor(),
])

val_dataset = HypertenseDataset(opts.dataroot, split="val", transform=eval_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)


def eval():
    setup_seed(3407)
    visualizer = Visualizer()
    visualizer.initTrain()
    val_metric = Metrics(type="val", logger = logger)
    start_epoch = 1
    best_acc = 0

    model = mymodel(backbone='resnet18', pretrained_base=False).cuda()
    criterion = BinaryFocalLoss() 

    if opts.load is not None and os.path.isfile(opts.load):
        checkpoint = torch.load(opts.load)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint["cur_epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print("epoch = ", checkpoint["cur_epoch"])
        print("best_acc = ", checkpoint["best_acc"])

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_epoch": epoch,
            "model_state": model.state_dict(),
            "best_acc": best_acc,
        }, path)
        print("Model saved as %s" % path)

    for epoch in range(1):
        start_time = time.perf_counter()
        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for image, label in val_loader:

                image = image.float().cuda()
                label = label.cuda()

                # Forward pass
                logits = model(image)
                logits = logits.reshape([label.shape[0]])

                loss = criterion(logits, label.float())
                val_metric.update(label, logits, loss)

            val_res = val_metric.get_results()
            val_metric.print_res(val_res, epoch)
            
            end_time = time.perf_counter()
            print(f'###########epoch {epoch}: {end_time - start_time}#########')
            logger.logger.info(f'###########epoch {epoch}: {end_time - start_time}#########')

if __name__ == "__main__":
    eval()