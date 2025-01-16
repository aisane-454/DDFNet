import os
import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HypertenseDataset
from utils.tools import setup_seed, worker_init_fn
from utils.visualizer import Visualizer
from utils.metric import Metrics
from utils.options import TrainOptions
from utils.logger import Logger
from utils.loss import BinaryFocalLoss
from models.fdlnet import mymodel
'''
# Example Command
python train.py --dataset_name data_218_hypertension_longer --name hypertension_lr0.001_epoch60 --lr 0.001 --checkpoints_dir ./output/checkpoints --optimizer adam --epoch 60 --batch_size 16 --dataroot ./datasets/data_218_hypertension_longer --save_epoch_freq 20

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


train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

eval_transforms = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = HypertenseDataset(opts.dataroot, split="train", transform=train_transforms)
val_dataset = HypertenseDataset(opts.dataroot, split="val", transform=eval_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, worker_init_fn=worker_init_fn ,prefetch_factor=4,persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)


def train():
    setup_seed(3407)
    visualizer = Visualizer()
    visualizer.initTrain()
    train_metric = Metrics(type="train", logger = logger)
    val_metric = Metrics(type="val", logger = logger)
    start_epoch = 1
    best_acc = 0

    model = mymodel(backbone='resnet18', pretrained_base=False).cuda()

    if opts.optimizer.lower() == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif opts.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    elif opts.optimizer.lower() == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.90, weight_decay=0.0001)
    elif opts.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.0001)
    elif opts.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.0001)
    
    criterion = BinaryFocalLoss() # logic
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

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
        logger.logger.info("Model saved as %s" % path)
        
    # Count Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Sum of parameters: {total_params}')

    for epoch in range(start_epoch, NUM_EPOCH+1):
        train_metric.reset()

        model.train()
        start_time = time.perf_counter()
        with torch.enable_grad():
            for image, label in train_loader:
                image = image.float().cuda()
                label = label.cuda()
                logits = model(image)
                logits = logits.squeeze()
                
                loss = criterion(logits, label.float())
                train_metric.update(label, logits, loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for image, label in val_loader:

                image = image.float().cuda()
                label = label.cuda()

                # Forward pass
                logits = model(image)
                logits = logits.squeeze()

                loss = criterion(logits, label.float())
                val_metric.update(label, logits, loss)

            train_res = train_metric.get_results()
            train_metric.print_res(train_res, epoch)
            val_res = val_metric.get_results()
            val_metric.print_res(val_res, epoch)
            visualizer.updateTrain(train_res, val_res, epoch)
            
            end_time = time.perf_counter()
            print(f'###########epoch {epoch}: {end_time - start_time}#########')
            logger.logger.info(f'###########epoch {epoch}: {end_time - start_time}#########')
            
            # save best_model
            if val_res['acc'] > best_acc:
                best_acc = val_res['acc']
                save_ckpt(os.path.join(opts.checkpoints_dir, opts.dataset_name, opts.name, "%s_%s_best.pth"%(opts.model, opts.loss)))
        
        # save model
        if  epoch % opts.save_epoch_freq == 0:
            save_ckpt(os.path.join(opts.checkpoints_dir, opts.dataset_name, opts.name, "%s_%s_epoch%d.pth"%(opts.model, opts.loss, epoch)))

if __name__ == "__main__":
    train()