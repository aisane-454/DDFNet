from tensorboardX import SummaryWriter

class Visualizer(object):
    '''
    visualizer
    '''
    def __init__(self):
        self.writer = SummaryWriter('./tensorboard_res')
    def initTrain(self):
        self.writer.add_scalars('loss', {'train': 0, 'val': 0}, 0)
        self.writer.add_scalars('acc', {'train': 0, 'val': 0}, 0)
        self.writer.add_scalars('auc', {'train': 0, 'val': 0}, 0)
        self.writer.add_scalars('f1', {'train': 0, 'val': 0}, 0)
        self.writer.add_scalars('kappa', {'train': 0, 'val': 0}, 0)
    def updateTrain(self, train_res, val_res, epoch):
        # print("??????????")
        self.writer.add_scalars('loss', {'train': train_res['loss'], 'val': val_res['loss']}, epoch)
        self.writer.add_scalars('acc', {'train': train_res['acc'], 'val': val_res['acc']}, epoch)
        self.writer.add_scalars('auc', {'train': train_res['auc'], 'val': val_res['auc']}, epoch)
        self.writer.add_scalars('f1', {'train': train_res['f1'], 'val': val_res['f1']}, epoch)
        self.writer.add_scalars('kappa', {'train': train_res['kappa'], 'val': val_res['kappa']}, epoch)