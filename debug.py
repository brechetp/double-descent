import utils
import torch
import models.cnn
#imresize = (170, 150)
imresize=None
dataset = utils.get_dataset('rsna', imresize=imresize)

train_loader, train_size, val_loader, val_size, test_loader, test_size = utils.get_dataloader(*dataset[:2], 100)

model = models.cnn.CNN(1)

for idx, (x, man, age) in enumerate(train_loader):

    print(idx, x.size(), man, age)

    y = model(x, man)

