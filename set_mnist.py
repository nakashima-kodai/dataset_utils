import os
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(iter, img_tensor, label):
    grid_img = img_tensor[0].numpy().transpose((1, 2, 0))

    image_name = str(iter) + '.png'
    label = label.item()
    save_dir = save_dir = os.path.join('../svhn2mnist_m', 'test00')
    mkdir(save_dir)
    save_path = os.path.join(save_dir, image_name)
    print('save_path: {}'.format(save_path))
    plt.imsave(save_path, grid_img)

    with open('../svhn2mnist_m/test_label.txt', 'a') as f:
        temp = 'test00/' + image_name + ' ' + str(label)
        f.write(temp)
        f.write('\n')


transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.SVHN('../svhn', split='test', download=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

for iter, (image, label) in enumerate(data_loader):
    save_image(iter, image, label)
