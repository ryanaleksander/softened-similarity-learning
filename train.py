import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import os
import argparse

from torch.utils.data import DataLoader
from dataset import Market1501Dataset
from configparser import ConfigParser
from collections import OrderedDict
from memory_table import MemoryTable
from model import SSLResnet
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', required=True)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    train_loader, memory_table = load_data(config['Train Data'])
    train(config['Train'], train_loader, memory_table)

def loss_fn(outputs, targets, memory_tb, t=0.1, ld=0.9):
    prob = -ld * memory_tb.probability(outputs, targets, t)
    reliables = torch.LongTensor(memory_tb.reliables[targets]).T.tolist()
    reliables_prob = torch.stack(list(map(lambda r: memory_tb.probability(outputs, r, t), reliables))).T
    reliables_prob = -((1 - ld) / len(reliables)) * torch.log(reliables_prob).sum(dim=1)
    return (prob + reliables_prob).sum()

def load_data(params):
    """Load data for training"""

    print('Loading dataset...')
    transform = transforms.Compose([
        transforms.Resize([768, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = Market1501Dataset(params['path'], data='train', transform=transform)
    train_loader = DataLoader(train_set, params.getint('batch_size'), shuffle=False)
    dataset_size = len(train_set)
    print('Done loading data')
    print('Loading memory table...')
    memory = MemoryTable(lambda_c=params.getfloat('lambda_c'), lambda_p=params.getfloat('lambda_p'), reliable_num=params.getint('reliable_num'))
    vectors = torch.load(params['memory_vectors_path'], map_location='cpu')
    parts = torch.load(params['memory_parts_path'], map_location='cpu')
    parts  = parts.chunk(8, 1)
    parts = list(map(lambda part: part.squeeze(1), parts))
    cameras = torch.load(params['memory_cameras_path']).tolist()
    memory.update(vectors, parts, cameras)
    memory.update_reliables()
    print('Done loading memory table')

    print('****Dataset info****')
    print(f'Numer of training samples: {len(train_loader) * params.getint("batch_size")}')
    return train_loader, memory

def train(params, train_loader, memory):
    device = 'cuda'
    net = SSLResnet(pretrained=True)
    net.to(device)
    # Load pretrained model
    if 'pretrained' in params and params['pretrained'] is not None:
        net.load_state_dict(torch.load(params['pretrained']))

    optimizer = torch.optim.SGD(net.parameters(), lr=params.getfloat('lr'))

    # Learning rate decay config
    lr_steps = [int(step) for step in params.get('lr_steps').split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    for i in range(params.getint('num_epochs')):
        net.train()
        print('*' * 5 + 'Epoch ' + str(i) + '*' * 5)
        batch_output = None
        batch_parts = None
        training_loss = 0
        training_progress = tqdm(enumerate(train_loader))
        for batch_index, (labels, images, _) in training_progress:
            optimizer.zero_grad()
            images = images.to(device)
            parts, embedding = net(images)
            embedding=embedding.to('cpu')
            loss = loss_fn(embedding, labels, memory, t=params.getfloat('t'), ld=params.getfloat('lambda_d'))
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch loss: ", training_loss / len(train_loader))
        with open('loss.txt', 'a') as f:
            f.write('Epoch ' + str(i) + ': ' + str(training_loss / len(train_loader)) + '\n')

        # Update vectors
        batch_output = []
        batch_parts = []
        net.eval()
        with torch.no_grad():
            for batch_index, (labels, images, cams) in enumerate(train_loader):
                images = images.to(device)
                parts, embedding = net(images)
                batch_output.append(embedding)
                batch_parts.append(parts)

            vectors = torch.cat(batch_output)
            parts = torch.cat(batch_parts)

        if (params.getint('num_epochs') + 1) % params.getint('ckpt') == 0:
            print('Saving checkpoint...' )
            torch.save(net.state_dict(), os.path.join('ckpt', 'checkpoint_' + str(i) + '.pth'))
            torch.save(parts, os.path.join(params['ckpt_path'], 'parts_' + str(i) + '.pth'))
            torch.save(vectors, os.path.join(params['ckpt_path'], 'vectors_' + str(i) + '.pth'))

        # Update memory table
        parts = parts.to('cpu')
        parts  = parts.chunk(8, 1)
        parts = list(map(lambda part: part.squeeze(1), parts))
        vectors = vectors.to('cpu')
        memory.update(vectors, parts)
        memory.update_reliables()

        scheduler.step()

    print('Training complete, saving final model...')
    torch.save(parts, os.path.join(params['ckpt_path'], 'parts_final' + str(i) + '.pth'))
    torch.save(vectors, os.path.join(params['ckpt_path'], 'vectors_final' + str(i) + '.pth'))
    torch.save(net.state_dict(), os.path.join(['ckpt_path'], 'model_final' + str(i) + '.pth'))

# device = 'cuda'

# memory = MemoryTable()
# vectors = torch.load('vectors.pth', map_location='cpu')
# parts = torch.load('parts.pth', map_location='cpu')
# parts  = parts.chunk(8, 1)
# parts = list(map(lambda part: part.squeeze(1), parts))
# cameras = torch.load('cameras.pth').tolist()
# memory.update(vectors, parts, cameras)
# memory.update_reliables()
# torch.cuda.empty_cache()
# net.to(device)
# for i in range(epoch):
#     net.train()
#     print('*' * 5 + 'Epoch ' + str(i) + '*' * 5)
#     batch_output = None
#     batch_parts = None
#     training_loss = 0
#     training_progress = tqdm(enumerate(train_loader))
#     for batch_index, (labels, images, _) in training_progress:
#         optimizer.zero_grad()
#         images = images.to(device)
#         parts, embedding = net(images)
#         embedding=embedding.to('cpu')
#         loss = loss_fn(embedding, labels, memory)
#         print(loss)
#         training_loss += loss.item()
#         loss.backward()
#         optimizer.step()

#     torch.save(net.state_dict(), os.path.join('ckpt', 'checkpoint_' + str(i) + '.pth'))
#     print("Epoch loss: ", training_loss / len(train_set))
#     with open('loss.txt', 'a') as f:
#         f.write('Epoch ' + str(i) + ': ' + str(training_loss / len(train_set)) + '\n')
#     batch_output = []
#     batch_parts = []
#     batch_cams = []
#     net.eval()
#     with torch.no_grad():
#         for batch_index, (labels, images, cams) in enumerate(train_loader):
#             batch_cams.extend(cams)
#             images = images.to(device)
#             parts, embedding = net(images)
#             batch_output.append(embedding)
#             batch_parts.append(parts)

#         vectors = torch.cat(batch_output)
#         parts = torch.cat(batch_parts)

#     torch.save(parts, os.path.join('ckpt', 'parts_' + str(i) + '.pth'))
#     torch.save(vectors, os.path.join('ckpt', 'vectors_' + str(i) + '.pth'))
#     # vectors, parts = load_tmp()
#     parts = parts.to('cpu')
#     parts  = parts.chunk(8, 1)
#     parts = list(map(lambda part: part.squeeze(1), parts))
#     vectors = vectors.to('cpu')
#     memory.update(vectors, parts)
#     memory.update_reliables()

#     lr_scheduler.step()

# torch.save(net.state_dict(),'model.pth')

if __name__ == '__main__':
    main()