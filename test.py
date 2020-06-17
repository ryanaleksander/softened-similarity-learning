import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import os
import argparse

from torch.utils.data import DataLoader
from dataset import Market1501Dataset
from collections import OrderedDict
from memory_table import MemoryTable
from model import SSLResnet
from tqdm import tqdm
from configparser import ConfigParser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', required=True)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    transform = transforms.Compose([
        transforms.Resize([768, 256]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    net = SSLResnet(torchvision.models.resnet50(pretrained=False))
    net.load_state_dict(torch.load(config['Test']['model_path']))

    test_loader, query_loader = load_data(config['Test Data'], transform, load_img=True)
    test_filepath = os.path.join(config['Test']['output_path'], 'test.pth')
    query_filepath = os.path.join(config['Test']['output_path'], 'query.pth')
    print('Creating embeddings...')
    create_embeddings(net, test_filepath, test_loader)
    create_embeddings(net, query_filepath, query_loader)

    print('Calculating Top-1 accuracy...')
    test_loader, query_loader = load_data(config['Test Data'], transform, load_img=False)
    evaluate(test_loader, query_loader, test_filepath, query_filepath)

def load_data(params, transform=None, load_img=True):
    test_set = Market1501Dataset(params['path'], data='test', load_img=load_img, transform=transform)
    query_set = Market1501Dataset(params['path'], data='query', load_img=load_img, transform=transform)
    if load_img:
        test_loader = DataLoader(test_set, 8, shuffle=False)
        query_loader = DataLoader(query_set, 8, shuffle=False)
    else:
        test_loader = DataLoader(test_set, len(test_set), shuffle=False)
        query_loader = DataLoader(query_set, len(query_set), shuffle=False)
    return test_loader, query_loader

def create_embeddings(net, output, dataloader):
    device = 'cuda'
    net.eval()
    net.to(device)
    with torch.no_grad():
        batch_output = []
        for batch_index, (labels, images, cams) in enumerate(dataloader):
            images = images.to(device)
            _, embedding = net(images)
            embedding = embedding.to('cpu')
            batch_output.append(embedding)

        vectors = torch.cat(batch_output)
        torch.save(vectors, output)

def evaluate(test_loader, query_loader, test_filepath, query_filepath):
    query_vectors = torch.load(test_filepath)
    test_vectors = torch.load(query_filepath)
    results = torch.cdist(query_vectors, test_vectors)
    ids = results.argmin(dim=1).to('cpu')
    all_labels = next(iter(query_loader))[0]
    test_labels = next(iter(test_loader))[0]
    test_labels = torch.LongTensor(test_labels)
    test_labels = test_labels[ids]
    accuracy = (all_labels == test_labels).sum().item() / len(all_labels)
    print("Top-1 Accuracy: %.2f" % accuracy)
    
if __name__ == '__main__':
    main()