import torch
import torch.cuda

class MemoryTable(object):
    def __init__(self, reliable_num=4, lambda_p=0.5, lambda_c=1):
        self.reliable_num = reliable_num
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.reliables = []

    def __getitem__(self, index):
        return torch.stack(self.vectors[index]), self.reliables[index]

    def global_distance(self):
        return torch.cdist(self.vectors, self.vectors)

    def parts_dissimilarity(self):
        total = 0
        for part in self.parts:
            total += torch.cdist(part, part)
        return total / len(self.parts)

    def overall_dissimilarity(self, global_dist, parts_diss, cce):
        result = (1 - self.lambda_p) * global_dist + self.lambda_p * parts_diss + cce
        return result

    def cross_camera_encouragment(self, lambda_c=0.02):
        cce = torch.zeros((len(self.cameras), len(self.cameras)), device='cpu')
        for i, camera in enumerate(self.cameras):
            truth_vector = self.cameras == camera
            cce[i][truth_vector] = lambda_c 
        return cce

    def update(self, vectors, parts, cameras=None):
        self.vectors = vectors
        self.parts = parts
        if cameras is not None: 
            self.cameras = cameras
        
    def update_reliables(self):
        global_dist = self.global_distance()
        parts_diss = self.parts_dissimilarity()
        cce = self.cross_camera_encouragment()

        overall_diss = self.overall_dissimilarity(global_dist, parts_diss, cce)
        overall_diss = overall_diss.to('cpu')
        self.reliables = overall_diss.argsort(dim=1)[:,1:self.reliable_num+1]

    def get_reliables(self, indices):
        return [self.vectors[indices[i]] for i in indices]
        
    def probability(self, outputs, targets, t=0.1):
        cls_prob = torch.exp((self.vectors[targets] * outputs).sum(dim=1) / t)
        sum_prob = torch.exp(outputs.matmul(self.vectors.T) / t).sum(dim=1)
        return cls_prob / sum_prob