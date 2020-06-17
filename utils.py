import torch

def softened_similarity_loss(outputs, targets, memory_tb, part_num=8, t=0.1, ld=0.6):
    prob = -ld * memory_tb.probability(outputs, targets, t)
    reliables = torch.LongTensor(memory_tb.reliables[targets]).T.tolist()
    reliables_prob = torch.stack(list(map(lambda r: memory_tb.probability(outputs, r, t), reliables))).T
    reliables_prob = -((1 - ld) / part_num) * torch.log(reliables_prob).sum(dim=1)
    return (prob + reliables_prob).sum()

def load_vectors():
    embeddings = []
    parts = []
    for f in os.listdir('tmp'):
        if f.startswith('embedding'):
            embeddings.append(torch.load(os.path.join('tmp', f)))
        if f.startswith('parts'):
            parts.append(torch.load(os.path.join('tmp', f)))
    parts = torch.cat(parts)
    parts  = parts.chunk(8, 1)
    parts = list(map(lambda part: part.to('cpu').squeeze(1), parts))
    return torch.cat(embeddings).to('cpu'), parts