from torch.utils.data import DataLoader

import torch


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond


# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from humanml.data.dataset import KIT
        return KIT
    elif name == "gta" or 'GTA':
        from humanml.data.dataset import GTA
        return GTA
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit", "GTA"]:
        return t2m_collate
    else:
        return collate


def get_dataset(name, num_frames, split='train', hml_mode='train'):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit", "GTA"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train'):
    dataset = get_dataset(name, num_frames, split, hml_mode)
    collate = get_collate_fn(name, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, #[REP]
        num_workers=8, drop_last=True, collate_fn=collate
    )
    loader = repeater(loader)

    return loader


def get_mdm_args(**kwargs):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = 'text'
    num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 22
    nfeats = 6

    if kwargs['dataset'] == 'laion' or kwargs['dataset'] == 'amass':
        data_rep = 'hml_vec'
        njoints = 135
        nfeats = 1

    if kwargs['dataset'] == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif kwargs['dataset'] == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {
        'modeltype': '',
        'njoints': njoints,
        'nfeats': nfeats,
        'num_actions': num_actions,
        'translation': True,
        'pose_rep': 'rot6d',
        'glob': True,
        'glob_rot': True,
        'latent_dim': kwargs['latent_dim'],
        'ff_size': 1024,
        'num_layers': kwargs['layers'],
        'num_heads': 4,
        'dropout': 0.1,
        'activation': "gelu",
        'data_rep': data_rep,
        'cond_mode': cond_mode,
        'cond_mask_prob': kwargs['cond_mask_prob'],
        'action_emb': action_emb,
        'arch': kwargs['arch'],
        'emb_trans_dec': kwargs['emb_trans_dec'],
        'clip_version': clip_version,
        'dataset': kwargs['dataset']
    }
