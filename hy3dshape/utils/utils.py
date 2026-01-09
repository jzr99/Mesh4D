import torch
from Anymate.model import EncoderDecoder
from sklearn.cluster import DBSCAN

def load_checkpoint(path, device, num_joints):
    print(f"Loading model from {path}")
    model_state = torch.load(path)
    model_weights = model_state['state_dict']
    
    try:
        model_config = model_state['model_config']
        model = EncoderDecoder(device=device, dtype=torch.float32, **model_config, load_bert=False)
        model.to(device)
        model.load_state_dict(model_weights, strict=True)
    except:
        encoder = path.split('/')[-1].split('.')[0].split('-')[0]
        decoder = path.split('/')[-1].split('.')[0].split('-')[1]
        model = EncoderDecoder(encoder=encoder, decoder=decoder, device=device, dtype=torch.float32, num_joints=num_joints, load_bert=False)
        model.to(device)
        model.load_state_dict(model_weights, strict=True)
        
    print(f"Loaded model from {path}")

    return model

def get_joint(pc, model, device='cuda', save=None, vox=None, eps=0.03, min_samples=1):
    model.eval()
    data = {'points_cloud': pc.unsqueeze(0)}
    if vox is not None:
        data['vox'] = vox.unsqueeze(0)
    with torch.no_grad():
        model.decoder.inference_mode(eps=eps, min_samples=min_samples)
        joints = model(data, device=device)
        joints = torch.tensor(joints, dtype=torch.float32).to(device)

        if save is not None:
            torch.save(joints, save)

        return joints
    
def get_connectivity(pc, joints, model, device='cuda',return_prob=False, save=None):
    model.eval()
    data = {'points_cloud': pc.unsqueeze(0), 'joints': joints.unsqueeze(0), 'joints_num': torch.tensor([joints.shape[0]]), 
            'joints_mask': torch.ones(joints.shape[0], device=device).unsqueeze(0)}
    with torch.no_grad():
        conns = model(data, device=device).softmax(dim=-1)
        conns = conns.squeeze(0) if return_prob else torch.argmax(conns, dim=-1).squeeze(0)

        if save is not None:
            torch.save(conns, save)

        return conns

def get_skinning(pc, joints, conns, model, vertices=None, bones=None, device='cuda', save=None):
    model.eval()
    
    if bones is None:
        bones = []
        for i in range(joints.shape[0]):
            if conns[i] != i:
                bones.append(torch.cat((joints[conns[i]], joints[i]), dim=-1))
        bones = torch.stack(bones, dim=0)

    data = {'points_cloud': pc.unsqueeze(0), 'bones': bones.unsqueeze(0), 'bones_num': torch.tensor([bones.shape[0]]),
            'bones_mask': torch.ones(bones.shape[0], device=device).unsqueeze(0)}
    
    if vertices is not None:
        data['vertices'] = vertices.unsqueeze(0)
        model.decoder.inference = True

    with torch.no_grad():
        skins = model(data, device=device).softmax(dim=-1).squeeze(0)
        
        if save is not None:
            torch.save(skins, save)

        return skins
