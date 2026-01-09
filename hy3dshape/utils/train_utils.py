import os
import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Anymate.dataset import AnymateDataset, my_collate
from Anymate.model import EncoderDecoder
from Anymate.utils.loss_utils import cross_entropy_with_probs_batch, cos_loss, cos_loss_clamp, chamfer_distance_with_average
from Anymate.utils.vol_utils import get_co, get_gt, extract_keypoints
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

import point_cloud_utils as pcu
from sklearn.cluster import DBSCAN
from diffusers import DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
from Anymate.utils.diffusion_utils import my_collate_diff, randn_tensor


def ddp_setup(rank: int, world_size: int, port: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = str(port)
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def accumulate(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='model_best.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    if is_best:
        torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        torch.save(state, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

def train_model(rank, world_size, config, args, shared_dict, port=12355):
    ddp_setup(rank, world_size, port)
    lowest_loss = 1e20
    model_config = config['model']
    model = EncoderDecoder(device=f'cuda:{rank}', dtype=torch.float32, **model_config)
    model.to(f'cuda:{rank}')

    if rank == 0:
        print('only_embed', model.only_embed)
        print('return_latents', model.return_latents)
        print(model)
    if not args.finetune:
        model.encoder.requires_grad_(False)
    model = DDP(model, device_ids=[rank])
    optimizer_config = config['optimizer']
    if args.finetune:
        optimizer = torch.optim.Adam(model.module.parameters(), **optimizer_config)
    else:
        if args.encoder == 'miche':
            optimizer = torch.optim.Adam(model.module.decoder.parameters(), **optimizer_config)
        elif args.encoder == 'bert':
            optimizer = torch.optim.Adam(list(model.module.decoder.parameters()) + list(model.module.point_proj.parameters()), **optimizer_config)
    # optionally resume from a checkpoint
    if args.resume:
        try:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.module.load_state_dict(checkpoint['state_dict'], strict=True)

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        except:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in optimizer.param_groups[0]['params']) / 1000000.0))
    my_collate_func = my_collate_diff if args.mode == 'diffusion' else my_collate
    if world_size > 1:
        if not args.split:
            train_dataset = shared_dict['train_dataset']
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            train_loader = DataLoader(train_dataset, batch_size=args.train_batch, sampler=train_sampler, collate_fn= my_collate_func)
        else:
            train_dataset = AnymateDataset(name=args.trainset + f'_{rank}', root=args.root) #should changed to dpp version
            train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, collate_fn= my_collate_func)
    else:
        train_dataset = AnymateDataset(name=args.trainset, root=args.root)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, collate_fn= my_collate_func)
        
    if rank == 0:
        test_loader = DataLoader(AnymateDataset(name=args.testset, root=args.root), batch_size=args.test_batch, shuffle=False, collate_fn= my_collate_func )

    if not args.schedule:
        args.schedule = [args.epochs//2]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, gamma=args.gamma)
    # step the scheduler to the start epoch
    for _ in range(args.start_epoch):
        scheduler.step()
    if rank == 0:    
        logger = SummaryWriter(log_dir=args.logdir)
        print('start ')
        print('test_frequency', args.test_freq)
        print('start from epoch', args.start_epoch)
    # start training
    for epoch in range(args.start_epoch, args.epochs):
        test_dict = None
        is_best = False
        lr = scheduler.get_last_lr()
        if rank == 0:
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr[0]))
        train_loss, grad_norm = train(train_loader, model, optimizer, args)
        if rank == 0 and (epoch == 0 or (epoch+1)%args.test_freq== 0):
            print('Testing epoch', epoch+1)
            test_dict = test(test_loader, model, args, world_size=world_size)
        
        
        scheduler.step()
        if rank == 0:
            print('Epoch{:d}. train_loss: {:.6f}.'.format(epoch + 1, train_loss))
            print('Epoch{:d}. grad_norm: {:.6f}.'.format(epoch + 1, grad_norm))
            info = {'train_loss': train_loss, 'grad_norm': grad_norm, 'lr': lr[0]}
            # print('Epoch{:d}. val_loss: {:.6f}.'.format(epoch + 1, val_loss))
            if test_dict is not None:
                for key, value in test_dict.items():
                    print('Epoch{:d}. {:s}: {:.6f}.'.format(epoch + 1, key, value))
                    
                test_loss = test_dict['test loss'] if not args.mode == 'diffusion' else test_dict['chamfer']
                is_best = test_loss < lowest_loss
                lowest_loss = min(test_loss, lowest_loss)
                for key, value in test_dict.items():
                    info[key] = value
                    
            for tag, value in info.items():
                logger.add_scalar(tag, value, epoch+1)
            save_dict = {'epoch': epoch + 1, 'state_dict': model.module.state_dict(), 'lowest_loss': lowest_loss, 'optimizer': optimizer.state_dict(), 'model_config': model_config}
            save_checkpoint(save_dict, is_best=is_best, checkpoint=args.checkpoint, snapshot=args.epochs//20)
            
def get_criterion(args):
    if args.loss == 'cos':
        criterion = cos_loss
    elif args.loss == 'ce':
        criterion = cross_entropy_with_probs_batch
    elif args.loss == 'cos_clamp':
        criterion = cos_loss_clamp
    else:
        criterion = chamfer_distance_with_average
    return criterion

def get_train_loss(model, data, args):
    criterion = get_criterion(args)
    loss = 0.0
    if args.mode == 'skin':
        y_pred, idx = model(data, downsample=1024)
        y_pred = torch.softmax(y_pred, dim=-1)
        y = data['skins'].to(args.device)
        y = y[:, idx]
        loss = criterion(y_pred, y)
        
    elif args.mode == 'conn':
        y_pred = model(data, args.device)
        y_pred = torch.softmax(y_pred, dim=-1)
        y = data['conns'].to(args.device)
        y = y[:, :y_pred.shape[1], :y_pred.shape[1]].float()
        loss = criterion(y_pred, y)
        
    elif args.mode == 'joints': # joints mode
        if args.decoder == 'transformer_latent':
            y_pred = model(data, args.device)
            joints_gt = data['joints'].to(args.device)
            loss = 0.0
            for i in range(joints_gt.shape[0]):
                joints_gt_i = joints_gt[i,:data['joints_num'][i], :3]
                loss += criterion(y_pred[i:i+1], joints_gt_i.unsqueeze(0))
            loss /= joints_gt.shape[0]
            
        elif args.decoder == 'triplane' or args.decoder == 'implicit_transformer':
            criterion = torch.nn.BCEWithLogitsLoss()
            y_pred = model(data, args.device, downsample=True)
            joints_gt = data['joints'].to(args.device)
            for i in range(joints_gt.shape[0]):
                joints_gt_i = joints_gt[i,:data['joints_num'][i], :3]
                vol = get_co(data['vox'][i])
                if data['vox'][i].shape[0] > 50000:
                    vol = vol[y_pred[i][1]]
                    gt = get_gt(vol.to(args.device), joints_gt_i)
                    loss += criterion(y_pred[i][0].squeeze(-1).unsqueeze(0), gt.unsqueeze(0))
                else:
                    gt = get_gt(vol.to(args.device), joints_gt_i)
                    loss += criterion(y_pred[i].squeeze(-1).unsqueeze(0), gt.unsqueeze(0))
            loss /= joints_gt.shape[0]
            
    elif args.mode == 'diffusion':
        noise_scheduler = DDIMScheduler(num_train_timesteps=args.num_train_step)
        
        samples = data['joints_repeat'].to(model.device).float()
        #use 256 input joints
        samples = samples[...,:args.num_training_points,:]

        samples = samples.to(model.device)
        noise = torch.randn(samples.shape, device=samples.device)
        assert samples.device == noise.device
        bs = samples.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=samples.device,
            dtype=torch.int64
        )

        noisy_joints = noise_scheduler.add_noise(samples, noise, timesteps)
        noisy_joints = noisy_joints.to(model.device)
        noisy_joints = noisy_joints.permute(0, 2, 1)

        noise_pred = model(data, noisy_joints=noisy_joints, timesteps = timesteps)
        noise_pred = noise_pred.permute(0, 2, 1)
        loss = F.mse_loss(noise_pred, noise)

    return loss 

def train(train_loader, model, optimizer, args):
    if not args.finetune:
        model.train()
        model.module.encoder.eval()
    else:
        model.train()
    loss_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    
    for data in tqdm(train_loader):
        loss = get_train_loss(model, data, args)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = 0
        
        for p in optimizer.param_groups[0]['params']:
            grad_norm += p.grad.data.norm(2).item()
        grad_norm_meter.update(grad_norm)
        optimizer.step()
        loss_meter.update(loss.item())
    
    return loss_meter.avg, grad_norm_meter.avg

def test(test_loader, model, args, world_size=1): 
    model.eval()
    assert args.mode in ['skin', 'joints', 'conn', 'diffusion'], 'mode should be choose from [skin, joints, conn, diffusion], got {}'.format(args.mode)
    
    if args.mode == 'skin' or args.mode == 'conn':
        loss_meter = AverageMeter()
        cos_sim_meter = AverageMeter()
        cos_clamp_meter = AverageMeter()
        for i, data in enumerate(tqdm(test_loader)):
            if world_size > 1 and i > 1000:
                break
            with torch.no_grad():
                y_pred = model(data, args.device)
                y_pred = torch.softmax(y_pred, dim=-1)
                
                if args.mode == 'skin':
                    y = data['skins'].to(args.device)
                elif args.mode == 'conn':
                    y = data['conns'].to(args.device)
                    y = y[:, :y_pred.shape[1], :y_pred.shape[1]].float()

                loss = 0.0
                loss = cross_entropy_with_probs_batch(y_pred, y)
                loss_meter.update(loss.item())
                cos_sim = cos_loss(y_pred, y)
                cos_sim_meter.update(cos_sim.mean().item())  # 1 - loss.item()
                cos_clamp = cos_loss_clamp(y_pred, y)
                cos_clamp_meter.update(cos_clamp.mean().item())
                
        loss_dict = {'test loss': loss_meter.avg, 'cos_sim': cos_sim_meter.avg, 'cos_clamp': cos_clamp_meter.avg}
    # get the loss of the joints prediction   
    elif args.mode == 'joints':
        if args.decoder == 'transformer_latent':
            loss_meter = AverageMeter()
            emd_meter = AverageMeter()
            for i, data in tqdm(enumerate(test_loader)):
                if world_size > 1 and i > 1000:
                    break
                with torch.no_grad():
                    y_pred = model(data, args.device)
                    joints_gt = data['joints'].to(args.device)

                    loss = 0.0
                    emd = 0.0
                    for i in range(joints_gt.shape[0]):
                        joints_gt_i = joints_gt[i,:data['joints_num'][i], :3]
                        y_pred_i = y_pred[i]

                        y_pred_i = y_pred[i].detach().cpu().numpy()
                        clustering = DBSCAN(eps=0.03, min_samples=1).fit(y_pred_i) # Consider add eps and min_samples as arguments
                        cluster_centers = []
                        for cluster in set(clustering.labels_):
                            cluster_centers.append(y_pred_i[clustering.labels_ == cluster].mean(axis=0))
                        y_pred_i = torch.from_numpy(np.array(cluster_centers)).to(args.device)

                        if y_pred_i.shape[0] < 2:
                            print(data['name'][i] + ' has less than 2 points')
                            continue
                        loss += chamfer_distance_with_average(y_pred_i.unsqueeze(0), joints_gt_i.unsqueeze(0))
                        emd_i, pi = pcu.earth_movers_distance(y_pred_i.cpu().numpy().astype(np.float64), joints_gt_i.cpu().numpy().astype(np.float64))
                        emd += emd_i
                    if loss == 0 or emd == 0:
                        continue
                    loss /= joints_gt.shape[0]
                    loss_meter.update(loss.item())
                    emd_meter.update(emd)
                loss_dict = {'test loss': loss_meter.avg, 'emd': emd_meter.avg}
                
        elif args.decoder == 'triplane' or 'implicit_transformer':
            loss_meter = AverageMeter()
            emd_meter = AverageMeter()
            chamfer_meter = AverageMeter()
            criterion = torch.nn.BCEWithLogitsLoss()
            for data in tqdm(test_loader):
                with torch.no_grad():
                    y_pred = model(data, args.device)
                    joints_gt = data['joints'].to(args.device)
                    loss = 0.0
                    emd = 0.0
                    chamfer = 0.0
                    for i in range(joints_gt.shape[0]):
                        joints_gt_i = joints_gt[i,:data['joints_num'][i], :3]
                        vol = get_co(data['vox'][i])
                        gt = get_gt(vol.to(args.device), joints_gt_i)
                        loss += criterion(y_pred[i].squeeze(-1).unsqueeze(0), gt.unsqueeze(0))
                        key_points = extract_keypoints(y_pred[i].cpu(), data['vox'][i])
                        if len(key_points) < 2:
                            continue
                        key_points = key_points / 32 - 1
                        chamfer += chamfer_distance_with_average(torch.from_numpy(key_points).unsqueeze(0).to(joints_gt_i.device), joints_gt_i.unsqueeze(0))
                        emd_i, _ = pcu.earth_movers_distance(key_points.astype(np.float64), joints_gt_i.cpu().numpy().astype(np.float64))
                        emd += emd_i
                    if loss == 0 or emd == 0 or chamfer == 0:
                        continue
                    loss /= joints_gt.shape[0]
                    loss_meter.update(loss.item())
                    emd_meter.update(emd)
                    chamfer_meter.update(chamfer.item())
            loss_dict = {'test loss': loss_meter.avg, 'emd': emd_meter.avg, 'chamfer': chamfer_meter.avg}
            
    elif args.mode == 'diffusion':
        loss_meter = AverageMeter()
        emd_meter = AverageMeter()
        chamfer_meter = AverageMeter()
        generator=torch.Generator(device='cpu').manual_seed(args.seed+1)
        scheduler = DDIMScheduler(num_train_timesteps=args.num_train_step)
        scheduler.set_timesteps(args.num_train_step)
        points_shape = [args.test_batch, args.num_training_points, 3]
        for data in tqdm(test_loader):
            joints_gt = data['joints'].to(dtype=torch.float64)
            points_noise = randn_tensor(points_shape, generator=generator)
            points = points_noise.permute(0, 2, 1).to(model.device)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    time_steps = torch.ones(args.test_batch, 1, dtype=torch.long) * t
                    time_steps = time_steps.to(model.device)
                    model_output = model(data, noisy_joints=points, timesteps = time_steps)

                    points = scheduler.step(model_output, t, points, generator=generator).prev_sample
            points = points.permute(0, 2, 1).cpu() 

            chamfer_sum = 0.0
            emd_sum = 0.0
            
            for i in range(args.test_batch):
                joints_gt_i = joints_gt[i,:data['joints_num'][i], :3]
                points_i = points[i]
                points_i = points_i.reshape( -1, 3)
                emd, p = pcu.earth_movers_distance(points_i.cpu().numpy(),joints_gt_i[:,:3].cpu().numpy())
                emd_sum += emd
                chamfer_sum += chamfer_distance_with_average(points_i.unsqueeze(0),joints_gt_i[:,:3].unsqueeze(0))  

            emd_meter.update(emd_sum)
            chamfer_meter.update(chamfer_sum.item())
        loss_dict = {'chamfer': chamfer_meter.avg, 'emd': emd_meter.avg}

    return loss_dict
