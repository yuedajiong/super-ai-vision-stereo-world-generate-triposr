import torch
import numpy as np
import os

import math
class SuperVisionDataset(torch.utils.data.Dataset):
    class Coordinate:
        def view_to_world(distance, azimuth, elevation, is_degree):  #pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)  #horizontal plane y=0
            if is_degree:
                azimuth_radian = azimuth /180.0*math.pi
                elevation_radian = elevation /180.0*math.pi
            else:
                azimuth_radian = azimuth
                elevation_radian = elevation

            x = distance * torch.cos(elevation_radian) * torch.sin(azimuth_radian)
            y = distance * torch.sin(elevation_radian)
            z = distance * torch.cos(elevation_radian) * torch.cos(azimuth_radian)
            camera_position = torch.stack([x, y, z], dim=1)  #world

            at = torch.tensor(((0, 0, 0),), dtype=torch.float32)
            up = torch.tensor(((0, 1, 0),), dtype=torch.float32)
            z_axis = torch.nn.functional.normalize(at - camera_position, eps=1e-5)  #first
            x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
            y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)

            R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
            t = -torch.bmm(R, camera_position[:, :, None])[:, :, 0]
            R = R.transpose(1, 2)
            return R, t

        def world_to_view(R, t, to_degree):
            camera_position = -torch.bmm(R, t[:, :, None])[:, :, 0]

            distance = torch.norm(camera_position)

            azimuth_radian = torch.atan2(camera_position[:, 0], camera_position[:, 2])
            elevation_radian = torch.atan2(camera_position[:, 1], torch.sqrt(camera_position[:, 0]**2 + camera_position[:, 2]**2))

            if to_degree:
                azimuth = azimuth_radian /math.pi*180.0
                elevation = elevation_radian /math.pi*180.0

            return distance, azimuth, elevation

        def Rt_to_matrix(R, t):
            matrix = torch.eye(4)
            matrix[:3, :3] = R
            matrix[:3, 3] = t
            return matrix

    def __init__(self, is_train, data_path, image_size):
        def pick_pose_from_file(data_file):
            item = data_file[len('image__'):-len('.png')].split('__')  #image__distance_2.30__elevation_330__azimuth_330.png
            distance = float(item[0][len('distance_'):])
            elevation = int(item[1][len('elevation_'):])
            azimuth = int(item[2][len('azimuth_'):])
            matrix = self.Coordinate.Rt_to_matrix(*self.Coordinate.view_to_world(torch.tensor([distance]), torch.tensor([azimuth]), torch.tensor([elevation]), is_degree=True))
            return torch.Tensor(matrix)

        import torchvision
        if is_train:
            transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([image_size,image_size]), torchvision.transforms.ToTensor()])  #augment
        else:
            transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([image_size,image_size]), torchvision.transforms.ToTensor()])

        import os
        import PIL
        self.data_all = []
        for data_file in sorted(os.listdir(data_path)):
            with open(os.path.join(data_path, data_file), "rb") as handler:
                rgba = PIL.Image.open(handler).convert("RGBA")
                rgba = transforms(rgba)
                image = rgba[0:3,:,:].permute(1,2,0)
                mask = rgba[3:4,:,:].permute(1,2,0)
                pose = pick_pose_from_file(data_file)
                self.data_all.append((image,mask,pose))
    
    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]
        return item

def infer(image_size, image_path, output_file, remove_bg, foreground_ratio, render_video, device):
    print('superv ...')
    if remove_bg:
        import rembg  #pip install rembg
        import PIL
        def remove_background(image, rembg_session, **rembg_kwargs):
            if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
                image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
            return image

        def resize_foreground(image, ratio):
            alpha = np.where(image[..., 3] > 0)
            y1, y2, x1, x2 = (alpha[0].min(), alpha[0].max(), alpha[1].min(), alpha[1].max())
            fg = image[y1:y2, x1:x2]  #crop the foreground
            
            size = max(fg.shape[0], fg.shape[1])
            ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
            ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
            new_image = np.pad(fg, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))  #pad to square
            
            new_size = int(new_image.shape[0] / ratio)  #compute padding according to the ratio
            ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
            ph1, pw1 = new_size - size - ph0, new_size - size - pw0
            new_image = np.pad(new_image, ((ph0, ph1), (pw0, pw1), (0, 0)), mode="constant", constant_values=((0, 0), (0, 0), (0, 0)))  #pad to size, double side
            return PIL.Image.fromarray(new_image)

        rembg_session = rembg.new_session()
        image = remove_background(PIL.Image.open(image_path), rembg_session)
        image = resize_foreground(np.array(image), foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = PIL.Image.fromarray((image * 255.0).astype(np.uint8))
        image = np.array(image).astype(np.float32) / 255.0
    else:
        image = np.array(PIL.Image.open(image_path).convert("RGB"))
 
    image = torch.stack([torch.from_numpy(image)], dim=0)  #batched  
    print('image', image.shape)
    image = torch.nn.functional.interpolate(image.permute(0, 3, 1, 2), (image_size, image_size), mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1)
    print('image', image.shape)
    image = image[:, None].to(device)
    print('image', image.shape)

    print('superv >>> image ok, network to') 
    from network import TSR
    model = TSR(img_size=image_size, depth=16, embed_dim=768, num_channels=1024, num_layers=16, cross_attention_dim=768, radius=3, valid_thresh=0.001, num_samples_per_ray=128, n_hidden_layers=9, official=True)
    model.load_state_dict(torch.load('./ckpt/TripoSR/model.ckpt', map_location='cpu'))
    model.to(device)

    print('superv >>> network ok, infer to')
    with torch.no_grad():
        print('image', image.shape)
        scene_codes = model(image)

    print('superv >>> infer ok, mesh to')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    mesh = model.extract_mesh(scene_codes)[0]
    mesh.export(output_file)  #.ply

    print('superv >>> mesh ok, video to')
    if render_video:
        video_file = output_file[:-len(output_file.split('.')[-1])]+'mp4'
        os.makedirs(os.path.dirname(video_file), exist_ok=True)
        render_images = model.render_images(scene_codes, n_views=16, return_type="pil")[0]
        import imageio  #pip install imageio[ffmpeg]
        with imageio.get_writer(video_file, fps=30) as writer:
            for frame in render_images: 
                writer.append_data(np.array(frame))
            writer.close()
    print('superv !!!')

def train(image_size, batch_size, epochs, checkpoint_path, best_checkpoint_file=None, device=None):    
    def get_ray_bundle(height, width, focal_length, tform_cam2world):
        def meshgrid_xy(tensor1, tensor2):
            ii, jj = torch.meshgrid(tensor1, tensor2, indexing='ij')
            return ii.transpose(-1, -2), jj.transpose(-1, -2)
        ii, jj = meshgrid_xy(torch.arange(width).to(tform_cam2world), torch.arange(height).to(tform_cam2world))
        directions = torch.stack([(ii - width * .5) / focal_length, -(jj - height * .5) / focal_length, -torch.ones_like(ii)], dim=-1)
        ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
        ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
        return ray_origins, ray_directions

    is_train = 1
    dataset_train = SuperVisionDataset(is_train=is_train, data_path='./data/image/resin/lion/images_split/train/', image_size=image_size)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=bool(is_train), num_workers=1, drop_last=bool(is_train), collate_fn=None, pin_memory=False)

    #images,masks,poses = next(iter(dataloader_train))
    #print('images', images.shape)    #[-1, 32, 32, 3]
    #print('masks',  masks.shape)     #[-1, 32, 32, 1]
    #print('poses',  poses)           #[4, 4]
 
    #data = np.load('./data/nerf/tiny_nerf_data.npz')  #TODO  render and load objavase for training
    #images, poses, focal = data["images"],data["poses"], data["focal"]
    #print('images', images.shape)  #(106, 100, 100, 3)
    #print('poses', poses.shape)    #(106, 4, 4)
    #print('focal', focal)          #138.88887889922103

    #height, width = images.shape[1:3]   #(-1, 224, 224, 3)
    #near_thresh, far_thresh = 1, 1000   #2.0, 6.0

    #images = torch.from_numpy(images[..., :3]).to(device)
    #poses = torch.from_numpy(poses).to(device)
    #focal = torch.from_numpy(focal).to(device)  #138.88887889922103
    focal_length = 351.6771/4

    #image_processor = ImagePreprocessor(image_size)

    from network import TSR
    #model = TSR(img_size=image_size, depth=16, embed_dim=768, num_channels=1024, num_layers=16, cross_attention_dim=768, radius=3, valid_thresh=0.001, num_samples_per_ray=128, n_hidden_layers=9, official=True)
    model = TSR(img_size=image_size, depth=16//2, embed_dim=768, num_channels=1024, num_layers=16//2, cross_attention_dim=768, radius=99, valid_thresh=0.00001, num_samples_per_ray=128, n_hidden_layers=9, official=False)
    #model.load_state_dict(torch.load('./ckpt/TripoSR/model.ckpt', map_location='cpu'))
    model.to(device)
    model.train()
    
    print("parameters: %d M"%(sum(p.numel() for p in model.parameters())/1024/1024))

    #import torchsummary  #pip install torchsummary
    #torchsummary.summary(model, input_size=(1, image_size, image_size, 3), batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, epochs, epochs//10 if epochs>10 else 1)), gamma=0.95)

    def save_image(image_I, image_O, alpha_I, alpha_O, flag, epoch, index):
        temp_path = './outs/image/'+flag+'/'
        os.makedirs(temp_path, exist_ok=True)
        image_i = image_I.detach().cpu().numpy()
        image_o = image_O.detach().cpu().numpy()
        alpha_i = alpha_I.detach().cpu().numpy().repeat(repeats=3,axis=2)
        alpha_o = alpha_O.detach().cpu().numpy().repeat(repeats=3,axis=2)
        import PIL
        PIL.Image.fromarray((image_i*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__image_1i.png'.format(epoch,index,flag))
        PIL.Image.fromarray((image_o*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__image_2o.png'.format(epoch,index,flag))
        PIL.Image.fromarray((alpha_i*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__alpha_1i.png'.format(epoch,index,flag))
        PIL.Image.fromarray((alpha_o*255).astype('uint8')).save(temp_path+'/image__epoch_{:04d}__index_{:04d}__{:s}__alpha_2o.png'.format(epoch,index,flag))

    for epoch in range(0, epochs):
        losses = []
        for index, (images,masks,poses) in enumerate(dataloader_train):
            target_img = images.to(device)
            target_msk = masks.to(device)
            target_pos = poses.to(device)

            target_img = target_img[:, None]  #[-1, 32, 32, 3]  ->  [-1, 1, 32, 32, 3]
            target_msk = target_msk[:, None]  #[-1, 32, 32, 1]  ->  [-1, 1, 32, 32, 1]
            print('target_img', target_img.shape)
            print('target_msk', target_msk.shape)

            scene_codes = model(target_img)
            #print('scene_codes', scene_codes.shape)  #[-1, 3, 40, 64, 64]

            images_all = []
            masks_all = []
            for idx, scene_code in enumerate(scene_codes):
                images_one = []
                masks_one = []
                for i in range(1):
                    rays_o, rays_d = get_ray_bundle(height=image_size, width=image_size, focal_length=focal_length, tform_cam2world=target_pos[idx])  #[32, 32, 3]  [32, 32, 3]
                    image, alpha = model.renderer(model.decoder, scene_code, rays_o, rays_d)
                    images_one.append(image)
                    masks_one.append(alpha)
                images_all.append(torch.stack(images_one, dim=0))
                masks_all.append(torch.stack(masks_one, dim=0))
            image_pred = torch.stack(images_all, dim=0)
            mask_pred = torch.stack(masks_all, dim=0)
            print('image_pred', image_pred.shape)  #[-1, 1, 32, 32, 3]
            print('mask_pred', mask_pred.shape)  #[-1, 1, 32, 32, 1]

            loss_img = torch.nn.functional.mse_loss(target_img, image_pred)
            loss_msk = torch.nn.functional.mse_loss(target_msk, mask_pred)
            loss = loss_img + loss_msk
            print('loss', loss.item(), '', loss_img.item(), loss_msk.item())

            #TODO loesses in TripoSR's:  #Loss = (1/n other views mse + current view lpisp + current view mask bce)      #Local Rendering Supervision(patchs)   

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   

            if index%1==0:
                save_image(target_img[0][0], image_pred[0][0], target_msk[0][0], mask_pred[0][0], flag='train', epoch=epoch, index=index)  

                mesh_path = './outs/image/'+'train'+'/'
                os.makedirs(mesh_path, exist_ok=True)
                mesh = model.extract_mesh(scene_codes)[0]
                mesh.export(os.path.join(mesh_path, "mesh__epoch_{:04d}.obj".format(epoch)))  #.ply          

        LOSS_train = sum(losses)/len(losses)
        print('epoch=%06d  loss=%.6f'%(epoch, LOSS_train))

        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        #torch.cuda.empty_cache()

def main(device=['cpu','cuda'][torch.cuda.is_available()]):
    #infer(image_size=512, image_path=['./data/image/test/images/Lion.png'][0], output_file='./outs/stereo/test/Lion.obj', remove_bg=True, foreground_ratio=0.85, render_video=True, device=device)
    train(image_size=[512,64][1], batch_size=1, epochs=10, checkpoint_path='./outs/ckpt/', best_checkpoint_file='./outs/ckpt/checkpoint.pth', device=device)

if __name__ == '__main__':  #cls; python -Bu superv.py
    main()

# wget https://huggingface.co/facebook/dino-vitb16/tree/main/*  ./ckpt/dino-vitb16/*
# wget https://huggingface.co/stabilityai/TripoSR/blob/main/* ./ckpt/TripoSR/*

