import torch
import pytorch3d

class Data:
    class Obj:  #vertex_color
        def load(file):
            vertexes = []
            faces = []
            with open(file, 'r') as handler:
                for line in handler:
                    line = line.strip().split()
                    if len(line)>0:
                        if line[0]=='v':
                            if len(line)==1+3:
                                vertex = [float(one) for one in line[1:1+3]]
                            elif len(line)==1+3+3:
                                vertex = [float(one) for one in line[1:1+3+3]]
                            else:
                                raise                        
                            vertexes.append(vertex)
                        elif line[0]=='f':
                            face = []
                            for two in line[1:]:
                                one = two.split('/')
                                face.append(float(one[0]) if '.' in one[0] else int(one[0])) 
                            faces.append(face)               
            return vertexes, faces
                
        def save(file, vertexes, faces, float_color, v_format=''):
            import os; os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, 'w') as handler: 
                for vertex in vertexes:
                    if len(vertex)==3:
                        handler.write(('v {'+v_format+'} {'+v_format+'} {'+v_format+'}\n').format(vertex[0], vertex[1], vertex[2]))
                    elif len(vertex)==3+3:
                        if float_color:
                            handler.write('v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(vertex[0], vertex[1], vertex[2], vertex[3], vertex[4], vertex[5]))
                        else:
                            handler.write('v {:.6f} {:.6f} {:.6f} {:d} {:d} {:d}\n'.format(vertex[0], vertex[1], vertex[2], int(vertex[3]), int(vertex[4]), int(vertex[5])))
                    else:
                        raise
                for face in faces:
                    handler.write('f {} {} {}'.format(face[0], face[1], face[2])+(' {}'.format(face[3]) if len(face)==1+3 else '')+'\n')

    def load(obj_file, device, is_vertex_color):
        if is_vertex_color:
            vertexes_coords_colors, faces = Data.Obj.load(obj_file)
            vertexes_coords_colors = torch.tensor(vertexes_coords_colors).to(device)
            if vertexes_coords_colors.shape[1]==6:
                colors = [vertexes_coords_colors[:,3:6] / (1.0 if torch.max(vertexes_coords_colors[:,3:6])<=1.0 else 255.0)]  #MUST:/255
            else:
                colors = [torch.full([vertexes_coords_colors.shape[0], 3], 0.5, device=device)]
            from pytorch3d.renderer import TexturesVertex
            textures=TexturesVertex(verts_features=colors)
            faces = [(torch.tensor(faces)-1).to(device)]   #MUST:-1
            vertexes = [vertexes_coords_colors[:,0:3]]
        else:
            from pytorch3d.io import load_obj
            vertexes, faces, aux = load_obj(obj_file, device=device, load_textures=True, create_texture_atlas=False, texture_atlas_size=4, texture_wrap="repeat", path_manager=None)
            verts_uvs = aux.verts_uvs.to(device)  #(V, 2)
            faces_uvs = faces.textures_idx.to(device)  #(F, 3)

            if len(aux.texture_images.items())>1: raise Exception('not support multiple UV')

            image = list(aux.texture_images.values())[0].to(device)[None]
            from pytorch3d.renderer import TexturesUV
            textures = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)

            faces = [faces.verts_idx.to(device)]
            vertexes = [vertexes.to(device)]
        return vertexes, faces, textures

    def save(obj_file, verts, faces, float_color, use_obj=1):
        import os; os.makedirs(os.path.dirname(obj_file), exist_ok=True)
        if use_obj:
            Data.Obj.save(obj_file, verts, faces+1, float_color=float_color, v_format=':.6f')   #MUST:+1
        else:
            from pytorch3d.io import save_obj
            save_obj(obj_file, verts, faces)

class View:
    def __init__(self, obj_file, is_vertex_color, device, normalize_vertexes=True):
        vertexes, faces, textures = Data.load(obj_file, device, is_vertex_color=is_vertex_color)
        self.mesh = pytorch3d.structures.Meshes(vertexes, faces, textures)
        if normalize_vertexes:
            vertexes = self.mesh.verts_packed()
            center = vertexes.mean(0); scale = max((vertexes - center).abs().max(0)[0])  
            self.mesh.offset_verts_(-center); self.mesh.scale_verts_((1.0 / float(scale)))    #recover: vertexes_normalized * scale + center
        self.lights = pytorch3d.renderer.AmbientLights(device=device)  #pytorch3d.renderer.DirectionalLights(direction=[[0.0,+3.0,0.0]], device=device)  #pytorch3d.renderer.PointLights(location=[[0.0,0.0,-3.0]], device=device)

    def look(self, distance, elevation, azimuth_all, device, image_size):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=elevation, azim=azimuth_all)  #distance:距离(远)  elevation:海拔(高)  azimuth:方位(平)-180,180
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)  #FoVPerspectiveCameras  FoVOrthographicCameras
        mesh_renderer = pytorch3d.renderer.MeshRenderer(rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=pytorch3d.renderer.RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)), shader=pytorch3d.renderer.SoftPhongShader(lights=self.lights, cameras=cameras, device=device))  #SoftPhongShader  HardPhongShader
        rendered_rgba = mesh_renderer(meshes_world=self.mesh.extend(azimuth_all.shape[0]), lights=self.lights, cameras=cameras)  #materials=materials
        #print('rendered_rgba', rendered_rgba.shape)
        #alpha_mask = (rendered_rgb[..., 3] > 0)
        #rendered_rgba = torch.cat((rendered_rgb, alpha_mask.unsqueeze(-1)), dim=-1)
        return rendered_rgba

def mesh(obj_file, is_vertex_color, image_path, elevation_number=24, azimuth_number=12, distance=2.0, image_size=512, step=4, device=['cpu','cuda'][torch.cuda.is_available()]):
    def save(rendered_rgba_all, elevation, azimuth_all, image_path):
        for rendered_rgba,azimuth in zip(rendered_rgba_all, azimuth_all):
            save_file=image_path+'/image__distance_%s__elevation_%03d__azimuth_%03d.png'%(str(distance).replace('.','_'),elevation,azimuth)
            import os; os.makedirs(os.path.dirname(save_file), exist_ok=True)
            from PIL import Image
            rendered_rgba[...,3] = (rendered_rgba[...,3]>=0.5)  #灰白/半透明->全透明
            rgba = (rendered_rgba[...,:].cpu().clamp(0.0,1.0)*255.0).numpy().astype('uint8')
            Image.frombuffer("RGBA", rgba.shape[0:2], rgba, "raw", "RGBA", 0, 1).save(save_file)
    import os
    if not os.path.exists(image_path):  
        print('render', 'todo', obj_file)
        elevation_all = torch.linspace(0, 360-(360//elevation_number), elevation_number)
        azimuth_all = torch.linspace(0, 360-(360//azimuth_number), azimuth_number)   
        view = View(obj_file=obj_file, is_vertex_color=is_vertex_color, device=device)
        for elevation in elevation_all:
            for i in range(0, len(azimuth_all), step):
                print('render','','elevation:', elevation.item(), '', 'azimuth:',azimuth_all[i:i+step])
                rendered_rgba_all = view.look(distance=distance, elevation=elevation, azimuth_all=azimuth_all[i:i+step], device=device, image_size=image_size)
                save(rendered_rgba_all=rendered_rgba_all, elevation=elevation, azimuth_all=azimuth_all[i:i+step], image_path=image_path)
    else:
        print('render', 'skip', obj_file)

def main():
    for obj in ['allosaurus','bull','eagle','lion','rhino','spinosaurus','therizinosaurus','unicorn']:
        mesh(obj_file='./data/mesh/resin/'+obj+'/model.obj', is_vertex_color=0, image_path='./data/image/resin/'+obj+'/images_mesh/')

if __name__ == '__main__':    #conda install fvcore iopath pytorch3d -c fvcore -c iopath -c pytorch3d
    main()
