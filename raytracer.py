import torch

shaded = lambda im: (im != 0).any(dim=-1)

class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir
        self.dim = origin.shape[:2]

class Scene:
    def __init__(self, background, ambient, objects):
        self.background = torch.tensor(background)
        self.ambient = torch.tensor(ambient)
        self.objects = objects

class Camera:
    def __init__(self, center, target, focus=300, width=200, height=100, ppi=1):
        self.center = torch.tensor(center)
        self.target = torch.tensor(target)
        self.focus = focus
        self.width = width
        self.height = height
        self.ppi = ppi
    
    def gen_ray(self):
        width = self.width
        height = self.height
        ppi = self.ppi

        x, y, z = self.target - self.center
        r = torch.sqrt(x**2 + y**2 + z**2)
        rr = torch.sqrt(x**2 + y**2)
        ry = torch.tensor([[c:=rr/r,0,s:=-z/r],[0,1,0],[-s,0,c]])
        rz = torch.tensor([[c:=x/rr,s:=-y/rr,0],[-s,c,0],[0,0,1]])

        w = torch.arange(-width/2, width/2, 1/ppi).view([1, -1]).repeat([height*ppi, 1])
        h = torch.arange(-height/2, height/2, 1/ppi).view([-1, 1]).repeat([1, width*ppi])
        d = torch.zeros([height*ppi, width*ppi])
        origin = torch.stack([d, -w, -h], dim=-1)
        eye = torch.tensor([-self.focus, 0, 0])
        dir = origin - eye
        dir = dir @ ry.T @ rz.T
        dir = dir / dir.norm(dim=-1, keepdim=True)

        origin = origin @ ry.T @ rz.T
        origin = origin + self.center

        return Ray(origin, dir)
    

class Sphere:
    def __init__(self, center, radius, ambient):
        self.center = torch.tensor(center)
        self.radius = torch.tensor(radius)
        self.ambient = torch.tensor(ambient)
    
    def intersect(self, ray):
        ray_to_center = self.center - ray.origin
        dist = ray_to_center.norm(dim=-1)
        c = (ray.dir * ray_to_center).sum(dim=-1)
        s = dist**2 - c**2

        hit_point_dist = c - torch.sqrt(self.radius**2 - s)
        hit_point_dist[hit_point_dist.isnan()] = torch.inf
        return hit_point_dist

    def shade(self, ray, mask, z_buffer, scene):
        color = torch.zeros([*ray.dim, 3])
        color[mask] = self.ambient * scene.ambient

        return color


def trace(scene, ray):
    z_buffer = [object.intersect(ray) for object in scene.objects]
    z_buffer = torch.stack(z_buffer, dim=0)
    first_hit_idx = torch.argmin(z_buffer, dim=0)

    image = torch.zeros([*ray.dim, 3])
    for idx, obj in enumerate(scene.objects):
        mask = first_hit_idx == idx
        mask = torch.logical_and(z_buffer[idx].isinf().logical_not(), mask)
        shade = obj.shade(ray, mask, z_buffer[idx], scene)
        assert not torch.logical_and(mask.logical_not(), shaded(shade)).any(), "Shading outside of interection!"
        image += shade
    
    return image

    
def render(scene, camera):
    ray = camera.gen_ray()
    image = trace(scene, ray)
    image[shaded(image).logical_not()] = scene.background
    return image
