import torch
from torch.nn.functional import normalize

shaded = lambda im: (im != 0).any(dim=-1)

class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir
        self.dim = origin.shape[:2]

class Scene:
    def __init__(self, background, ambient, objects, lights):
        self.background = torch.tensor(background)
        self.ambient = torch.tensor(ambient)
        self.objects = objects
        self.lights = lights

class Light():
    def __init__(self, center, color):
        self.center = torch.tensor(center)
        self.color = torch.tensor(color)

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
        dir = normalize(dir @ ry.T @ rz.T, dim=-1)

        origin = origin @ ry.T @ rz.T
        origin = origin + self.center

        return Ray(origin, dir)

class Object:
    def __init__(self, ambient, diffuse, specular, specular_n):
        self.ambient = torch.tensor(ambient)
        self.diffuse = torch.tensor(diffuse)
        self.specular = torch.tensor(specular)
        self.specular_n = torch.tensor(specular_n)
    
    def shade(self, ray, mask, hit_dist, scene):
        ambient = self.ambient * scene.ambient

        light_center = [light.center for light in scene.lights]
        light_center = torch.stack(light_center, dim=0).unsqueeze(0)
        light_color = [light.color for light in scene.lights]
        light_color = torch.stack(light_color, dim=0).unsqueeze(0)

        hit_point = ray.origin + ray.dir * hit_dist.unsqueeze(-1)
        hit_point = hit_point[mask].unsqueeze(1)
        normal = self.normal(ray, hit_point)

        to_light = normalize(light_center - hit_point, dim=-1)
        inner_prod = (normal * to_light).sum(-1).unsqueeze(-1)
        inner_prod[inner_prod < 0] = 0
        diffuse = self.diffuse * inner_prod * light_color

        reflection_light = normalize(2 * normal - to_light, dim=-1)
        inner_prod = (-ray.dir[mask].unsqueeze(1) * reflection_light).sum(dim=-1, keepdim=True)
        inner_prod[inner_prod < 0] = 0
        specular = self.specular * inner_prod**self.specular_n * light_color

        color = torch.zeros([*ray.dim, 3])
        color[mask] += ambient + diffuse.sum(dim=1) + specular.sum(dim=1)
        return color

class Rect(Object):
    def __init__(self, center, width, height, width_dir, normal, **kwargs):
        super().__init__(**kwargs)
        self.center = torch.tensor(center)
        self.width = width
        self.height = height
        self.width_dir = torch.tensor(width_dir)
        self.normal_vec = torch.tensor(normal)
        assert self.width_dir @ self.normal_vec == 0, "normal vector and width direction not perpendicular"

    def intersect(self, ray):
        origin_to_center = self.center - ray.origin
        origin_to_center_dist = (origin_to_center * self.normal_vec).sum(dim=-1,)
        ray_normal_angle = (ray.dir * self.normal_vec).sum(dim=-1)
        hit_dist = origin_to_center_dist / ray_normal_angle
        hit_point = ray.origin + ray.dir * hit_dist.unsqueeze(dim=-1)
        center_to_hit = hit_point - self.center
        w_sq = (center_to_hit * self.width_dir).sum(dim=-1)**2
        h_sq = (center_to_hit**2).sum(dim=-1) - w_sq

        hit_dist[torch.logical_or(w_sq > self.width**2/4, h_sq > self.height**2/4)] = torch.inf
        hit_dist[hit_dist < 0] = torch.inf
        return hit_dist

    def normal(self, *args):
        return self.normal_vec

class Sphere(Object):
    def __init__(self, center, radius, **kwargs):
        super().__init__(**kwargs)
        self.center = torch.tensor(center)
        self.radius = torch.tensor(radius)

    def intersect(self, ray):
        ray_to_center = self.center - ray.origin
        dist = ray_to_center.norm(dim=-1)
        c = (ray.dir * ray_to_center).sum(dim=-1)
        s = dist**2 - c**2

        hit_dist = c - torch.sqrt(self.radius**2 - s)
        hit_dist[hit_dist.isnan()] = torch.inf
        return hit_dist

    def normal(self, ray, hit_point):
        normal = (hit_point - self.center) / self.radius
        return normal


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
    image[image >= 1] = 1
    return image
