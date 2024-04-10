from math import ceil
import torch
from torch.nn.functional import normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

HIT_TOO_CLOSE = 0.1
EPSILON = 1e-3

inner = lambda a, b: (a * b).sum(-1, keepdim=True)

class Ray:
    def __init__(self, origin, dir):
        self.origin = origin
        self.dir = dir
        self.len = len(origin)
    def __getitem__(self, mask):
        return Ray(self.origin[mask], self.dir[mask])

class Scene:
    def __init__(self, background, ambient, objects, lights):
        self.background = torch.tensor(background)
        self.ambient = torch.tensor(ambient)
        self.objects = objects
        self.lights = lights
        self.total_rays = 0

class Light():
    def __init__(self, center, color):
        self.center = torch.tensor(center)
        self.color = torch.tensor(color)

class Object:
    def __init__(self, intrinsic, diffuse=0.3, specular=0.1, specular_n=3,
                 refraction=0, refraction_index=1):
        self.intrinsic = torch.tensor(intrinsic)
        self.diffuse_factor = diffuse
        self.specular_factor = specular
        self.specular_n = specular_n
        self.refraction_factor = refraction
        self.refraction_index = refraction_index
    
    def shade(self, ray, hit_dist, scene, level):
        light_center = [light.center for light in scene.lights]
        light_center = torch.stack(light_center, dim=0).unsqueeze(1)
        light_color = [light.color for light in scene.lights]
        light_color = torch.stack(light_color, dim=0).unsqueeze(1)

        hit_point = ray.origin + ray.dir * hit_dist.unsqueeze(-1)

        #[hit, 3]
        normal = self.normal(ray, hit_point)

        #[l, hit, 3]
        to_light = normalize(light_center - hit_point, dim=-1)
        to_light_ray = Ray(hit_point.repeat([len(to_light), 1]), to_light.view([-1, 3]))
        z_buffer = [object.intersect(to_light_ray) for object in scene.objects]
        z_buffer = torch.stack(z_buffer, dim=0)
        occluded = z_buffer.isinf().all(dim=0).logical_not().view(to_light.shape[:2])
        to_light[occluded] = 0


        diffuse = self.diffuse(ray, to_light, normal, light_color)
        specular = self.specular(ray, to_light, normal, light_color)
        refraction = self.refraction(ray, to_light, normal, light_color)

        diffuse_traced = self.diffuse_traced(ray, hit_point, normal, scene, level)
        specular_traced = self.specular_traced(ray, hit_point, normal, scene, level)
        refraction_traced = self.refraction_traced(ray, hit_point, normal, scene, level)

        color = torch.zeros_like(ray.dir)
        color += scene.ambient
        color += diffuse
        color += specular
        color += refraction
        color += diffuse_traced
        color += specular_traced
        color += refraction_traced
        color *= self.intrinsic
        return color

    def diffuse(self, ray, to_light, normal, light_color):
        if (self.diffuse_factor == 0):
            return 0

        inner_prod = inner(normal, to_light)
        inner_prod[inner_prod < 0] = 0
        diffuse = self.diffuse_factor * (inner_prod * light_color).sum(dim=0)
        return diffuse

    def specular(self, ray, to_light, normal, light_color):
        if (self.specular_factor == 0):
            return 0

        reflection_light = 2 * normal * inner(to_light, normal) - to_light
        inner_prod = inner(-ray.dir, reflection_light)
        inner_prod[inner_prod < 0] = 0
        specular = self.specular_factor * (inner_prod**self.specular_n * light_color).sum(dim=0)
        return specular

    def refraction(self, ray, to_light, normal, light_color):
        if (self.refraction_factor == 0):
            return 0
        
        c = inner(to_light, normal).abs()
        idx = self.refraction_index ** -c.sgn()
        factor = idx * c.abs() - (1 + idx**2 * (c**2 - 1)).sqrt()
        refraction_light = idx * -to_light + factor * normal
        inner_prod = inner(-ray.dir, refraction_light)
        inner_prod[inner_prod < 0] = 0
        refraction = self.refraction_factor * (inner_prod**self.specular_n * light_color).sum(dim=0)
        return refraction
    
    def diffuse_traced(self, ray, hit_point, normal, scene, level):
        if (self.diffuse_factor == 0):
            return 0
        if(level == 0):
            return 0

        traced = gen_and_trace_ray(normal, 1, self.diffuse_factor, hit_point, scene, level)
        return traced

    def specular_traced(self, ray, hit_point, normal, scene, level):
        if (self.specular_factor == 0):
            return 0
        if(level == 0):
            return 0

        eye_reflection = ray.dir - 2 * normal * inner(ray.dir, normal)
        traced = gen_and_trace_ray(eye_reflection, self.specular_n, self.specular_factor, hit_point, scene, level)
        return traced

    def refraction_traced(self, ray, hit_point, normal, scene, level):
        if (self.refraction_factor == 0):
            return 0
        if(level == 0):
            return 0

        c = inner(-ray.dir, normal)
        idx = self.refraction_index ** -c.sgn()
        factor = idx * c.abs() - (1 + idx**2 * (c**2 - 1)).abs().sqrt()
        eye_refraction = idx * ray.dir + factor * normal
        traced = gen_and_trace_ray(eye_refraction, self.specular_n, self.refraction_factor, hit_point, scene, level)
        return traced
    
class Rect(Object):
    def __init__(self, center, width, height, width_dir, normal, **kwargs):
        super().__init__(**kwargs)
        self.center = torch.tensor(center)
        self.width = width
        self.height = height
        self.width_dir = torch.tensor(width_dir)
        self.normal_vec = torch.tensor(normal)
        assert inner(self.width_dir, self.normal_vec) == 0, "normal vector and width direction not perpendicular"

    def intersect(self, ray):
        origin_to_center = self.center - ray.origin
        origin_to_plane_dist = inner(origin_to_center, self.normal_vec)
        ray_normal_angle = inner(ray.dir, self.normal_vec)
        hit_dist = origin_to_plane_dist / ray_normal_angle
        hit_point = ray.origin + ray.dir * hit_dist
        center_to_hit = hit_point - self.center
        w_sq = inner(center_to_hit, self.width_dir) ** 2
        h_sq = inner(center_to_hit, center_to_hit) - w_sq

        hit_dist[torch.logical_or(w_sq > self.width**2/4, h_sq > self.height**2/4)] = torch.inf
        hit_dist[hit_dist < HIT_TOO_CLOSE] = torch.inf
        return hit_dist.squeeze(-1)

    def normal(self, ray, *args):
        sign = -inner(self.normal_vec, ray.dir).sgn()
        return self.normal_vec * sign

class Sphere(Object):
    def __init__(self, center, radius, **kwargs):
        super().__init__(**kwargs)
        self.center = torch.tensor(center)
        self.radius = radius

    def intersect(self, ray):
        ray_to_center = self.center - ray.origin
        dist_sq = inner(ray_to_center, ray_to_center)
        c = inner(ray.dir, ray_to_center)
        s = dist_sq - c**2
        
        to_center = c > 0
        outside = dist_sq > (self.radius**2 + HIT_TOO_CLOSE * (to_center * 2 - 1))
        
        proj_to_hit = torch.sqrt(self.radius**2 - s)
        proj_to_hit *= torch.logical_and(outside, to_center) * -2 + 1

        hit_dist = c + proj_to_hit
        hit_dist[hit_dist.isnan()] = torch.inf
        hit_dist[hit_dist <= EPSILON] = torch.inf
        return hit_dist.squeeze(-1)

    def normal(self, ray, hit_point):
        normal = (hit_point - self.center) / self.radius
        return normal


class Camera:
    def __init__(self, center, target, focus=300):
        self.center = torch.tensor(center)
        self.target = torch.tensor(target)
        self.focus = focus
    
    def gen_ray(self, width, height, ppi, super_sample):
        image_dim = [height * ppi * super_sample, width * ppi * super_sample]

        rotation = rotate(self.target - self.center)

        w = torch.arange(-width/2, width/2, 1/(ppi * super_sample)).view([1, -1]).expand(image_dim)
        h = torch.arange(-height/2, height/2, 1/(ppi * super_sample)).view([-1, 1]).expand(image_dim)
        d = torch.zeros(image_dim)
        origin = torch.stack([d, -w, -h], dim=-1).view([-1, 3])
        eye = torch.tensor([-self.focus, 0, 0])
        dir = origin - eye
        dir = normalize(dir @ rotation.T, dim=-1)

        origin = origin @ rotation.T
        origin = origin + self.center

        return Ray(origin, dir)

        
    def render(self, scene, width=200, height=100, ppi=2, super_sample=2):
        ray = self.gen_ray(width, height, ppi, super_sample)
        image = trace(scene, ray, level=2)
        image[(image == 0).all(dim=-1)] = scene.background
        image[image >= 1] = 1
        image = image.view([height * ppi * super_sample, width * ppi * super_sample, 3])
        if(super_sample > 1):
            image = image.permute([2,0,1])
            image = torch.nn.functional.conv2d(image, torch.ones([3, 1, super_sample, super_sample]) / super_sample ** 2, stride=super_sample, groups=3)
            image = image.permute([1,2,0])
        return image.cpu()


def gen_and_trace_ray(center, concentration, intensity, hit_point, scene, level):
    if(concentration == 1):
        num_bounce = 30
    elif(concentration <= 50):
        num_bounce = 15
    elif(concentration <= 1e2):
        num_bounce = 5
    else:
        num_bounce = 1

    num_bounce = num_bounce ** (level/2)
    num_bounce = ceil(num_bounce * intensity)
    
    if num_bounce == 1:
        x = torch.ones([num_bounce, len(hit_point)])
    else:
        x = torch.rand([num_bounce, len(hit_point)]) ** (1/(2*concentration))
    
    y = (1 - x**2).sqrt()
    phi = torch.rand_like(x) * 2 * torch.pi
    wave_packet = torch.stack([x, y * phi.cos(), y * phi.sin()], dim=-1)

    wave_packet = (rotate(center) @ wave_packet.unsqueeze(-1)).squeeze(-1)
    bounce_ray = Ray(hit_point.repeat([num_bounce, 1]), wave_packet.view([num_bounce * len(hit_point), 3]))
    traced = trace(scene, bounce_ray, level=level-1)
    traced = traced.view([num_bounce, len(hit_point), 3])
    traced = intensity * (x.unsqueeze(-1) ** concentration * traced).mean(0)
    return traced

def trace(scene, ray, level):
    scene.total_rays += ray.len
    image = torch.zeros([ray.len, 3])
    
    z_buffer = [object.intersect(ray) for object in scene.objects]
    z_buffer = torch.stack(z_buffer, dim=0)
    first_hit_idx = torch.argmin(z_buffer, dim=0)

    for idx, obj in enumerate(scene.objects):
        is_hit = z_buffer[idx].isinf().logical_not()
        mask = torch.logical_and(first_hit_idx == idx, is_hit)
        if mask.count_nonzero() == 0:
            continue
        shade = obj.shade(ray[mask], z_buffer[idx][mask], scene, level)
        image[mask] += shade
    return image

def rotate(vec):
    x, y, z = vec.split(1, dim=-1)
    r = torch.sqrt(x**2 + y**2 + z**2)
    rr = torch.sqrt(x**2 + y**2)
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    
    c, s = rr / r, -z / r
    ry = torch.stack([torch.concat([c, zero, s], dim=-1),
                    torch.concat([zero, one, zero], dim=-1),
                    torch.concat([-s, zero, c], dim=-1)], dim=-2)
    
    c, s = x / rr, y / rr
    singular = (x**2 + y**2) == 0
    c[singular] = 1
    s[singular] = 0
    rz = torch.stack([torch.concat([c, -s, zero], dim=-1),
                    torch.concat([s,c, zero], dim=-1),
                    torch.concat([zero, zero, one], dim=-1)], dim=-2)
    return rz @ ry
