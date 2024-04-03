import torch
from torch.nn.functional import normalize

RAYTRACE_BOUNCE = 20
HIT_TOO_CLOSE = 0.1


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
    def __init__(self, intrinsic, diffuse, specular, specular_n):
        self.intrinsic = torch.tensor(intrinsic)
        self.diffuse = torch.tensor(diffuse)
        self.specular = torch.tensor(specular)
        self.specular_n = torch.tensor(specular_n)
    
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

        inner_prod = (normal * to_light).sum(-1, keepdim=True)
        inner_prod[inner_prod < 0] = 0
        inner_prod[occluded] = 0
        diffuse = self.diffuse * inner_prod * light_color

        reflection_light = 2 * normal * (to_light * normal).sum(-1, keepdim=True) - to_light
        inner_prod = (-ray.dir * reflection_light).sum(dim=-1, keepdim=True)
        inner_prod[inner_prod < 0] = 0
        inner_prod[occluded] = 0
        specular = self.specular * inner_prod**self.specular_n * light_color

        if(level > 1):
            x_sq = torch.rand([RAYTRACE_BOUNCE,ray.len])
            y = (1 - x_sq).sqrt()
            phi = torch.rand_like(x_sq) * 2 * torch.pi
            wave_packet = torch.stack([x_sq.sqrt(), y * phi.cos(), y * phi.sin()], dim=-1)
            inner_prod = wave_packet[...,0].unsqueeze(-1)
            rotation = rotate(normal)
            wave_packet = (rotation @ wave_packet.unsqueeze(-1)).squeeze(-1)
            bounce_ray = Ray(hit_point.repeat([RAYTRACE_BOUNCE, 1]), wave_packet.view([RAYTRACE_BOUNCE * ray.len, 3]))
            traced = trace(scene, bounce_ray, level=level-1)
            traced = traced.view([RAYTRACE_BOUNCE, ray.len, 3])
            diffuse_traced = self.diffuse * (inner_prod * traced).mean(dim=0)

            x = torch.rand([RAYTRACE_BOUNCE,ray.len]) ** (1/(self.specular_n+1))
            x_sq = x ** 2
            y = (1 - x_sq).sqrt()
            wave_packet = torch.stack([x_sq.sqrt(), y * phi.cos(), y * phi.sin()], dim=-1)
            inner_prod = wave_packet[...,0].unsqueeze(-1)
            eye_reflection = ray.dir - 2 * normal * (ray.dir * normal).sum(-1, keepdim=True)
            rotation = rotate(eye_reflection)
            wave_packet = (rotation @ wave_packet.unsqueeze(-1)).squeeze(-1)
            bounce_ray = Ray(hit_point.repeat([RAYTRACE_BOUNCE, 1]), wave_packet.view([RAYTRACE_BOUNCE * ray.len, 3]))
            traced = trace(scene, bounce_ray, level=level-1)
            traced = traced.view([RAYTRACE_BOUNCE, ray.len, 3])
            specular_traced = self.specular * (inner_prod**self.specular_n * traced).mean(dim=0)


        color = torch.zeros([ray.len, 3])
        color += scene.ambient
        color += diffuse.sum(dim=0)
        color += specular.sum(dim=0)
        if(level > 1):
            color += diffuse_traced
            color += specular_traced
        color *= self.intrinsic
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
        hit_dist[hit_dist < HIT_TOO_CLOSE] = torch.inf
        return hit_dist

    def normal(self, ray, *args):
        sign = -(self.normal_vec * ray.dir).sum(dim=-1, keepdim=True).sgn()
        return self.normal_vec * sign

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
        hit_dist[hit_dist < HIT_TOO_CLOSE] = torch.inf
        return hit_dist

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
        return image


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
