import time
import matplotlib.pyplot as plt

from raytracer import Camera, Light, Rect, Sphere, Scene

scene = Scene(
    background = [0, 0, 0.],
    ambient = [0.1, 0.1, 0.1],
    objects = [
        Rect([80, 0, 0], 160, 140, [0,1,0], [0,0,1],
            intrinsic=[1, 1, 1], diffuse=0.3, specular=0.8, specular_n=50),
        Rect([80, -80, 30], 140, 60, [1,0,0], [0,-1,0],
            intrinsic=[0.7, 0.3, 0.1], diffuse=0.6, specular=0.2, specular_n=50),
        Rect([150, 0, 30], 160, 60, [0,1,0], [1,0,0],
            intrinsic=[1., 1., 1.], diffuse=0., specular=0.9, specular_n=1e3),
        Sphere([70, 00, 20], 20, 
            intrinsic=[0.7, 0.7, 1], diffuse=0.8, specular=0.4, specular_n=4),
        Sphere([50, -40, 20], 20, 
            intrinsic=[1, 0.8, 0.0], diffuse=0.6, specular=0.9, specular_n=1e2),
        Sphere([85, 45, 20], 20, 
            intrinsic=[0.9, 0., 0.6], diffuse=0.3, specular=0.5, specular_n=2e2,
            refraction=0.9, refraction_index=1.25),
],
    lights = [Light([30, 20, 100], [1, 1, 1])]
)

camera = Camera([0,0,50], [100,0,10])
start_time = time.time()
image = camera.render(scene, width=180, height=100, ppi=5, super_sample=3, recursive=2)
end_time = time.time()
print(f"Traced {scene.total_rays} rays. Took {end_time - start_time:.2f} seconds.")
plt.imsave("demo.png", image.numpy())
