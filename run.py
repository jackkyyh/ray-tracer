import matplotlib.pyplot as plt

from raytracer import Camera, Light, Rect, Sphere, Scene

scene = Scene(
    background = [0, 0, 0.],
    ambient = [0.15, 0.15, 0.15],
    objects = [
        Rect([80, 80, 30], 140, 60, [1,0,0], [0,-1,0],
            intrinsic=[0.9, 0.4, 0.2], diffuse=0.3, specular=0, specular_n=0),
        Rect([80, 0, 0], 160, 140, [0,1,0], [0,0,1],
            intrinsic=[1, 1, 1], diffuse=0.3, specular=0.8, specular_n=50),
        Rect([80, -80, 30], 140, 60, [1,0,0], [0,-1,0],
            intrinsic=[1, 1, 1], diffuse=0.3, specular=0.8, specular_n=50),
        Rect([150, 0, 30], 160, 60, [0,1,0], [1,0,0],
            intrinsic=[1., 1., 1.], diffuse=0., specular=0.9, specular_n=1e5),
        Sphere([70, 00, 20], 20, 
            intrinsic=[0.8, 0.8, 1], diffuse=0.8, specular=0.3, specular_n=4),
        Sphere([50, -40, 20], 20, 
            intrinsic=[1, 0.8, 0.0], diffuse=0.6, specular=0.9, specular_n=1e2)
],
    lights = [Light([30, 20, 100], [1, 1, 1])]
)

camera = Camera([0,0,50], [100,0,10])
image = camera.render(scene, width=200, height=100, ppi=2, super_sample=2)
print(f"Traced {scene.total_rays} rays.")
plt.imsave("demo.png", image.numpy())
