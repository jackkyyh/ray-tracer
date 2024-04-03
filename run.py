import matplotlib.pyplot as plt

from raytracer import Camera, Light, Rect, Sphere, Scene

scene = Scene(
    background = [0, 0, 0.],
    ambient = [0.7, 0.7, 0.7],
    objects = [
        Rect([80, 80, 30], 140, 60, [1,0,0], [0,-1,0],
                ambient=[0.3, 0.3, 0.9], diffuse=0.5, specular=0, specular_n=0),
        Rect([80, 0, 0], 160, 140, [0,1,0], [0,0,1],
            ambient=[0.1, 0.1, 0.1], diffuse=0.3, specular=0.9, specular_n=50),
        Sphere([70, 00, 20], 20, 
            ambient=[0.3, 0.3, 0.7], diffuse=0.5, specular=0.3, specular_n=4),
        Sphere([50, -40, 20], 20, 
            ambient=[0.5, 0.5, 0], diffuse=0.5, specular=0.3, specular_n=4)
],
    lights = [Light([30, 20, 100], [1, 1, 1])]
)

camera = Camera([0,0,50], [100,0,10])
image = camera.render(scene, width=200, height=100, ppi=2, super_sample=2)
print(f"Traced {scene.total_rays} rays.")
plt.imsave("demo.png", image.numpy())
