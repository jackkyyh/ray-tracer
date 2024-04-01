import matplotlib.pyplot as plt

from raytracer import Camera, Light, Sphere, Scene, render


background = [0, 0, 0.]
ambient = [1, 1, 1]
camera = Camera([0,0,50], [100,0,10], ppi=3)
light = Light([30, 20, 100], [1, 1, 1])
blue_sphere2 = Sphere([100, 00, 0], 30, ambient=[0.3, 0.3, 0.7], diffuse=0.5)
yellow_sphere = Sphere([70, -40, 0], 30, ambient=[0.5, 0.5, 0], diffuse=0.5)

scene = Scene(background, ambient, [blue_sphere2, yellow_sphere], [light])

image = render(scene, camera)
plt.imsave("demo.png", image.numpy())