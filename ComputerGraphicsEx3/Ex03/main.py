from hw3 import *

sphere_a = Sphere([-0.5, 0.2, -1],0.5)
sphere_a.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)

cuboid = Cuboid(
    [-1, -.75, -2],
    [-1,-2, -2],
    [ 1,-2, -1.5],
    [ 1, -.75, -1.5],
    [ 2,-2, -2.5],
    [ 2, -.75, -2.5]
    )


cuboid.set_material([1, 0, 0], [1, 0, 0], [0, 0, 0], 100, 0.5)
cuboid.apply_materials_to_faces()

sphere_b = Sphere([0.8, 0, -0.5],0.3)
sphere_b.set_material([0, 1, 0], [0, 1, 0], [0.3, 0.3, 0.3], 100, 0.2)
plane = Plane([0,1,0], [0,-2,0])
plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)

background = Plane([0,0,1], [0,0,-10])
background.set_material([0, 0.5, 0], [0, 1, 0], [1, 1, 1], 100, 0.5)

objects = [cuboid,plane,background]

light = PointLight(intensity= np.array([1, 1, 1]),position=np.array([2,2,1]),kc=0.1,kl=0.1,kq=0.1)

lights = [light]

ambient = np.array([0.1,0.2,0.3])

camera = np.array([0,0,1])

im = render_scene(camera, ambient, lights, objects, (64,64), 3)
plt.imshow(im)
plt.imsave('scene3.png', im)