import os

base_dir = os.path.dirname(__file__)
mesh_dir = os.path.join(base_dir, "robot_model", "allegro_hand", "meshes", "visual")

print("Visual mesh files:")
for f in os.listdir(mesh_dir):
    print("  ", f)

mesh_dir = os.path.join(base_dir, "robot_model", "allegro_hand", "meshes", "collision")

print("\nCollision mesh files:")
for f in os.listdir(mesh_dir):
    print("  ", f)
