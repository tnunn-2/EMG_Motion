Prosthetic Hand (Anatomical wrist ordering) - Package
----------------------------------------------------

Contents
- prosthetic_hand.urdf      -- URDF file (uses primitive geometry)
- load_prosthetic.py       -- Example PyBullet loader and demo script

Description
- 3 DOF wrist in anatomical order: wrist_flex (pitch), wrist_dev (roll), wrist_rotate (yaw/pronation)
- 1 actuated gripper DOF implemented as a prismatic actuator controlling two symmetric fingers (fixed links)
- Geometry uses cylinders/boxes/spheres so no external mesh files are necessary

How to use
1. Copy the folder to your project.
2. Install pybullet (pip install pybullet).
3. Run: python load_prosthetic.py
4. In your control loop, command joint indices:
   0 : wrist_flex (revolute)     -- target in radians (approx -1.57..1.57)
   1 : wrist_dev  (revolute)     -- target in radians (approx -1..1)
   2 : wrist_rotate (revolute)   -- target in radians (approx -3.14..3.14)
   4 : finger_joint (prismatic)  -- target in meters (0.0 open .. 0.03 closed)

Mapping suggestion for NinaPro gestures:
- rest/open : finger_joint -> 0.0
- close/power grip : finger_joint -> 0.03
- pinch/tripod : finger_joint -> 0.02
- wrist flex/ext : set wrist_flex (joint 0)
- radial/ulnar : set wrist_dev (joint 1)
- pronation/supination : set wrist_rotate (joint 2)

