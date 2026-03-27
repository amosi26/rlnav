# Hexa Description

Paste your URDF into `hexa_description/urdf/robot.urdf` and put all mesh STL files into `hexa_description/meshes/`.

Note: The URDF in this repo has been rewritten to use:

```
package://hexa_description/meshes/...
```

So the meshes must be in this package for Gazebo to load them.
