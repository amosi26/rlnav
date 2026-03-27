from setuptools import setup

package_name = 'ros2_policy_runner'

from glob import glob
import os

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Minimal ROS2 wrapper for PPO PolicyRunner',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_runner_node = ros2_policy_runner.policy_node:main',
            'gz_base_driver_node = ros2_policy_runner.gz_base_driver:main',
        ],
    },
    data_files=[
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
)
