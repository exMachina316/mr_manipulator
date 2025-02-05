from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mr_manipulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_client = mr_manipulator.action_client:main',
            'image_proc = mr_manipulator.image_proc:main',
            'waypoint_transformer = mr_manipulator.waypoint_transformer:main',
            'hand_drawing = mr_manipulator.hand_drawing:main',
        ],
    },
)
