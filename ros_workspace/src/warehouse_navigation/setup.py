from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anany',
    maintainer_email='anany@todo.todo',
    description='Navigation node for shelf alignment and Nav2 control',
    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            'navigation = navigation.navigation:main',
            'record_path = navigation.record_path:main',
            'mission_planner = navigation.mission_planner_service:main',
            
        ],
    },
)

