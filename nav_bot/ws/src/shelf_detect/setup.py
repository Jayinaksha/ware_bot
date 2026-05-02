from setuptools import find_packages, setup

package_name = 'shelf_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anany',
    maintainer_email='anany@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	    'visualise = shelf_detect.b3rb_ros_draw_map:main',
        'visual_rack_node = shelf_detect.visual_rack_node:main',
        'simple_filter = shelf_detect.simple_filter:main',
        'qr_detector_node = shelf_detect.qr_detector_node:main',
        'hardware_interface = shelf_detect.hardware_interface:main',
        'odom_publisher = shelf_detect.odom_publisher:main',
        'map_autosaver = shelf_detect.map_autosave:main',
        ],
    },
)
