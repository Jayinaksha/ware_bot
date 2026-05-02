import rclpy
from rclpy.node import Node
from slam_toolbox.srv import SaveMap
import time

class MapAutoSaver(Node):
    def __init__(self):
        super().__init__('map_autosaver')
        
        # --- CONFIGURATION ---
        self.save_interval = 30.0  # Save every 30 seconds
        self.map_name = "/home/pi/r2d2_ws/src/car/maps/my_serial_map"
        
        # Create Client
        self.client = self.create_client(SaveMap, '/slam_toolbox/save_map')
        
        # Wait for service to be available (don't start timer until SLAM is ready)
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for SLAM Toolbox save service...')
            
        self.get_logger().info(f'Auto-Saver Running! Saving to {self.map_name} every {self.save_interval}s')
        
        # Create Timer
        self.timer = self.create_timer(self.save_interval, self.save_map)

    def save_map(self):
        req = SaveMap.Request()
        req.name.data = self.map_name
        
        future = self.client.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()
            # SLAM Toolbox returns 0 for success (usually)
            self.get_logger().info(f'Map Saved Successfully at {time.strftime("%H:%M:%S")}')
        except Exception as e:
            self.get_logger().error(f'Failed to save map: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MapAutoSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()