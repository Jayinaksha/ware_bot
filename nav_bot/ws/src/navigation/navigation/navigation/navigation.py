#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy # <--- CRITICAL IMPORTS

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
from action_msgs.msg import GoalStatus
from custom_definitions.msg import RackArray # Ensure you sourced setup.bash!
from std_srvs.srv import Trigger 

import math

SERVER_WAIT_TIMEOUT_SEC = 5.0
START_X, START_Y, START_YAW_DEG = 0.0, 0.0, 0.0
END_X, END_Y, END_YAW_DEG = START_X, START_Y, START_YAW_DEG + 180.0

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        
        self.logger = self.get_logger()
        self.logger.info("Initializing Navigation Node...")

        # --- QoS FOR MAP (CRITICAL FIX) ---
        # SLAM Toolbox publishes the map as "Transient Local".
        # We must match this, or we will never receive the map data.
        map_qos_policy = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1
        )

        self.map_subscription = self.create_subscription(
            OccupancyGrid, 
            '/map', 
            self.map_callback, 
            map_qos_policy) # <--- Apply QoS here

        # --- SUBSCRIBERS ---
        self.odom_subscription = self.create_subscription(
            Odometry, 
            '/odometry/filtered', 
            self.odom_callback, 
            10)
            
        self.rack_subscription = self.create_subscription(
            RackArray, 
            '/detected_racks', 
            self.rack_callback, 
            10)
        
        # --- CLIENTS & SERVICES ---
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.action_client_through_poses = ActionClient(self, NavigateThroughPoses, '/navigate_through_poses')
        self.srv = self.create_service(Trigger, 'start_mission', self.start_mission_callback)
        self.timer = self.create_timer(1.0, self.navigation_loop)
        self.scan_client = self.create_client(Trigger, '/perform_rack_scan')
        # --- STATE ---
        self.simple_map_curr = None
        self.rack_positions_in_world_coord = []
        self.capture_positions_in_world_coord_ordered = []
        self.entrance_waypoints = []
        self.exit_waypoints = []
        
        self.target_rack_index = 0
        self.nav_state = 'IDLE' 
        self.entering_indx = 0
        self.mission_started = False 
        self.goal_completed = True
        self.goal_handle_curr = None
        self.is_scanning = False

    def start_mission_callback(self, request, response):
        # Allow starting with fewer racks for testing (e.g., > 0 instead of == 5)
        num_racks = len(self.rack_positions_in_world_coord)
        
        if num_racks < 1:
            response.success = False
            response.message = f'FAILED: Found {num_racks} racks. Drive around more!'
            self.logger.warn(response.message)
            return response
        
        if self.simple_map_curr is None:
            response.success = False
            response.message = 'FAILED: No Map received yet! Cannot navigate.'
            self.logger.warn(response.message)
            return response
            
        self.mission_started = True
        self.nav_state = 'slam_completed'
        self.process_racks_for_navigation()
        
        response.success = True
        response.message = f'Mission Started! Navigating to {num_racks} racks.'
        self.logger.info(response.message)
        return response

    def map_callback(self, msg: OccupancyGrid):
        if self.simple_map_curr is None:
            self.logger.info("MAP RECEIVED! Ready to convert coordinates.")
        self.simple_map_curr = msg
        
    def odom_callback(self, msg: Odometry):
        pass
        
    def rack_callback(self, msg: RackArray):
        if self.mission_started: return

        # Only update if we have new data
        if len(msg.racks) > 0:
            if self.simple_map_curr is None:
                # We can't convert coords yet, but we see the racks.
                # Just log it so we know connection is good.
                self.logger.info(f"Seeing {len(msg.racks)} racks (Waiting for Map...)", throttle_duration_sec=2.0)
                return

            temp_racks = []
            for rack in msg.racks:
                 wx, wy = self.get_world_coord_from_map_coord(rack.x, rack.y)
                 temp_racks.append((wx, wy, rack.theta_deg))
            
            self.rack_positions_in_world_coord = temp_racks
            self.logger.info(f"Tracking {len(temp_racks)} racks in World Coords", throttle_duration_sec=2.0)

    def process_racks_for_navigation(self):
        # Sort and calculate approach points
        capture_points = []
        for (wx, wy, theta) in self.rack_positions_in_world_coord:
            px, py, angle = self.find_capture_position_near_rack(wx, wy, theta)
            capture_points.append((px, py, angle))
            
        # Sort by distance from (0,0)
        self.capture_positions_in_world_coord_ordered = sorted(
            capture_points, 
            key=lambda p: math.hypot(p[0], p[1])
        )

    def navigation_loop(self):
        if not self.mission_started or not self.goal_completed: return
        if self.is_scanning: return
        # --- STATE MACHINE ---
        if self.nav_state == 'visiting_racks':
            
            # Check if we are done with all racks
            if self.target_rack_index >= len(self.capture_positions_in_world_coord_ordered):
                self.nav_state = 'exiting_warehouse'
                return
            # CHECK: Did we just arrive at a rack?
            # We know we arrived because goal_completed is True. 
            # We check a flag 'ready_to_scan' to see if we haven't scanned this one yet.
            if self.ready_to_scan:
                self.logger.info("Arrived at Rack. Requesting Scan...")
                # CALL THE SERVICE
                if self.scan_client.wait_for_service(timeout_sec=1.0):
                    self.is_scanning = True # Stop the loop while scanning
                    
                    req = Trigger.Request()
                    future = self.scan_client.call_async(req)
                    future.add_done_callback(self.scan_complete_callback)
                else:
                    self.logger.error("Hardware Node not ready! Skipping scan.")
                    self.ready_to_scan = False
                    self.target_rack_index += 1
                return
            # IF NOT SCANNING, MOVE TO NEXT RACK
            target = self.capture_positions_in_world_coord_ordered[self.target_rack_index]
            goal = self.create_goal_pose(target[0], target[1], target[2])
            
            if(self.send_goal_from_world_pose(goal)): 
                self.ready_to_scan = True # Set flag so we scan upon arrival
            return
        
        if self.nav_state == 'slam_completed':
            # Go to Alignment Point (Near Rack 5/Start)
            rack5 = self.capture_positions_in_world_coord_ordered[-1] 
            goal = self.create_goal_pose(rack5[0] + 0.5, END_Y, rack5[2] + 90.0)
            self.logger.info(f"Moving to Alignment: {goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}")
            if(self.send_goal_from_world_pose(goal)): self.nav_state = 'reaching_origin'
            return
        
        if self.nav_state == 'reaching_origin':
            goal = self.create_goal_pose(START_X, START_Y, START_YAW_DEG)
            self.logger.info("Moving to Origin...")
            if(self.send_goal_from_world_pose(goal)): self.nav_state = 'entering_warehouse'
            return
            
        if self.nav_state == 'entering_warehouse':
            self.find_entering_waypoints()
            if self.entering_indx < len(self.entrance_waypoints):
                wp = self.entrance_waypoints[self.entering_indx]
                goal = self.create_goal_pose(wp[0], wp[1], wp[2])
                self.logger.info(f"Entering Waypoint {self.entering_indx}")
                if(self.send_goal_from_world_pose(goal)): self.entering_indx += 1
            else:
                 self.nav_state = 'navigating_to_racks123'
            return
        
        if self.nav_state == 'navigating_to_racks123':
            if self.target_rack_index >= 3 or self.target_rack_index >= len(self.capture_positions_in_world_coord_ordered):
                self.nav_state = 'intermediate_point_bw_123_45'
                return
            target = self.capture_positions_in_world_coord_ordered[self.target_rack_index]
            goal = self.create_goal_pose(target[0], target[1], target[2])
            self.logger.info(f"Visiting Rack {self.target_rack_index}")
            if(self.send_goal_from_world_pose(goal)): self.target_rack_index += 1
            return

        if self.nav_state == 'intermediate_point_bw_123_45':
            # Safety check if we have enough racks
            if len(self.capture_positions_in_world_coord_ordered) > 3:
                r3 = self.capture_positions_in_world_coord_ordered[2]
                r4 = self.capture_positions_in_world_coord_ordered[3]
                goal = self.create_goal_pose(r3[0], r4[1], r3[2]) # Simple corner logic
                self.logger.info("Moving to intermediate point...")
                if(self.send_goal_from_world_pose(goal)): self.nav_state = 'navigating_to_racks45'
            else:
                self.nav_state = 'exiting_warehouse'
            return

        if self.nav_state == 'navigating_to_racks45':
            if self.target_rack_index >= len(self.capture_positions_in_world_coord_ordered):
                self.nav_state = 'exiting_warehouse'
                return
            target = self.capture_positions_in_world_coord_ordered[self.target_rack_index]
            goal = self.create_goal_pose(target[0], target[1], target[2])
            self.logger.info(f"Visiting Rack {self.target_rack_index}")
            if(self.send_goal_from_world_pose(goal)): self.target_rack_index += 1
            return

        if self.nav_state == 'exiting_warehouse':
            self.logger.info("Mission Complete. Stopping.")
            self.mission_started = False
            return

    # --- UTILS ---
    def get_world_coord_from_map_coord(self, map_x, map_y):
        if not self.simple_map_curr: return (0.0, 0.0)
        res = self.simple_map_curr.info.resolution
        origin_x = self.simple_map_curr.info.origin.position.x
        origin_y = self.simple_map_curr.info.origin.position.y
        # Convert Pixel to Meters
        world_x = (map_x * res) + origin_x
        world_y = (map_y * res) + origin_y
        return (world_x, world_y)

    def find_capture_position_near_rack(self, rx, ry, r_theta, offset=0.85):
        rad = math.radians(r_theta)
        # Approach from "front" (Normal vector)
        capture_angle = rad - (math.pi / 2.0)
        cx = rx + (offset * math.cos(capture_angle))
        cy = ry + (offset * math.sin(capture_angle))
        # Face the rack (Opposite to Normal)
        face_angle = r_theta - 90.0
        return (cx, cy, face_angle)

    def create_goal_pose(self, x, y, yaw_deg):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x = x
        p.pose.position.y = y
        cy = math.cos(math.radians(yaw_deg) * 0.5)
        sy = math.sin(math.radians(yaw_deg) * 0.5)
        p.pose.orientation = Quaternion(z=sy, w=cy)
        return p

    # --- ACTION CLIENT BOILERPLATE (Standard) ---
    def send_goal_from_world_pose(self, goal_pose):
        if not self.action_client.wait_for_server(timeout_sec=1.0):
            self.logger.warn("Nav2 Action Server not ready")
            return False
        
        goal = NavigateToPose.Goal()
        goal.pose = goal_pose
        self.goal_completed = False
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self.logger.error("Goal Rejected")
            self.goal_completed = True
            return
        self.goal_handle_curr = handle
        handle.get_result_async().add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        self.goal_completed = True
        self.logger.info("Goal Succeeded")

    def find_entering_waypoints(self):
        # (Same logic as before, just ensuring list isn't empty)
        if self.capture_positions_in_world_coord_ordered:
            r1 = self.capture_positions_in_world_coord_ordered[0]
            self.entrance_waypoints = [
                (r1[0] + 1.3, START_Y, START_YAW_DEG),
                (r1[0], START_Y - 1.0, r1[2])
            ]
            
    def scan_complete_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.logger.info("Scan Finished! Moving to next rack.")
            else:
                self.logger.warn("Scan failed or timed out.")
        except Exception as e:
            self.logger.error(f"Service call failed: {e}")
        
        # Reset flags to resume navigation
        self.is_scanning = False
        self.ready_to_scan = False
        self.target_rack_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()