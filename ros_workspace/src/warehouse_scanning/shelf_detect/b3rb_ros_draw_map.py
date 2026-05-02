#!/usr/bin/env python3
# Map visualizer + shelf post-processing + odom arrow + mean long-side angle
import math
import rclpy
from rclpy.node import Node

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

import numpy as np
from scipy.ndimage import label

from nav_msgs.msg import OccupancyGrid, Odometry
from tf_transformations import euler_from_quaternion

from custom_definitions.msg import Rack, RackArray



class MapVisualizer(Node):
    def __init__(self):
        super().__init__('map_visualizer')

        self.subscription_map = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        self.subscription_odom = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10)
        
        self.publisher_racks = self.create_publisher(
            RackArray,
            '/detected_racks',
            10
        )

        # latest odom pose (x,y,yaw) in meters
        self.robot_pose = None
        self.simple_map = None

        plt.ion()
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.robot_pose = (px, py, yaw)

    def map_callback(self, msg: OccupancyGrid):
        plt.clf()
        self.simple_map = msg
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution

        # occupancy grid as 2D array (rows=h, cols=w)
        grid = np.array(msg.data).reshape((h, w))

        # Build RGB image for display (consistent with your previous mapping)
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[grid == 0] = [0, 0, 0]         # free -> black
        image[grid == 100] = [0, 255, 0]     # occupied -> green
        image[grid == -1] = [127, 127, 127]  # unknown -> gray

        plt.imshow(image)
        plt.gca().invert_yaxis()

        # -------------------------------------------------------------------
        # STEP 1: Extract only green occupied pixels (pillars)
        # -------------------------------------------------------------------
        occ = (grid == 100).astype(np.uint8)
        labeled, num = label(occ, (np.ones((3, 3), dtype=int)))

        pillar_clusters = []
        for cluster_id in range(1, num + 1):
            ys, xs = np.where(labeled == cluster_id)
            if len(xs) == 0:
                continue
            # filter out large clusters (walls)
            if len(xs) > 10:
                continue
            cx = np.mean(xs)
            cy = np.mean(ys)
            pillar_clusters.append([cx, cy])

        pillar_clusters = np.array(pillar_clusters)  # pixel coords (x, y)

        if len(pillar_clusters) >= 1:
            # draw pillar points
            plt.scatter(pillar_clusters[:, 0], pillar_clusters[:, 1], s=6, c='lime')

        # Not enough pillars -> just draw and return
        if len(pillar_clusters) < 4:
            self._draw_robot_arrow(res, w, h)
            plt.pause(0.01)
            return

        # -------------------------------------------------------------------
        # STEP 2: Exhaustive shelf candidate search (no 'used' set)
        # -------------------------------------------------------------------
        shelves = []
        # shelf dims in meters
        L_min, L_max = 0.9, 1.4  # long side in meters
        W_min, W_max = 0.3, 0.7  # short side in meters

        N = len(pillar_clusters)
        for i in range(N):
            for j in range(i + 1, N):
                # distance in meters between i and j
                dij_px = np.linalg.norm(pillar_clusters[i] - pillar_clusters[j])
                d_ij = dij_px * res
                if not (L_min <= d_ij <= L_max):
                    continue

                for k in range(N):
                    if k == i or k == j:
                        continue
                    dik_px = np.linalg.norm(pillar_clusters[i] - pillar_clusters[k])
                    d_ik = dik_px * res
                    if not (W_min <= d_ik <= W_max):
                        continue

                    # expected 4th point in pixel coordinates
                    fourth_px = pillar_clusters[j] + (pillar_clusters[k] - pillar_clusters[i])

                    # match to nearest pillar
                    best_m = -1
                    min_dist = 1e9
                    for m in range(N):
                        if m == i or m == j or m == k:
                            continue
                        dist = np.linalg.norm(pillar_clusters[m] - fourth_px)
                        if dist < min_dist:
                            min_dist = dist
                            best_m = m

                    # require closeness threshold in meters -> convert to pixels
                    if best_m != -1 and (min_dist * res) < 0.15:
                        shelves.append([i, j, k, best_m])

        # STEP 3: Filter/annotate candidates and draw rectangles; compute angles
        candidates = []
        long_side_angles_rad = []   # collect angles (radians) of long side p_i->p_j (in meters)
        centres_list = []           # reset per callback

        for s in shelves:
            pts_px = pillar_clusters[s]  # 4x2 pixel coordinates
            p_i = pts_px[0]
            p_j = pts_px[1]
            p_k = pts_px[2]
            p_m = pts_px[3]

            # compute angle between vectors i->j and i->k (in pixels)
            vec_ij = p_j - p_i
            vec_ik = p_k - p_i
            norm_ij = np.linalg.norm(vec_ij)
            norm_ik = np.linalg.norm(vec_ik)
            if norm_ij == 0 or norm_ik == 0:
                continue
            cos_theta = np.dot(vec_ij, vec_ik) / (norm_ij * norm_ik)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = np.arccos(cos_theta)
            angle_deg = np.degrees(angle_rad)

            box_color = 'red' if abs(angle_deg - 90) <= 40 else 'blue'

            # compute bounding (in pixel units)
            pts_all = np.array([p_i, p_j, p_k, p_m])
            min_x_px, max_x_px = np.min(pts_all[:, 0]), np.max(pts_all[:, 0])
            min_y_px, max_y_px = np.min(pts_all[:, 1]), np.max(pts_all[:, 1])
            area_m2 = ((max_x_px - min_x_px) * res) * ((max_y_px - min_y_px) * res)

            candidates.append({
                's': s,
                'pts_px': pts_all,
                'min_x_px': min_x_px,
                'max_x_px': max_x_px,
                'min_y_px': min_y_px,
                'max_y_px': max_y_px,
                'area': area_m2,
                'color': box_color
            })

            # compute long-side angle (p_i -> p_j). Convert to meters first
            long_vec_m = (p_j - p_i) * res  # (dx_m, dy_m)
            ang = math.atan2(long_vec_m[1], long_vec_m[0])  # radians
            long_side_angles_rad.append(ang)

        # Draw bounding rectangles for all candidates and mark centers
        for c in candidates:
            pts = c['pts_px']
            p_i = pts[0]; p_j = pts[1]; p_k = pts[2]; p_m = pts[3]
            rect_x = [p_i[0], p_j[0], p_m[0], p_k[0], p_i[0]]
            rect_y = [p_i[1], p_j[1], p_m[1], p_k[1], p_i[1]]
            plt.plot(rect_x, rect_y, color=c['color'], linewidth=2)

            centre_x = (p_i[0] + p_j[0] + p_k[0] + p_m[0]) / 4.0
            centre_y = (p_i[1] + p_j[1] + p_k[1] + p_m[1]) / 4.0
            plt.scatter(centre_x, centre_y, c='yellow', s=30)

        # compute long-side angles (rack orientation)
            #self.get_logger().info(f"Detected {len(candidates)} shelf candidates.")
        

            v1 = (p_j - p_i) * res    # long side 1
            v2 = (p_m - p_k) * res    # long side 2

            ang1 = math.atan2(v1[1], v1[0])
            ang2 = math.atan2(v2[1], v2[0])

            # circular mean of the two long-side angles
            sin_sum = math.sin(ang1) + math.sin(ang2)
            cos_sum = math.cos(ang1) + math.cos(ang2)

            mean_angle_rad = math.atan2(sin_sum, cos_sum)
            mean_angle_rad = (mean_angle_rad + math.pi) % (2 * math.pi) - math.pi
            mean_angle_deg = math.degrees(mean_angle_rad)
            
            centres_list.append((centre_x, centre_y,mean_angle_deg))


            # show angle
            cx = (p_i[0] + p_j[0] + p_k[0] + p_m[0]) / 4.0
            cy = (p_i[1] + p_j[1] + p_k[1] + p_m[1]) / 4.0
            
            

            plt.text(cx, cy - 10,
                    f"{mean_angle_deg:.1f}°",
                    color='red',
                    fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5))

            # store inside candidate
            c['alignment_angle_deg'] = mean_angle_deg
            
            
            unique_candidates = []
            min_dist_m = 0.05   # 5cm

            for cx, cy, theta in centres_list:     # you must store (cx, cy, pts) in centres_list
                # check if already represented
                is_duplicate = False
                for ux, uy,ang in unique_candidates:
                    dist = math.hypot((cx - ux) * res, (cy - uy) * res)
                    if dist < min_dist_m:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_candidates.append((cx, cy,theta))
            #self.get_logger().info(f"Unique shelf candidates after filtering: {len(unique_candidates)}")
            if len(unique_candidates)==5 :
                Arr= RackArray()
                for c in unique_candidates:
                    rack= Rack()
                    rack.x= c[0]
                    rack.y= c[1]
                    rack.theta_deg= c[2]
                    Arr.racks.append(rack)
                    
                self.publisher_racks.publish(Arr)
                #self.get_logger().info(f"Published RackArray with {len(Arr.racks)} racks.")
                


        # draw robot arrow (using latest odom)
        self._draw_robot_arrow(res, w, h)

        plt.title("Shelf Detection (Red=Rectangular, Blue=Skewed)")
        plt.pause(0.01)

        # you can log it if you need
        

    def _draw_robot_arrow(self, res, w, h):
        if self.robot_pose is None:
            return
        rx, ry, yaw = self.robot_pose
        rx_px, ry_px = self.get_map_pose_from_world_coords(rx, ry)
        # small arrow length in meters -> pixels
        arrow_len_m = 0.1
        dx_px = (arrow_len_m / res) * math.cos(yaw)
        dy_px = (arrow_len_m / res) * math.sin(yaw)
        # plot arrow (quiver works well)
        plt.quiver([rx_px], [ry_px], [dx_px], [dy_px],
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)
        plt.scatter(rx_px, ry_px, c='blue', s=20)

    def get_map_pose_from_world_coords(self, world_x, world_y):
        msg = self.simple_map
        if msg is None:
            return (0, 0)
        # Convert world coordinates (meters) to map pixel coordinates
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        res = msg.info.resolution

        map_x = int((world_x - origin_x) / res)
        map_y = int((world_y - origin_y) / res)

        return (map_x, map_y)


def main():
    rclpy.init()
    node = MapVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
