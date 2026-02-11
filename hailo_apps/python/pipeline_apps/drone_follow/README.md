# Drone Follow Demo

## Instructions

1. Run PX4 SITL with Gazebo and x500 drone with mono camera:
   ```bash
   make px4_sitl gz_x500_mono_cam
   ```

2. Run QGroundControl:
   ```bash
   ./QGroundControl-x86_64.AppImage
   ```

3. Run the video bridge:
   ```bash
   python video_bridge.py
   ```

4. Run the drone follow application:
   ```bash
   python drone_follow.py --input udp://0.0.0.0:5600 --target-bbox-height 0.5
   ```
