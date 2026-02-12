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

## HTTP Control Server

The application includes an HTTP server (running on port 8080 by default) that provides status information and target selection capabilities.

### Default Behavior

By default (no target set), the drone follows the person with the **largest bounding box** in the frame.

### API Endpoints

- `GET /status` - Get current tracking status
  - Returns: `{"following_id": <id or null>, "last_seen": <timestamp or null>, "available_ids": [list of IDs]}`
  
- `POST /follow/<detection_id>` - Start following a specific tracked person (requires `--enable-tracking`)
  - Returns 200: `{"status": "success", "following_id": <id>}` if the ID is found in the current frame
  - Returns 404: `{"status": "error", "message": "...", "available_ids": [...]}` if the ID is not in the current frame
  
- `POST /follow/clear` - Clear target and return to following the largest person
  - Returns: `{"status": "success", "following_id": null, "message": "Cleared target, now following largest person"}`

### Basic Usage (Without Tracking)

Even without tracking enabled, the server is available and shows status:

```bash
# Check current status
curl http://localhost:8080/status
```

### Target Selection with Tracking

When running with `--enable-tracking`, you can select a specific person to follow by their tracking ID:

1. Run with tracking enabled:
   ```bash
   python drone_follow.py --input udp://0.0.0.0:5600 --target-bbox-height 0.5 --enable-tracking
   ```

2. Check which people are visible:
   ```bash
   curl http://localhost:8080/status
   # Returns: {"following_id": null, "last_seen": null, "available_ids": [1, 3, 5]}
   ```

3. Select a specific person to follow:
   ```bash
   # Follow the person with tracking ID 3
   curl -X POST http://localhost:8080/follow/3
   # Returns: {"status": "success", "following_id": 3}
   
   # If ID not found:
   # Returns: {"status": "error", "message": "Detection ID 42 not found in current frame", "available_ids": [1, 3, 5]}
   ```

4. Clear target and return to following the largest person:
   ```bash
   curl -X POST http://localhost:8080/follow/clear
   ```

### Configuration

- Change the server port with `--follow-server-port <port>` (default: 8080)
