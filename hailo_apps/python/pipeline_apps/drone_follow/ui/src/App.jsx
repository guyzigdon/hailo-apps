import { useState, useEffect, useRef, useCallback } from "react";

const POLL_INTERVAL = 100; // ms

export default function App() {
  const [detections, setDetections] = useState([]);
  const [followingId, setFollowingId] = useState(null);
  const [videoDims, setVideoDims] = useState({ width: 0, height: 0 });
  const imgRef = useRef(null);

  // Poll detections
  useEffect(() => {
    let active = true;
    const poll = async () => {
      while (active) {
        try {
          const res = await fetch("/api/detections");
          if (res.ok) {
            const data = await res.json();
            setDetections(data.detections || []);
            setFollowingId(data.following_id);
          }
        } catch {
          // server not ready
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL));
      }
    };
    poll();
    return () => {
      active = false;
    };
  }, []);

  // Track image natural dimensions
  const onImgLoad = useCallback(() => {
    const img = imgRef.current;
    if (img) {
      setVideoDims({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }, []);

  const handleFollow = async (id) => {
    try {
      await fetch(`/api/follow/${id}`, { method: "POST" });
    } catch {
      // ignore
    }
  };

  const handleClear = async () => {
    try {
      await fetch("/api/follow/clear", { method: "POST" });
    } catch {
      // ignore
    }
  };

  const vw = videoDims.width;
  const vh = videoDims.height;

  return (
    <div className="app">
      <div className="status-bar">
        <span className="status-text">
          {followingId != null
            ? `Following: ID ${followingId}`
            : "Auto (largest person)"}
        </span>
        <button className="clear-btn" onClick={handleClear}>
          Clear Target
        </button>
      </div>

      <div className="video-container">
        <img
          ref={imgRef}
          className="video-feed"
          src="/api/video"
          alt="Live feed"
          onLoad={onImgLoad}
        />
        {vw > 0 && vh > 0 && (
          <svg className="overlay" viewBox={`0 0 ${vw} ${vh}`}>
            {detections.map((det) => {
              const x = det.bbox.x * vw;
              const y = det.bbox.y * vh;
              const w = det.bbox.w * vw;
              const h = det.bbox.h * vh;
              const isFollowing =
                det.id != null && det.id === followingId;
              const hasId = det.id != null;

              return (
                <g
                  key={det.id ?? `${det.bbox.x}-${det.bbox.y}`}
                  onClick={hasId ? () => handleFollow(det.id) : undefined}
                  style={{ cursor: hasId ? "pointer" : "default" }}
                >
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    fill="none"
                    stroke={isFollowing ? "#00ff00" : "#ffffff"}
                    strokeWidth={isFollowing ? 3 : 2}
                    strokeOpacity={0.9}
                  />
                  <text
                    x={x + 4}
                    y={y - 6}
                    fill={isFollowing ? "#00ff00" : "#ffffff"}
                    fontSize={14}
                    fontFamily="monospace"
                    fontWeight="bold"
                    style={{
                      textShadow: "1px 1px 2px rgba(0,0,0,0.8)",
                    }}
                  >
                    {hasId ? `ID: ${det.id}` : "person"}{" "}
                    ({Math.round(det.confidence * 100)}%)
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}
