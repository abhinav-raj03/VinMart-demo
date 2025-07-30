# tracker/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.4,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
        )

    def update_tracks(self, detections, frame):
        # Format: [[x, y, w, h, conf, class_id], ...]
        formatted_detections = [
            {
                "bbox": det[:4],
                "confidence": det[4],
                "class": det[5]
            } for det in detections
        ]

        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)

        # Add class_id to track object
        for track, det in zip(tracks, formatted_detections):
            track.class_id = det["class"]

        return tracks
