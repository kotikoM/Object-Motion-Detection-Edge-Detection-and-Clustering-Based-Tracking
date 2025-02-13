import cv2
import numpy as np

from src.dbscan import dbscan
from src.edge_detection import detect_edges


def extract_edge_points(edge_image, threshold = 50):
    """Extract points from the edge-detected image."""
    points = np.argwhere(edge_image > threshold)
    return points.astype(np.float32)


def apply_dbscan(points, eps = 10, min_samples = 10):
    """Apply DBSCAN clustering on extracted edge points."""
    labels = dbscan(points, eps, min_samples)
    print(f'Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.')
    return labels


def apply_edge_detection(image, type):
    """Apply edge detection on input image with concrete type."""
    # This part takes most time when doing it manually without optimization.
    return detect_edges(image, type)


def process_image(image_path, kernel='sobel', threshold=30, eps=30, min_samples=10):
    # Load the original image
    original_image = cv2.imread(image_path)
    original_resized = cv2.resize(original_image, (640, 480))

    # Detect edges using a chosen kernel
    edges = detect_edges(original_resized, kernel)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Extract points from the edge-detected image
    edge_points = extract_edge_points(edges, threshold)
    print(f'Clustering {edge_points.shape[0]} points.')

    # Create a black frame to visualize edge points as a binary image
    filtered_frame = np.zeros_like(edges_colored)

    # Fill in the edge points as white pixels on the black frame
    for point in edge_points:
        filtered_frame[int(point[0]), int(point[1])] = [255, 255, 255]  # Set the pixel to white

    # Apply DBSCAN clustering
    labels = apply_dbscan(edge_points, eps, min_samples)

    # Create a black frame for clustering visualization
    black_frame = np.zeros_like(original_resized)
    cluster_frame = draw_clusters_on_black_frame(black_frame, edge_points, labels)

    # Display the images separately
    cv2.imshow('Edges', edges_colored)
    cv2.imshow('Filtered Edges', filtered_frame)
    cv2.imshow('Cluster Plot', cluster_frame)

    # Wait until any key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_clusters_on_black_frame(frame, edge_points, labels):
    """Draw edge points and centroids of clusters on a black frame with different colors for each cluster."""
    # Define a colormap or a set of colors
    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]

    unique_labels = np.unique(labels)  # Get unique cluster labels
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels) if
                 label != -1}  # Map colors to labels

    # Draw edge points
    for i, point in enumerate(edge_points):
        label = labels[i]
        if label != -1:  # Only draw points belonging to a cluster
            color = color_map[label]  # Get color for the current label
            cv2.circle(frame, (int(point[1]), int(point[0])), 2, color, -1)  # Draw edge point as a small circle

    return frame


def decorate(frame, centroids, speeds):
    colors = [
        (0, 0, 255),  # Red
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (255, 255, 0),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
    ]

    # Map each label to a color
    sorted_labels = sorted(centroids.keys())
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(sorted_labels)}

    # Annotate the frame
    for label, centroid in centroids.items():
        x, y = centroid
        speed = speeds.get(label, 0)
        speed_text = f"{speed:.2f} px/f"

        # Draw speed text slightly above the centroid
        text_x = int(y) - 35
        text_y = int(x) - 25
        cv2.putText(frame, speed_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw a uniquely-colored circle around the centroid
        center = (int(y), int(x))
        radius = 20  # Radius of the circle
        color = color_map[label]  # Get the unique color for this label
        thickness = 2  # Thickness of the circle
        cv2.circle(frame, center, radius, color, thickness)


def calculate_speed(prev_point, next_point, t):
    x_1, y_1 = prev_point
    x_2, y_2 = next_point

    # Calculate the Euclidean distance between the two points
    distance = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
    return distance / t


def calculate_cluster_speeds(prev_centroids, centroids, t):
    # Dictionary to store cluster speeds
    speeds = {}

    if prev_centroids is not None:
        for label, centroid in centroids.items():
            if label in prev_centroids:  # Check if the cluster exists in the previous frame
                speed = calculate_speed(prev_centroids[label], centroid, t)
                speeds[label] = speed  # Store the calculated speed

    return speeds


def relabel(prev_centroids, new_points):
    # Helper function
    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    new_centroids = {label: np.mean(points, axis=0) for label, points in new_points.items()}
    # Handle case where no previous centroids exist
    if not prev_centroids:
        return new_centroids

    # Compute distances between all pairs of new and previous centroids
    distances = []
    for new_label, new_centroid in new_centroids.items():
        for prev_label, prev_centroid in prev_centroids.items():
            dist = distance(new_centroid, prev_centroid)
            distances.append((dist, new_label, prev_label))

    # Sort pairs by distance
    distances.sort()

    # Map closest new centroids to previous centroids
    result = {}
    used_new_labels = set()
    used_prev_labels = set()

    for _, new_label, prev_label in distances:
        if new_label not in used_new_labels and prev_label not in used_prev_labels:
            result[prev_label] = new_centroids[new_label]
            used_new_labels.add(new_label)
            used_prev_labels.add(prev_label)

    # Include unmatched new centroids with their original labels
    for new_label, new_centroid in new_centroids.items():
        if new_label not in used_new_labels:
            result[new_label] = new_centroid

    return result


def process_video(video_path, kernel='sobel', threshold=200, eps=10, min_samples=10):
    # Initialize video capture and parameters
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    prev_centroids = None
    t = 1  # Time interval (in frames)

    # Video parameters
    frame_width, frame_height = 640, 480
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out_original = cv2.VideoWriter('output_original.avi', fourcc, fps, (frame_width, frame_height))
    # out_clustered = cv2.VideoWriter('output_clustered.avi', fourcc, fps, (frame_width, frame_height))

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        # Apply edge detection and extract edge points
        edges = apply_edge_detection(frame_resized, kernel)
        edge_points = extract_edge_points(edges, threshold)
        print(f'Frame {frame_idx}: Clustering {edge_points.shape[0]} points.')

        # Apply DBSCAN clustering
        labels = apply_dbscan(edge_points, eps, min_samples)

        # Compute cluster centroids and speeds
        unique_labels = set(labels) - {-1}
        initial_labeling = {label: edge_points[labels == label] for label in unique_labels}
        centroids = relabel(prev_centroids, initial_labeling)
        speeds = calculate_cluster_speeds(prev_centroids, centroids, t)
        prev_centroids = centroids

        # Decorate the frame with cluster data
        decorate(frame_resized, centroids, speeds)

        # Create the clustered view on a black background
        cluster_view = draw_clusters_on_black_frame(np.zeros_like(frame_resized), edge_points, labels)

        # Display the video frames
        # cv2.imshow('Original Video', frame_resized)
        # cv2.imshow('Clustered Points', cluster_view)

        # Write frames to output files
        out_original.write(frame_resized)
        out_clustered.write(cluster_view)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Release resources
    cap.release()
    out_original.release()
    out_clustered.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    """IMAGE PROCESSING"""
    # cars = '../data/images/cars.jpg'
    # kernel = 'sobel'  # Options 'sobel', 'prewitt' and 'laplacian'
    # threshold = 240
    # eps = 15
    # min_samples = 10
    # process_image(cars, kernel, threshold, eps, min_samples)

    """VIDEO PROCESSING"""
    balls = '../data/videos/two_balls_video.avi'
    juggling_balls = '../data/videos/juggling_balls.mp4'
    traffic = '../data/videos/traffic_jam.mp4'
    vid = juggling_balls

    kernel = 'sobel'  # Options 'sobel', 'prewitt' and 'laplacian'
    threshold = 200
    eps = 10
    min_samples = 10
    process_video(vid, kernel, threshold, eps, min_samples)
