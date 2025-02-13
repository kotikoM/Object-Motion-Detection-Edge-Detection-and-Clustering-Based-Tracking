import cv2
import numpy as np
import random

# Video settings
width, height = 640, 480  # Frame size
fps = 30  # Frames per second
video_length = 5  # Length of the video in seconds
total_frames = fps * video_length

# Ball settings
ball_radius = 20
straight_ball_color = (255, 0, 0)  # Blue
random_ball_color = (0, 255, 0)  # Green

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../data/videos/two_balls_video.avi', fourcc, fps, (width, height))

# Initialize ball positions
straight_ball_pos = [100, 100]  # Starting position for the straight-moving ball
random_ball_pos = [500, 400]  # Starting position for the random-moving ball

# Set initial velocity for straight-moving ball
straight_ball_velocity = [3, 2]  # Moves in a straight line with constant velocity


# Function to generate random velocity
def random_velocity():
    return [random.randint(-5, 5), random.randint(-5, 5)]


# Initialize random ball velocity
random_ball_velocity = random_velocity()

# Loop through each frame
for frame_idx in range(total_frames):
    # Create a black background
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Update straight ball position
    straight_ball_pos[0] += straight_ball_velocity[0]
    straight_ball_pos[1] += straight_ball_velocity[1]

    # Ensure the straight-moving ball stays within the bounds
    if straight_ball_pos[0] - ball_radius < 0 or straight_ball_pos[0] + ball_radius > width:
        straight_ball_velocity[0] *= -1
    if straight_ball_pos[1] - ball_radius < 0 or straight_ball_pos[1] + ball_radius > height:
        straight_ball_velocity[1] *= -1

    # Update random ball position
    random_ball_pos[0] += random_ball_velocity[0]
    random_ball_pos[1] += random_ball_velocity[1]

    # Ensure the random-moving ball stays within the bounds
    if random_ball_pos[0] - ball_radius < 0 or random_ball_pos[0] + ball_radius > width:
        random_ball_velocity[0] *= -1
    if random_ball_pos[1] - ball_radius < 0 or random_ball_pos[1] + ball_radius > height:
        random_ball_velocity[1] *= -1

    # Randomly change velocity for the random-moving ball every 20 frames
    if frame_idx % 20 == 0:
        random_ball_velocity = random_velocity()

    # Draw the straight-moving ball
    cv2.circle(frame, (straight_ball_pos[0], straight_ball_pos[1]), ball_radius, straight_ball_color, -1)

    # Draw the random-moving ball
    cv2.circle(frame, (random_ball_pos[0], random_ball_pos[1]), ball_radius, random_ball_color, -1)

    # Write frame to the video
    out.write(frame)

    # Display the frame (optional)
    # cv2.imshow('Moving Balls', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release everything
out.release()
cv2.destroyAllWindows()
