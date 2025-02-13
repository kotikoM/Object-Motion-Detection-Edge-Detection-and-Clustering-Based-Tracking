# Object Motion Detection

A numerical project attempting to detect and measure moving objects in a video (as long as the background is mercifully simple). It relies on clustering and edge detection to do the heavy lifting.

## Features (or Lack Thereof)
- **Outline Generation**: Can draw outlines around moving objects... given enough time.
- **Velocity Calculation**: Estimates how fast objects are moving in pixels per frame.
- **Handcrafted Implementations**: Most of the key functions are written by hand, meaning minimal optimization and maximum pain.

## Technology Stack
- **Python**: Because why not?
- **NumPy**: Handles matrix operations so you don't have to.
- **OpenCV**: Because writing an image processing library from scratch would not be good life choice.

## Installation
If you really want to run this, make sure you have the following installed:
```sh
pip install numpy opencv-python
```

## How to Use
1. Feed it a video with a simple background and adjust the parameters.
2. Let it process (will need some time).
3. Inspect the output. If it's wrong, tweak settings and repeat step 1.

## More Details (If You're That Curious)
A more detailed breakdown is available here:
[Project Documentation](https://komachavariani.notion.site/Object-Motion-Detection-12e17eee0cbb802fa89efab05c881b3e?pvs=4)

## Results
Behold the fruit of this laborious endeavor:

[![Watch the demo](https://img.youtube.com/vi/XoqtPuh6WnE/0.jpg)](https://www.youtube.com/watch?v=XoqtPuh6WnE)

Special thanks to Cacao Motion for the perfect video:
[![Original Source](https://img.youtube.com/vi/urRQuGRkzcs/0.jpg)](https://www.youtube.com/watch?v=urRQuGRkzcs)

## Disclaimer
This project is best enjoyed with a strong cup of coffee, ample free time, and a willingness to embrace imperfection.

To My Other Favorite R.B. ...
