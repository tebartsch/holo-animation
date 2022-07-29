# holo-animation

Small script for visualizing complex valued functions. I.e. for `f(z) = z**2`:

![](https://github.com/tilmann-bartsch/holo-animation/blob/master/examples/square_function.gif)

The video above is created by
```python
from holo_animation import create_video

create_video(lambda z: z**2, './square-function.mp4',
             grid_center=0 + 0j,
             grid_width=1,
             grid_height=1,
             n_grid=20,
             fps=50,
             seconds=5)
```
