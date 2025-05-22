# DispLayOut
"Display out is cool, but more importantly, it displays with layout." - A Python visualizaiton lib with extremely easy-to-use ***layout control***.

**What I Expect:**
```Python
import displayout
from displayout import Display
from display.layout_utils import align_with_it

# quick visualization with simple layout control.
img_0, img_1 = torch.rand(1, c, h, w, dtype=torch.float32, device='cuda')
img_2, img_3 = np.random.rand(h', w', c, dtype=np.uint8)
img_4 = PIL.Image.open('./hello_world.png')

Display([
    [img_0, img_1, None],
    [align_with_it(img_2), ...],
    [None, img_3, img_4],
    [Margin('black', width=diplayout.Wdith)],
])


img_batch = torch.rand(6, c, h, w, dtype=torch.float32, device='cuda')
img_list = [img_0, img_1, img_2]


ax, fig = Display(img_batch, return_ax=True)
img_arr = Display(*img_list, return_arr=True)
```
