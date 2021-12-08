import ipywidgets.widgets as widgets
from jetracer.nvidia_racecar import NvidiaRacecar
import traitlets

controller = widgets.Controller(index=0)  
display(controller)

Controller()
car = NvidiaRacecar()
car.throttle_gain = 1
car.steering_offset=-0.2
car.steering = 0

left_link = traitlets.dlink((controller.axes[0], 'value'), (car, 'steering'), transform=lambda x: -x)
right_link = traitlets.dlink((controller.axes[1], 'value'), (car, 'throttle'), transform=lambda x: x)
