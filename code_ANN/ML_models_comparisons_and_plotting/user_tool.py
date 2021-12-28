from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
import NNmodels_userTool
import pandas as pd


# initial values
init_diameter_val = 10
init_depth_val = 10
init_power = 1
init_pulseDuration = 1
init_drillingTime = 0.3
init_repetitionRate = 600

data_dir = "data/"
file_name = 'dataWithDepthAndDiameter_TRAININGplusVALIDATON_new_1.xls'
input_size = 4

# read the excel file with pandas
df = pd.read_excel(data_dir + file_name)

# convert it to numpy
data = df.to_numpy()

# max and min of dataset
x_max = data[:, :input_size].max(0)
x_min = data[:, :input_size].min(0)

y_max = data[:, input_size:].max(0)
y_min = data[:, input_size:].min(0)
print("y_max", y_max)
print("x_mmax", x_max)


def drawTriangle(diameter_val, depth_val):

    left_corner = [0, depth_val]
    righ_corner = [diameter_val, depth_val]
    bottom_corner = [diameter_val/2, 0]
    triangle = np.c_[left_corner, righ_corner, bottom_corner, left_corner]

    return triangle


triangle = drawTriangle(init_diameter_val, init_depth_val)

fig, ax = plt.subplots()
line, = plt.plot(triangle[0], triangle[1], color='#1f77b4', lw=2)
ax.set_xlim([0, 160])
ax.set_ylim([0, 80])

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

#  slider to control the power
power_max = 30
axPower = plt.axes([0.25, 0.16, 0.65, 0.03])
power_slider = Slider(
    ax=axPower,
    label='Power [%]',
    valmin=0,
    valmax=100,
    valinit=init_power,
)

#  slider to control the pulse duration
axPulseDuration = plt.axes([0.25, 0.12, 0.65, 0.03])
pulseDuration_slider = Slider(
    ax=axPulseDuration,
    label='Pulse duration [ns]',
    valmin=x_min[2],
    valmax=x_max[2],
    valinit=init_pulseDuration,
)

#  slider to control the pulse repeptition rate
axPulseRepetitionRate = plt.axes([0.25, 0.08, 0.65, 0.03])
pulseRepetitionRate_slider = Slider(
    ax=axPulseRepetitionRate,
    label='Pulse repetition rate [kHz]',
    valmin=30,
    valmax=x_max[1],
    valinit=init_repetitionRate,
)

#  slider to control the drilling time
axDrillingTime = plt.axes([0.25, 0.04, 0.65, 0.03])
drillingTime_slider = Slider(
    ax=axDrillingTime,
    label='Drilling time [ms]',
    valmin=x_min[3],
    valmax=x_max[3],
    valinit=init_drillingTime,
)


model_name = "383_new_full"
laser_source_manufacturer = "IPG Laser GmbH"
laser_source_model = "YLP-1-150V-30"


# The function to be called anytime a slider's value changed

textvar_diameter = plt.text(0.02, 0.65, "Diameter: " + str(init_diameter_val), fontsize=10,
                            transform=plt.gcf().transFigure)

textvar_depth = plt.text(0.02, 0.6, "Depth: " + str(init_depth_val), fontsize=10,
                         transform=plt.gcf().transFigure)

textvar_laserSource = plt.text(0.02, 0.75, "Laser Source Manufacturer: " + laser_source_manufacturer, fontsize=10,
                               transform=plt.gcf().transFigure)
textvar_laserSource = plt.text(0.02, 0.7, "Laser Source Model: " + laser_source_model, fontsize=10,
                               transform=plt.gcf().transFigure)


def update(val):

    input_params = [(float(power_slider.val)/100)*power_max, float(pulseRepetitionRate_slider.val), float(
        pulseDuration_slider.val), float(drillingTime_slider.val)]
    predictions = NNmodels_userTool.predict(
        model_name, input_params)
    diameter_val = predictions['output'][0]
    depth_val = predictions['output'][1]
    triangle = drawTriangle(diameter_val, depth_val)

    line.set_ydata(triangle[1])
    line.set_xdata(triangle[0])
    fig.canvas.draw_idle()
    textvar_diameter.set_text(
        "Diameter: " + str("{:.2f}".format(diameter_val)))
    textvar_depth.set_text("Depth: " + str("{:.2f}".format(depth_val)))


power_slider.on_changed(update)
pulseDuration_slider.on_changed(update)
pulseRepetitionRate_slider.on_changed(update)
drillingTime_slider.on_changed(update)


plt.show()
