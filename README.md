# Sleep loss diminishes hippocampal reactivation and replay
This repository conatains all the code used to analyze data and generate figures for Giri, Kinsky et al. (2024).  

The main code used is found in the `analysis_and_figure code` folder. The `misc_code` contains miscellaneous code, some of which is used for data visualization and quality control.

### Interactive sleep scoring plots
Navigate to the `interactive_figures/sleep_scoring` folder and download a sleep scoring file (.html format) for a session of your choice.  

In this file you will find an interactive plot of sleep scoring for that session. An example sleep deprivation session is shown below:  

![Example SD session](sleep_scoring_example.png) 

### Interactive replay plot
Please download the .html file from this link
https://www.dropbox.com/s/e6nh5ucjsw0upgh/trajectory_replay_events.html?dl=0

In this file you will find an interactive plot as shown in the screenshot below:

![Example Image](replay_events_screenshot.png)

Each panel depicts all PBE events from NSD (top) and SD (bottom) sessions. Each panel shows events that were categorized as continuous (brown dots) and non-continuous (blue dots). Hovering your mouse pointer on top a data point will show posterior probabililty matrix depicting probabilities of each location (along the yaxis) across time (along the xaxis).
