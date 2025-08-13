import datetime
import matplotlib.pyplot as plt

MONITORS = list()


def time_ms():
    now = datetime.datetime.now()
    milliseconds_since_midnight = (
        now.hour * 3600000 +
        now.minute * 60000 +
        now.second * 1000 +
        now.microsecond // 1000
    )
    return milliseconds_since_midnight


# pylint: disable=global-statement
def add_monitor(tag):
    """Set top Tk objet"""
    global MONITORS
    MONITORS.append((time_ms(),tag))
    
def show_monitor(taglist:list=None):
    """Set top Tk objet"""
    global MONITORS

    for tag in taglist:
        x_axis=[]
        y_axis=[]
        count =0
        for (time,tag2) in MONITORS:
            if tag == tag2:
                count+=1
                x_axis.append(time)
                y_axis.append(count)
        plt.plot(x_axis,y_axis, label=tag)

    plt.title(f"Monitoring {taglist}")
    plt.legend()
    plt.xlabel("ms")
    plt.ylabel("occurences")
    plt.show()