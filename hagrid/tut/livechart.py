


import numpy as np
import time
import matplotlib
#matplotlib.use('GTKAgg')
from matplotlib import pyplot as plt


import matplotlib.animation as animation
from matplotlib import style
from glob import glob
import pickle


#style.use('fivethirtyeight')


class Animator(object):

    def __init__(self, sDir):
        print("init")
        plt.ion()
        self.sDir = sDir
        self.oFig = plt.figure()
        self.ax1 = self.oFig.add_subplot(1,1,1)
        self.lsEp = sorted(glob(sDir+"/*"))
        self.iEp = 0

    def nextFile(self):
        print("nextfile")
        if self.iEp < len(self.lsEp):
            self.iEp += 1
            return self.lsEp[self.iEp-1]
            

    def animate(self):
        print("animate")
        sfEp = self.nextFile()
        with open(sfEp, "rb") as fIn:
            oSeg = pickle.load(fIn)
        
        self.ax1.clear()
        self.ax1.plot(oSeg.gVpred, label="vpred")
        self.ax1.plot(oSeg.adv, label="adv")
        self.ax1.plot(oSeg.tdlamret, label="tdlamret")
        self.ax1.plot(oSeg.gRew, label="rew")
        self.ax1.scatter(np.arange(len(oSeg.gbDone)), np.zeros(len(oSeg.gbDone)), label="done", linewidth=0, marker="o", s=oSeg.gbDone*20)
        #plt.plot(df.cum2, label="done")
        self.ax1.legend()
        plt.draw()
        plt.pause(0.5)
        #plt.xlim(0,100)
        #self.ax1.title(sfEp)

    def __call__(self):
        print("call - animate")
        self.animate()


def animate(i):
    #graph_data = open('example.txt','r').read()
    
    ax1.clear()
    ax1.plot(xs, ys)


def main():
    #fig = plt.figure()
    
    #ax1 = fig.add_subplot(1,1,1)

    #ani = animation.FuncAnimation(fig, animate, interval=1000)

    oAnim = Animator("../model-data/epdata")
    #animation.FuncAnimation(oAnim.oFig, oAnim, interval=0.05)

    while True:
        oAnim()

    plt.show()


def randomwalk(dims=(256, 256), n=20, sigma=5, alpha=0.95, seed=1):
    """ A simple random walk with memory """

    r, c = dims
    gen = np.random.seed(seed)
    pos = gen.rand(2, n) * ((r,), (c,))
    old_delta = gen.randn(2, n) * sigma

    while True:
        delta = (1. - alpha) * gen.randn(2, n) * sigma + alpha * old_delta
        pos += delta
        for ii in xrange(n):
            if not (0. <= pos[0, ii] < r):
                pos[0, ii] = abs(pos[0, ii] % r)
            if not (0. <= pos[1, ii] < c):
                pos[1, ii] = abs(pos[1, ii] % c)
        old_delta = delta
        yield pos


def run(niter=1000, doblit=True):
    """
    Display the simulation using matplotlib, optionally using blit for speed
    """

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.hold(True)
    rw = randomwalk()
    x, y = rw.next()

    plt.show(False)
    plt.draw()

    if doblit:
        # cache the background
        background = fig.canvas.copy_from_bbox(ax.bbox)

    points = ax.plot(x, y, 'o')[0]
    tic = time.time()

    for ii in range(niter):

        # update the xy data
        x, y = iter(rw).next()
        points.set_data(x, y)

        if doblit:
            # restore background
            fig.canvas.restore_region(background)

            # redraw just the points
            ax.draw_artist(points)

            # fill in the axes rectangle
            fig.canvas.blit(ax.bbox)

        else:
            # redraw everything
            fig.canvas.draw()

    plt.close(fig)
    print("Blit = %s, average FPS: %.2f" % (
        str(doblit), niter / (time.time() - tic)))

#if __name__ == '__main__':
#    run(doblit=False)
#    run(doblit=True)




def data_gen(t=0):
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

#fig, ax = plt.subplots()
#line, = ax.plot([], [], lw=2)
#ax.grid()
#xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

#ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
#                              repeat=False, init_func=init)
#plt.show()


if __name__ == "__main__":
    main()
