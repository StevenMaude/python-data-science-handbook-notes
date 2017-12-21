# Chapter 4: Matplotlib

## Visualisation with Matplotlib

Built on NumPy arrays, and designed to work with the SciPy stack.

Works with many operating systems and graphics backends, supporting many
output types: a big strength of Matplotlib, and led to a large user and
developer base, as well as it being prominent in scientific Python use.

More recently, the interface and style of Matplotlib has begun to show
their age. Newer tools like ggplot and ggvis in R, and web visualisation
toolkits based on D3 and HTML5 canvas, make Matplotlib feel clunky and
old-fashioned. But, Matplotlib does still have strengths in being
well-tested and cross-platform.

Recent versions of Matplotlib have made it relatively easy to set global
plotting styles, and people have developed new packages that build on it
to drive Matplotlib via cleaner APIs (e.g. Seaborn, ggpy, HoloViews,
Altair and pandas itself). Even with wrappers like these, it is useful
to dive into Matplotlib's syntax to adjust the final plot output. It may
be that Matplotlib remains a vital tool for data visualisation, even if
new tools mean users move away from using Matplotlib's API directly.

### General Matplotlib

#### Importing Matplotlib

Often use shorthand for importing Matplotlib, as for NumPy and pandas:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
```

The `plt` interface will be used most often here.

#### Setting styles

Use `plt.style` directive to choose an aesthetic style for figures:

```python
plt.style.use('classic')
```

Here we set the `classic` style, which ensures that plots use the
classic Matplotlib style.

#### How to display plots

Viewing Matplotlib plots depends on context. The best use of Matplotlib
differs depending on how you use it.

##### Plotting from a script

Here, `plt.show()` is what you want. `plt.show()` starts an event loop,
looks for all currently active figure objects and opens one or more
interactive windows that display your figure or figures.

For example, you may have `myplot.py`:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()
```

You can run this script:

```sh
$ python myplot.py
```

which will result in a window opening with your figure displayed.

The `plt.show()` command does a lot: interacting with your system's
interactive graphical backend. How this works can vary from system to
system, and even from installation to installation.

`plt.show()` should only be used once per Python session. Usually, it is
used at the end of the script. Multiple `show()` commands can have
unpredictable, backend-dependent behaviour, and should typically be
avoided.

##### Plotting from an IPython shell

It can be convenient to use Matplotlib interactively with an IPython
shell. IPython can work well with Matplotlib if you use Matplotlib mode,
by using the `%matplotlib` magic command:

```python
In [1]: %matplotlib
Using matplotlib backend as TkAgg

In [2]: import matplotlib.pyplot as plt
```

From then on, any `plt` plot command will cause a figure window to open,
and further commands can be run to update the plot. Some changes (such
as modifying properties of lines that are already drawn) will not draw
automatically: use `plt.draw()` to force an update. Using `plt.show()`
in Matplotlib mode is not required.

##### Plotting from an IPython notebook

The IPython notebook is a browser-based interactive data analysis tool
that combines text, code, graphics, HTML elements and more into a single
executable document.

Plotting interactively within an IPython notebook can be done with the
`%matplotlib` command and works in a similar way to the IPython shell.
It is also possible to embed graphics directly in the notebook:

* `%matplotlib notebook` leads to *interactive* plots embedded in the
  notebook.
* `%matplotlib inline` leads to *static* images of plots embedded in the
  notebook.

After running `%matplotlib inline` in a notebook (once per
kernel/session), any cell within the notebok that creates a plot will
embed a PNG image of the resulting graphic.

For example:

```python
import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--'); # NB: using a semi-colon suppresses the command output.
```

##### Saving figures to file

Matplotlib can save figures in a wide variety of formats using the
`savefig()` command. For example, to save the previous figure as a PNG
file:

```python
fig.savefig('my_figure.png')
```

We now have the file in the current working directory shown by:

```
!ls -lh my_figure.png
```

To confirm it contains the figure, we can use the IPython `Image` object
to display the file's contents:

```python
from IPython.display import Image
Image('my_figure.png')
```

In `savefig()`, the file format is inferred from the filename extension.
Depending on the installed backends, many different file formats are
available. The list of supported file types can be found by:

```python
fig.canvas.get_supported_filetypes()
```

which shows output like:

```python
{'eps': 'Encapsulated Postscript',
 'jpeg': 'Joint Photographic Experts Group',
 'jpg': 'Joint Photographic Experts Group',
 'pdf': 'Portable Document Format',
 'pgf': 'PGF code for LaTeX',
 'png': 'Portable Network Graphics',
 'ps': 'Postscript',
 'raw': 'Raw RGBA bitmap',
 'rgba': 'Raw RGBA bitmap',
 'svg': 'Scalable Vector Graphics',
 'svgz': 'Scalable Vector Graphics',
 'tif': 'Tagged Image File Format',
 'tiff': 'Tagged Image File Format'}
```

#### Dual interfaces

A potentially confusing feature of Matplotlib is its dual interfaces: a
convenient MATLAB-style state-based interface, and a more powerful
object-oriented interface.

##### MATLAB-style interface

Originally Matplotlib was written as a Python alternative for MATLAB users,
and much of its syntax reflects that. The MATLAB-style tools are contained in
the pyplot (`plt`) interface. For example, the following code may look familiar
to MATLAB users:

```python
plt.figure()  # create a plot figure

# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));
```

This is a *stateful* interface. It keeps track of the "current" figure and
axes, which are where all `plt` commands are applied. You can get a reference
to these using the `plt.gcf()` (get current figure) and `plt.gca()` (get
current axes) routines.

While this stateful interface is fast and convenient for simple plots, it is
easy to run into problems. For example, once the second panel is created, how
do we go back and add something to the first? This is possible, but clunky.
Fortunately, there is a better way.

##### Object-oriented interface

The object-oriented interface is available for these more complicated
situations, and for when you want more control over your figure. Rather than
depending on some notion of an "active" figure or axes, in the object-oriented
interface the plotting functions are *methods* of explicit `Figure` and `Axes`
objects. To recreate the previous plot using this style of code, you might do:

```python
# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
```

For simple plots, the choice of which style to use is largely a matter of
preference, but the object-oriented approach can become a necessity as plots
become more complicated. In this chapter, the most convenient interface will
be used as appropriate. In most cases, the difference is as small as switching
`plt.plot()` to `ax.plot()`, but there are a few gotchas that will be
highlighted as they are encountered.

## Simple line plots

A simple plot is the visualisation of a single function y=f(x).

As with following sections, we'll start by setting up the notebook for
plotting and importing the packages we will use:

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

For all Matplotlib plots, we start by creating a figure and axes; for
example, in their simplest form:

```python
fig = plt.figure()
ax = plt.axes()
```

A Matplotlib *figure* (an instance of the `plt.Figure` class) can be
thought of as a single container that contains all the objects
representing axes, graphics, text and labels. The *axes* (an instance of
the `plt.Axes` class) is what we see from the code above: a bounding box
with ticks and labels, which will eventually contain the plot elements
that constitute the visualisation. Below, we'll use the variable name
`fig` to refer to `Figure` instance, and `ax` to refer to an `Axes`
instance or group of `Axes` instances.

Once we have created an axes, we can use `ax.plot()` to plot data:

```python
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));
```

Alternatively, we can use the pylab interface and let the figure and
axes be created for us in the background:

```python
plt.plot(x, np.sin(x));
```

Calling `plot()` again allows us to create a single figure with multiple
lines:

```python
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x));
```

### Adjusting the plot: line colours and styles

`plt.plot()` takes arguments to specify line colour and style.

For colour adjustment, use the `color` keyword, which can be specified
in a multiple of ways:

```python
plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
```

If no colour is specified, Matplotlib will cycle through a set of
default colours for multiple lines.

The line style can be adjusted using the `linestyle` keyword:

```python
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted
```

It is possible to combine colour and style into a single non-keyword
argument to `plt.plot()`:

```python
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red
```

There are meny other keyword arguments that can be used to adjust the
appearance of the plot; see the docstring of `plt.plot()`.

### Adjusting the plot: axes limits

Matplotlib does a decent job of choosing default axes limits, but
sometimes more control is needed. The most basic way is to use
`plt.xlim()` and `plt.ylim()` methods.

```python
plt.plot(x, np.sin(x))

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);
```

If you want an axis displayed in reverse, you can reverse the order of
arguments:

```python
plt.plot(x, np.sin(x))

plt.xlim(10, 0)
plt.ylim(1.2, -1.2);
```

A useful related method is `plt.axis()` (NB: `axis` not `axes`). This
lets you set the x and y limits by passing a list: `[xmin, xmax, ymin,
ymax]`.

```python
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);
```

The `plt.axis()` method also has other features, for example, allowing
tightening the bounds around the current plot automatically:

```python
plt.plot(x, np.sin(x))
plt.axis('tight');
```

or higher-level specifications, such as ensuring an equal aspect ratio,
so one unit in x equals one unit in y:

```python
plt.plot(x, np.sin(x))
plt.axis('equal');
```

For more, see the `plt.axis()` docstring.

### Labelling plots

Titles and axis labels are the simplest labels of plots with methods to
quickly set them:

```python
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)");
```

The position, size and style of these labels can be adjusted using
optional arguments: see the documentation and docstrings.

When multiple lines are shown within a single axes, it can be useful to
create a plot legend to label the lines. This is done via
`plt.legend()`. However, it can be easier to specify the label of each
line using the `label` keyword of the `plt.plot()`:

```python
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend();
```

### Matplotlib gotchas

Most `plt` functions translate directly to `ax` methods (such as
`plt.plot()` →  `ax.plot()`, `plt.legend()` →  `ax.legend()` etc.), this
is not always the case. Functions to set limits, labels and titles are
slightly modified:

* `plt.xlabel()` →  `ax.set_xlabel()`
* `plt.ylabel()` →  `ax.set_ylabel()`
* `plt.xlim()` →  `ax.set_xlim()`
* `plt.ylim()` →  `ax.set_ylim()`
* `plt.title()` →  `ax.set_title()`

In the object-oriented interface to plotting, it is often more
convenient to use `ax.set()` to set these all at once:

```python
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot');
```

## Simple scatter plots

Another commonly used plot type is the simple scatter plot. Instead of
points being joined by line segments, here points are represented
individually using a dot, circle or other shape.

We start as we did for line plots:

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```

### Scatter plots with `plt.plot`

`plt.plot()` (and `ax.plot()`) can also produce line plots. It turns out
that this same function can produce scatter plots too:

```python
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black');
```

The third argument in the `plt.plot()` call is a character representing
the type of symbol used for the plotting. Just as you can specify `-`,
`--` to control the line style, the marker style has its own set of
short string codes. Most possibilities are intuitive; here are some
examples:

```python
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
plt.plot(rng.rand(5), rng.rand(5), marker,
         label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);
```

For more possibilities, these character codes can be used together with
line and colour codes to plot points along with a line connecting them:

```python
plt.plot(x, y, '-ok');
```

Additional keyword arguments to `plt.plot()` specify a wide range of
properties of the lines and markers:

```python
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);
```

### Scatter plots with `plt.scatter`

A second, more powerful method of creating scatter plots is the `plt.scatter()`
function, which can be used very similarly to the `plt.plot()` function.

```python
plt.scatter(x, y, marker='o');
```

The primary difference of `plt.scatter` from `plt.plot` is that it can be used
to create scatter plots where properties of each individual point (size, face
colour, edge colour etc.) can be individually controlled or mapped to data.

The following example uses this to create a random scatter plot with points of
many colours and sizes. The `alpha` keyword adjusts the transparency level to
better see overlapping results.

```python
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar();  # show color scale
```

The colour argument is automatically mapped to a colour scale (shown by the
`colorbar()` command, and the size argument is given in pixels. In this way,
the colour and size of points can be used to convey information in the
visualisation, to visualise multidimensional data.

This is used in this example:

```python
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);
```

Each sample is one of three types of flowers that has had the size of its
petals and sepals measured. The plot shows the sepal length and width on the
axes, the species of flower by colour, and the size of point relates to petal
width.

### Efficiency of `plot` versus `scatter`

Apart from the different features of `plt.plot` and `plt.scatter`, why might
you want to choose one over the other. While it doesn't matter as much for
small amounts of data, as datasets get larger than a few thousand points,
`plt.plot()` can be noticeably more efficient than `plt.scatter()`. This is
because since `plt.scatter()` can render a different colour and/or size for
each point, the renderer must do the extra work of constructing each point
individually. In `plt.plot()`, the points are essentially clones of each other,
so the work of determining the points' appearance is done once only for the set
of data, and so `plt.plot()` is preferable for large datasets.

## Visualising errors

For scientific measurements, accurate accounting for errors is nearly as
important, if not more important, than accurate reporting  of the number
itself.

For example, imagine that I am using some astrophysical observations to
estimate the Hubble Constant, the local measurement of the expansion rate of
the Universe. I know that the current literature suggests a value of around 71
(km/s)/Mpc, and I measure a value of 74 (km/s)/Mpc with my method. Are the
values consistent? The only correct answer, given this information, is this:
there is no way to know.

Suppose I augment this information with reported uncertainties: the current
literature suggests a value of around 71 ± 2.5 (km/s)/Mpc, and my method has
measured a value of 74 ± 5 (km/s)/Mpc. Now are the values consistent? That is
a question that can be quantitatively answered.

In visualization of data and results, showing these errors effectively can make
a plot convey much more complete information.

### Basic error bars

Basic error bars can be created with a single Matplotlib function call:

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k');
```

`fmt` is a format code controlling the appearance of lines and points, and has
the same syntax as the shorthand used in `plt.plot()`.

`errorbar` has options to further adjust the outputs. For example, it can be
helpful to make the error bars lighter than the points:

```python
plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);
```

It is also possible to specify horizontal error bars (`xerr`), one-sided error
bars, and more.

### Continuous errors

In some cases, showing error bars on continuous quantities is desired.
Matplotlib doesn't directly have a routine for this, but it is possible to
combine `plt.plot` and `plt.fill_between` for a useful result.

(Also note that Seaborn provides visualisation of continuous errors too.)

This is a simple Gaussian process regression, using scikit-learn. This fits a
very flexible non-parametric function to data with a continuous measure of the
uncertainty:

```python
from sklearn.gaussian_process import GaussianProcess

# define the model and draw some data
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# Compute the Gaussian process fit
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                     random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region
```

We now have `xfit`, `yfit` and `dyfit` which sample the continuous fit to the
data. We could pass these to `plt.errorbar` but we don't want to plot 1000
points with 1000 errorbars. Instead, we can use `plt.fill_between()` with a
light colour to visualise this continuous error.

```python
# Visualize the result
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')

plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                 color='gray', alpha=0.2)
plt.xlim(0, 10);
```

With `fill_between()`, we pass the x value, then the lower and upper y bounds,
and the area between these regions is filled.

The figure gives an insight into what the Gaussian process regression does.
Near measured data points, the model is strongly constrained, with small model
errors, while further away, the model is not strongly constrained and errors
increase.

## Density and contour plots

Sometimes it is useful to display 3D data in 2D using contours or colour-coded
regions. There are three Matplotlib functions that can help here: `plt.contour`
for contour plots, `plt.contourf` for filled contour plots and `plt.imshow` for
showing images.

Again, setting up as before:

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
```

### Visualising a 3D function

We'll start by demonstrating a contour plot using a function z=f(x,y), using
the following function:

```python
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```

A contour plot can be created with `plt.contour()`. It takes three arguments:
a grid of *x* values, a grid of *y* values and a grid of *z* values. The *x*
and *y* values represent positions on the plot, and the *z* values will be
represented by the contour levels. Perhaps the most straightforward way to
prepare such data is to use the `np.meshgrid` function, which builds 2D grids
from 1D arrays:

```python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
```

To create a line-only contour plot:

```python
plt.contour(X, Y, Z, colors='black');
```

When a single colour is used, negative values are represented by dashed lines,
and positive values by solid lines. Alternatively, the lines can be
colour-coded by specifying a colormap with the `cmap` argument:

```python
plt.contour(X, Y, Z, 20, cmap='RdGy');
```

Above, 20 is the number of equally spaced intervals within the data range.

`RdGy` is the *Red-Grey* colormap, which is a good choice for centred data.
Matplotlib has a range of colormaps available, which are found inside the
`plt.cm` module.

The spaces between the lines can look distracting, so we can change this to
a filled contour plot using `plt.contourf()` which uses largely the same syntax
as `plt.contour()`.

```python
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar(); # Create additional axis with labelled colour info for plot.
```

Here, the colorbar makes it clear that the black regions are "peaks", while the
red regions are "valleys".

One potential issue with this plot is that it is "splotchy": colour steps are
discrete rather than continuous. This could be remedied by setting the number
of contours to a high number, but results in an inefficient plot as Matplotlib
must render a new polygon for each step in the level. A better solution is to
use the `plt.imshow()` function, which interprets a 2D grid of data as an
image:

```python
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image');
```

There are gotchas with `imshow()`:

* it doesn't accept an *x* and *y* grid, so you must specify the extent
  [*xmin*, *xmax*, *ymin*, *ymax*] of the image on the plot.
* it follows the standard image array defintiion where the origin is in the
  upper left, not in the lower left as in most contour plots. This must be
  changed when showing gridded data.
* it will automatically adjust the axis aspect ratio to match the input data;
  this can be changed by setting, e.g. `plt.axis(aspect='image')` to make
  *x* and *y* units match.

Finally, it can be sometimes useful to combine contour plots and image plots.
Here a partially transparent background image is used with contours
overplotted, and labels on the contours themselves:

```python
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8) # Label contours.

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5) # alpha sets transparency.
plt.colorbar();
```

## Histograms, binnings and density

A simple histogram can be a first step in understanding a dataset.

It's possible to create a basic histogram in one line, once the normal imports
are done:

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

data = np.random.randn(1000)

plt.hist(data);
```

The `hist()` function has many options to tune both the calculation and the
display; here's an example of a more customised histogram:

```python
plt.hist(data, bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none');
```

`stepfilled` with `alpha` can be useful to compare histograms of different
distributions:

```python
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);
```

If you would like to compute the histogram (count the points in a given bin),
but not display it, use `np.histogram()`:

```python
counts, bin_edges = np.histogram(data, bins=5)
print(counts)
```

### 2D histograms and binnings

Just as we create histograms in 1D by dividing the number line into bins, we
can create histograms in 2D by dividing points among 2D bins.

Start by defining some data — an *x* and *y* array drawn from a multivariate
Gaussian distribution:

```python
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
```

#### `plt.hist2d`: 2D histogram

This is an easy way to create a 2D histogram:

```python
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
```

`plt.hist2d` has extra options to adjust the plot and binning, just as
`plt.hist`. And, just as there is `np.histogram`, there is `np.histogram2d`
which can be used as:

```python
counts, xedges, yedges = np.histogram2d(x, y, bins=30)
```

There is also `np.histogramdd()` for histogram binning in more than two
dimensions.

#### `plt.hexbin`: hexagonal binnings

The 2D histogram creates a tesselation of squares across the axes. Another
shape for such a tesselation is the regular hexagon. Matplotlib provides the
`plt.hexbin` function to represent a 2D dataset binned with a grid of hexagons.

```python
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
```

`plt.hexbin` also has a number of options, including the ability to specify
weights for each point, and to change the output in each bin to any NumPy
aggregate (mean of weights, standard deviation of weights etc.).

#### Kernel density estimation

Another method to evaluate densities in multiple dimensions is *kernel density
estimation* (KDE). KDE can be thought of as a way to "smear out" the points in
space and add up the result to obtain a smooth function.

`scipy.stats` contains a quick and simple KDE implementation:

```python
from scipy.stats import gaussian_kde

# fit an array of size [Ndim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")
```

KDE has a smoothing length that effectively turns the dial between detail and
smoothness. `gaussian_kde` uses a rule-of-thumb to attempt to find a nearly
optimal smoothing length for the input data.

Other KDE implementations are available, e.g.
`sklearn.neighbors.KernelDensity` and
`statsmodels.nonparametric.kernel_density.KDEMultivariate`. KDE visualisations
with Matplotlib can be verbose; using Seaborn (see below) can be more terse.

## Customising plot legends

Plot legends assign meaning to various plot elements.

The simplest legend can be created with `plt.legend()`, creating a
legend automatically for any labelled plot elements:

```python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np

x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend();
```

But there are many ways we might want to customise a legend. For
example, we can specify the location and turn off the frame:

```python
ax.legend(loc='upper left', frameon=False)
fig
```

We can use `ncol` to specify the number of columns in the legend:

```python
ax.legend(frameon=False, loc='lower center', ncol=2)
fig
```

We can use a rounded box (`fancybox`) or add a shadow, change the
transparency of the frame, or the text padding:

```python
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig
```

### Choosing elements for the legend

The legend includes all labelled elements by default. If we do not want
this, we can fine tune which elements and labels appear by using the
objects returned by plot commands. `plt.plot()` can create multiple
lines at once, and returns a list of created line instances. Passing
any of these to `plt.legend()` tell it which to identify, along with the
labels we want to specify:

```python
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)

# lines is a list of plt.Line2D instances
plt.legend(lines[:2], ['first', 'second']);
```

It can be clearer to instead apply labels to the plot elements you'd
like to show on the legend:

```python
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True);
```

By default, the legend ignores elements without a `label` set.

### Legend for size of points

Sometimes the legend defaults are insufficient for the given
visualisation. For example, if you use the size of points to mark
features of the data and want to create a legend to reflect this.

Here is such an example: size of points indicate populations of
California cities. The legend should specify the scale of the sizes of
the points, and this is achieved by plotting labelled data with no
entries:

```python
import pandas as pd
cities = pd.read_csv('data/california_cities.csv')

# Extract the data we're interested in
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size and color but no label
plt.scatter(lon, lat, label=None,
            c=np.log10(population), cmap='viridis',
            s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')

plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Here we create a legend:
# we'll plot empty lists with the desired size and label
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

plt.title('California Cities: Area and Population');
```

The legend always references some object that is on the plot, so if we want to
display a particular shape we need to plot it. The circles we want for the
legend are not on the plot, so we fake them by plotting empty lists.

By plotting empty lists, we create labelled plot objects that are picked up by
the legend, and now the legend tells us useful information.

(It creates a legend using one scatter point for each "plot", where the point
size equals the area, which is specified as 100, 300 and 500, and will be to
the same scale as the real plots, even though no points are actually plotted.)

### Multiple legends

Sometimes you would like to add multiple legends to the same axes. Matplotlib
does not make this easy. The standard `legend` interface only allows creating
a single legend for the plot. Using `plt.legend()` or `ax.legend()` repeatedly
just overrides a previous entry. We can work around this by creating a new
legend artist from scratch, and then using the lower level `ax.add_artist()`
method to manually add the second artist to the plot:

```python
fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color='black')
ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
          loc='upper right', frameon=False)

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
             loc='lower right', frameon=False)
ax.add_artist(leg);
```

## Customising colorbars

Plot legends identify discrete labels of discrete points. For continuous
labels based on the colour of points, lines or regions, a labelled
colorbar can be a great tool. In Matplotlib, a colorbar is a separate
axes that can provide a key for the meaning of colours in a plot.

```python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
```

To create a simple colorbar, use `plt.colorbar()`:

```python
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])

plt.imshow(I)
plt.colorbar();
```

### Adjusting colorbars

The colormap can be specified using the `cmap` argument to the plotting
function that is creating the visualisation:

```python
plt.imshow(I, cmap='gray');
```

The available colormaps are in the `plt.cm` namespace.

#### Choosing the colormap

There are three categories of colormap:

* *Sequential colormaps*: these are made up of one continuous sequence
  of colour (e.g. `binary` or `viridis`).
* *Divergent colormaps*: these usually contain two distinct colors, which
  shows positive and negative deviations from a mean (e.g. `RdBu` or
  `PuOr`).
* *Qualitative colormaps*: these mix colors with no particular sequence
  (e.g. `rainbow` or `jet`).
  
Qualitative colormaps can be a poor choice for quantitative data: for
instance, they do not usually display a uniform progression in
brightness as the scale increases. This in turn means that the eye can
be drawn to certain portions of the colour regions, potentially
emphasising unimportant parts of the dataset. Colormaps with an even
brightness variation across the range are better here, and translate
well to greyscale printing too. `cubehelix` is a better alternative
rainbow scheme for continuous data.

Colormaps such as `RdBu` lose the positive-negative information on
conversion to greyscale!

#### Colour limits and extensions

Matplotlib allows for a large range of colorbar customisation. The
colorbar itself is an instance of `plt.Axes`, so all of the axes and
tick formatting tricks above apply. The colorbar has flexibility too:
e.g. we can narrow the colour limits and indicate the out-of-bounds
values with a triangular arrow at te top and bottom by setting `extend`.
This might be useful if displaying an image that is subject to noise.

```python
# make noise in 1% of the image pixels
speckles = (np.random.random(I.shape) < 0.01) # Produces a Boolean mask.
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1);
```

In the first image generated here, the default colour limits respond to
the noisy pixels, largely washing out the pattern we are interested in.
In the second image, the colour limits are set manually and
out-of-bounds values are indicated by the arrows.

#### Discrete colorbars

Colormaps are by default continuous, but sometimes you'd like them to be
discrete. The easiest way is to use `plt.cm.get_cmap()` and pass the
name of a suitable colormap with the number of desired bins:

```python
plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1);
```

#### Other notes

Can use `ticks` and `label` in `plt.colorbar()` to customise the
colorbar.

## Multiple subplots

Sometimes it is helpful to compare different views of data side by side.
For this purpose, Matplotlib has *subplots*: groups of smaller axes that
can exist together within a single figure. These subplots might be
insets, grids of plots or more complicated layouts.

```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
```

### `plt.axes`: subplots by hand

The most basic method of creating an axes is `plt.axes()`. As we've seen
previously, by default this creates a standard axes object that fills
the entire figure. `plt.axes` also takes an optional argument that is a
list of four numbers in the figure coordinate system. These numbers
represent `[left, bottom, width, height]` in the figure coordinate
system, which ranges from 0 at the bottom left of the figure to 1 at the
top right of the figure.

For example, we might create an inset axes at the top-right corner of
another axes by setting the *x* and *y* position to 0.65 (starting at
65% of the width, and 65% of the height of the figure), and the *x* and
*y* extents to 0.2 (the size of the axes is 20% of the width and 20% of
the height of the figure):

```python
ax1 = plt.axes()  # standard axes
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
```

The equivalent of this command in the object-oriented interface is
`fig.add_axes()`. Using this to create two vertically stacked axes:

```python
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2))

x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));
```

We now have two axes (the top with no tick labels) that are just touching: the
bottom of the upper panel (position 0.5) matches the top of the lower panel
(position 0.1 + 0.4).

### `plt.subplot`: simple grids of subplots

Aligned columns or rows of subplots are often enough used that Matplotlib has
routines to create these easily. The lowest level of these is `plt.subplot()`
which creates a single subplot within a grid. It takes three integer arguments:
the number of rows, the number of columns, and the index of the plot to be
created in this scheme, which runs from upper left to bottom right.

```python
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
             fontsize=18, ha='center')
```

`plt.subplots_adjust()` can be used to adjust the spacing between plots.

`fig.add_subplot()` is the equivalent object-oriented command:

```python
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center')
```

`hspace` and `wspace` arguments of `subplots_adjust()` specify the spacing
along the height and width of the figure, in units of the subplot size (in
this case, the space is 40% of the subplot height and width).

### `plt.subplots`: the whole grid in one go

The approach described above can be tedious when creating a large grid of
subplots, especially if you'd like to hide the x- and y-axis labels on the
inner plots. For this purpose, `plt.subplots()` is the easier tool to use.
Instead of creating a single subplot, `plt.subplots()` creates a full grid of
subplots in a single line, returning them in a NumPy array. The arguments are
the number of rows and number of columns, along with optional keywords `sharex`
and `sharey` allowing you to specify the relationships between different axes.

This example creates a 2x3 grid of subplots where all axes in the same row
share their y-axis scale and all axes in the same column share their x-axis
scale:

```python
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
```

Using `sharex` and `sharey` removes the inner labels on the grid to make the
plot cleaner. The resulting grid of axes instances is returned within a NumPy
array, allowing for specification of the desired axes using array indexing:

```python
# axes are in a two-dimensional array, indexed by [row, col]
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                      fontsize=18, ha='center')
fig
```

### `plt.GridSpec`: more complicated arrangements

To go beyond a regular grid to subplots that span multiple rows and columns,
`plt.GridSpec` is the best tool. It does not create a plot itself: it is simply
a convenient interface recognised by `plt.subplot()`. For example, a gridspec
for a grid of two rows and three columns with some specified width and height
space looks like:

```python
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
```

We can specify subplot locations and extents using Python slicing syntax:

```python
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);
```

## Text and annotation

Axes labels and titles are the most basic types of annotations, but the options
go beyond this.

`plt.text`/`ax.text` allow placing of text at a particular, x/y value, e.g.

```python
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
```

`ax.text()` takes x position, y position, a string, and optional keywords
specifying colour, size, style, alignment and other properties of the text.

`ha` is horizontal alignment.

### Transforms and text position

We previously anchored text annotations to data locations. Sometimes it is
preferable to anchor the text to a position on the axes or figure, independent
of the data. In Matplotlib, this is done by modifying the *transform*.

Any graphics display framework needs some scheme for translating between
coordinate systems. For example, a data point at x,y of 1,1 needs to be
represented at a certain location on the figure, which in turn needs to be
represented by pixels on the screen. Mathematically, such coordinate
transformations are straightforward and Matplotlib has tools it uses internally
to perform them (in the `matplotlib.transforms` module).

These details are usually not important to typical users, but it is helpful
to be aware of when considering the placement of text on a figure.

There are three predefined transforms that can be useful in this situation:

* `ax.transData`: transform associated with data coordinates
* `ax.transAxes`: transform associated with the axes (in units of axes dimensions)
* `fig.transFigure`: transform associated with the figure (in units of figure dimensions)

Here is an example of drawing text at locations using these transforms:

```python
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

# transform=ax.transData is the default, but we'll specify it anyway
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure);
```

Note that by default, the text is aligned above and to the left of the
specified coordinates: here the "." at the beginning of each string will
approximately mark the given coordinate location.

The `transData` coordinates give the usual data coordinates associated with the
x- and y-axis labels. The `transAxes` coordinates give the location from the
bottom-left corner of the axes, as a fraction of the axes size. The
`transFigure` coordinates are similar, specifying the position from the
bottom-left of the figure as a fraction of the figure size.

Notice now that if we change the axes limits, it is only the `transData`
coordinates that will be affected, while the others remain stationary:

```python
ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)
fig
```

### Arrows and annotation

Along with tick marks and text, arrows are another useful annotation.

Drawing arrows in Matplotlib is tricky. There is a `plt.arrow()` function
available, but is not too useful: the arrows it creates are SVG objects that
are subject to the varying aspect ratio of your plots, and the result is not
what the user intended. Instead, `plt.annotate()` is perhaps better. This
function creates some text and an arrow, and the arrows can be flexibly
specified.

```python
%matplotlib inline

fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));
```

The arrow style is controlled through the `arrowprops` dictionary, with many
options.

## Customising ticks

Matplotlib's default tick locators and formatters are sufficient in many
situations, but can be adjusted if not.

Matplotlib plots have an object hierarchy. Matplotlib aims to have a Python
object representing everything that appears on the plot: for example, the
`figure` is the bounding box within which plot elements appear. Each object
can act as a container of sub-objects: e.g. each `figure` can contain one or
more `axes` objects, each of which in turn contain other objects representing
plot contents.

The tick marks are no exception. Each `axes` has attributes `xaxis` and `yaxis`
which in turn have attributes that contain all the properties of the lines,
ticks and labels that make up the axes.

### Major and minor ticks

Within each axis, there is the concept of a *major* tick mark and a *minor*
tick mark. As the names imply, major ticks are usually bigger or more
pronounced, while minor ticks are usually smaller. By default, Matplotlib
rarely makes use of minor ticks, but one place you can see them is within
logarithmic plots:

```python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np

ax = plt.axes(xscale='log', yscale='log')
ax.grid();
```

We see that each major tick shows a large tickmark and a label, while each
minor tick shows a smaller tickmark with no label.

These tick properties — locations and labels — can be customised by setting
the `formatter` and `locator` objects of each axis. Let's examine these
for the x axis of the just shown logarithmic plot:

```python
print(ax.xaxis.get_major_locator()) # LogLocator
print(ax.xaxis.get_minor_locator()) # LogLocator
print(ax.xaxis.get_major_formatter()) # LogFormatterMathtext
print(ax.xaxis.get_minor_formatter()) # NullFormatter
```

### Hiding ticks or labels

A common formatting operation is hiding ticks or labels. This can be done using
`plt.NullLocator()` or `plt.NullFormatter()`:

```python
ax = plt.axes()
ax.plot(np.random.rand(50))

ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_formatter(plt.NullFormatter())
```

We've removed the labels, but kept the ticks, from the x-axis, and removed the
ticks (and the labels too) from the y-axis. Having no ticks at all can be
useful: for example, when you want to show a grid of images.

### Reducing or increasing the number of ticks

A problem with the default settings is that smaller subplots can end up with
crowded labels. For example:

```python
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)
```

Particularly for the x ticks, the numbers almost overlap, making them
difficult to read. We can fix this with `plt.MaxNLocator()` which allows
specification of the maximum number of ticks to display. Given this maximum
number, Matplotlib will use internal logic to choose the particular tick
locations.

```python
# For every axis, set the x and y major locator
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(3))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))
fig
```

### Fancy tick formats

The default tick formatting works well as a default, but sometimes you may
want more. Consider:

```python
# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);
```

First, it's more natural for this data to space the ticks and grid lines in
multiples of π. We can do this by setting a `MultipleLocator`, which locates
ticks at a multiple of the number you provide:

```python
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
fig
```

But now the tick labels look silly: they are multiples of π, but the decimal
representation doesn't immediately convey this. We can change the tick
formatter to fix this. There's no built-in formatter to do this, so we'll
use `plt.FuncFormatter` which accepts a user-defined function giving control
over the tick outputs:

```python
def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)

ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
fig
```

By enclosing the string in dollar signs, this enables LaTeX support.

## Customising Matplotlib: configurations and stylesheets

### Plot customisation by hand

It is possible to customise plot settings to end up with something nicer than
the default on an individual basis.

This is a drab default histogram:

```python
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np

%matplotlib inline

x = np.random.randn(1000)
plt.hist(x);
```

that can be adjusted to make it more visually pleasing:

```python
# use a gray background
ax = plt.axes(axisbg='#E6E6E6')
ax.set_axisbelow(True)

# draw solid white grid lines
plt.grid(color='w', linestyle='solid')

# hide axis spines
for spine in ax.spines.values():
    spine.set_visible(False)
    
# hide top and right ticks
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()

# lighten ticks and labels
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
    
# control face and edge color of histogram
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');
```

But this took a lot of effort, and would be repetitive to do for each plot.
Fortunately, there is a way to adjust these defaults once only.

### Changing the defaults: `rcParams`

Each time Matplotlib loads, it defines a runtime configuration (rc) containing
the default styles for every plot element you create. This configuration can
be adjusted at any time using `plt.rc()`.

Here we will modify the rc parameters to make our default plot look similar to
the improved version.

First, we save a copy of the current `rcParams` dictionary to easily reset
these changes in the current session:

```python
IPython_default = plt.rcParams.copy()
```

Now we use `plt.rc()` to change some of these settings:

```python
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
```

With these settings defined, we can create a plot to see these settings
applied:

```python
plt.hist(x);
```

Simple plots also can look nice with these same rc parameters:

```python
for i in range(4):
    plt.plot(np.random.rand(10))
```

Settings can be saved in a *.matplotlibrc* file.

### Stylesheets

Stylesheets are another way to customise Matplotlib. These let you create and
package your own styles, as well as use some built-in defaults. They are
formatted similarly to *.matplotlibrc* files but must have a *.mplstyle*
extension.

Available styles can be listed:

```python
plt.style.available
```

The way to switch to a stylesheet is:

```python
plt.style.use('stylename')
```

but this will change the style for the rest of the session. Alternatively,
a style context manager is available to set the style temporarily:

```python
with plt.style.context('stylename'):
    make_a_plot()
```

Here is a function that makes a histogram and line plot to show the effects
of stylesheets:

```python
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')
```

First, reset the runtime configuration:

```python
# reset rcParams
plt.rcParams.update(IPython_default);
```

Now see how the plots look with the default styling:

```python
hist_and_lines()
```

and use the context manager to apply another style:

```python
with plt.style.context('ggplot'):
    hist_and_lines()
```

#### Seaborn style

Matplotlib also has stylesheets inspired by Seaborn. These styles are loaded
automatically when Seaborn is imported:

```python
import seaborn
hist_and_lines()
```

## 3D plotting in Matplotlib

Matplotlib originally was designed for 2D plotting, then 3D plotting
tools were added later. 3D plots are enabled by importing `mplot3d`:

```python
from mpl_toolkits import mplot3d
```

Once this is imported, a 3D axes can be created by passing the keyword
`projection='3d'` to any of the normal axes creation routines:

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
```

With this 3D axes enabled, we can plot a variety of 3D plot types. 3D
plotting benefits from viewing figures interactively in a notebook: use
`%matplotlib notebook` instead of `%matplotlib inline` to do this.

### 3D points and lines

The most basic 3D plot is a line or collection of scatter plots created
from sets of (x, y, z) triples. These can be created using `ax.plot3D()`
and `ax.scatter3D()`. The call signature for these is almost identical
to that of their 2D counterparts.

This example plots a trigonometric spiral, along with some points drawn
randomly near the line:

```python
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
```

By default, the scatter points have their transparency adjusted to give
a sense of depth on the page.

### 3D contour plots

`mplot3d` contains tools to create 3D relief plots. Like the 2D
`ax.contour` plots, `ax.contour3D` requires all the input data to be in
the form of 2D regular grids, with the Z data evaluated at each point.

This is a 3D contour diagram of a 3D sinusoidal function:

```python
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
```

Sometimes the default viewing angle is not optimal. We can use the
`view_init` method to set the elevation and azimuthal angles. Here, we
set the elevation to 60 degrees (60 degrees above the xy-plane) and an
azimuth of 35 degrees (rotated 35 degrees anticlockwise about the
z-axis).

```python
ax.view_init(60, 35)
fig
```

Of course, with an interactive plot, this view adjustment can be carried
out by the user.

### Wireframes and surface plots

Two other types of 3D plots that work on gridded data are wireframes and
surface plots. These take a grid of values and project it onto the
specified 3D surface, making the resulting 3D forms easy to visualise.

Here is an example:

```python
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe');
```

A surface plot is like a wireframe plot, but each face of the wireframe
is a filled polygon. Adding a colormap to the filled polygons can help
make the surface topology clearer:

```python
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
                ax.set_title('surface');
```

The grid of values for a surface plot needs to be 2D, but not
necessarily rectilinear. Here is an example of creating a partial polar
grid:

```python
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');
```

### Surface triangulations

For some applications, evenly sampled grids as required by the above
routines is restrictive. In these cases, triangulation-based plots can
be useful. What if, rather than an even draw from a Cartesian or a polar
grid, we instead have a set of random draws?

```python
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)
```

We could create a scatter plot to get an idea of the surface we're
sampling from:

```python
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
```

This leaves much to be desired. The function that will help us is
`ax.plot_trisurf` which creates a surface by first finding a set of
triangles formed between adjacent points:

```python
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
                cmap='viridis', edgecolor='none');
```

This is not as clean as when plotted with a grid, but allows for
interesting 3D plots.

## Geographic data with Basemap

Geographic data is often visualised. Matplotlib's main tool for this
type of visualisation is the Basemap toolkit, one of several Matplotlib
toolkits that lives under the `mpl_toolkits` namespace. Basemap can be
clunky to use and often even simple visualisations can take longer to
render than is desirable. More modern visualisations, such as Leaflet or
the Google Maps API, may be a better choice for more intensive map
visualisations. Basemap is, however, still a useful tool to be aware of.

It requires separate installation, e.g. with `pip` or `conda install basemap`.

We add the Basemap import:

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
```

Geographic plots are just a few lines away (the graphics also require
the `PIL` package in Python 2, or the `pillow` package in Python 3):

```python
plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5);
```

The globe shown is not an image: it is a Matplotlib axes that
understands spherical coordinates and allows overplotting data on the
map.

Here a different map projection is used, zoomed in to North America and
the location of Seattle plotted:

```python
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6, 
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

# Map (long, lat) to (x, y) for plotting
x, y = m(-122.3, 47.6)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Seattle', fontsize=12);
```

### Map projections

The first thing to decide when using maps is the projection to use. It is
impossible to project a spherical map onto a flat surface without some
distortion or breaking its continuity. There are lots of choices of projection.
Depending on the intended use of the map projection, there are certain map
features (e.g. direction, area, distance, shape) that are useful to maintain.

The Basemap package implements many projections, referenced by a short format
code.

This is a convenience function to draw the world map along with latitude and
longitude lines:

```python
from itertools import chain

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
```

#### Cylindrical projections

Cylindrical projections are the simplest map projections. Lines of constant
latitude and longitude are mapped to horizontal and vertical lines. This type
of mapping represents equatorial regions well, but results in extreme
distortion near the poles. The spacing of latitude lines varies between
different cylindrical projections, leading to different conservation properties
and different distortion near the poles.

The following code generates an example of the *equidistant cylindrical
projection* which chooses a latitude scaling that preserves distance along
meridians. Other cylindrical projections are the Mercator (`projection='merc'`)
and the cylindrical equal area (`projection='cea'`).

```python
fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
draw_map(m)
```

`llcrnlat` and `urcrnlat` set the lower-left corner and upper-right corner
latitude for the map (and the `lon` equivalents set the longitude).

#### Pseudo-cylindrical projections

Pseudo-cylindrical projections relax the requirement that meridians (lines
of constant longitude) remain vertical; this can give better properties near
the poles of the projection. The Mollweide projection (`projection='moll'`) is
one example of this, in which all meridians are elliptical arcs. It is
constructed so as to preserve area across the map: though there are distortions
near the poles, the area of small patches reflects the true area. Other
pseudo-cylindrical projections are sinusoidal (`projection='sinu'`) and
Robinson (`projection='robin'`).

```python
fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='moll', resolution=None,
            lat_0=0, lon_0=0)
draw_map(m)
```

#### Perspective projections

Perspective projections are constructed using a particular choice of
perspective point, similar to if you photographed the Earth from a particular
point in space (a point which, for some projections, technically lies within
the Earth). One common example is the orthographic projection
(`projection='ortho'`) which shows one side of the globe as seen from a viewer
at a very long distance; it therefore can only show half the globe at a time.
Other perspective-based projections include the gnomonic projection
(`projection='gnom'`) and stereographic projection (`projection='stere'`).
These are often the most useful for showing small portions of the map.

```python
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None,
            lat_0=50, lon_0=0)
draw_map(m);
```

#### Conic projections

A conic projection projects the map onto a single cone, which is then unrolled.
This can lead to good local properties, but regions far from the focus point of
the cone may become distorted. One example is the Lambert Conformal Conic
projection (`projection='lcc'`): it projects the map onto a cone arranged in
such a way that two standard parallels (specified in Basemap by `lat_1` and
`lat_2`) have well-represented distances, with scale decreasing between them
and increasing outside of them. Other useful conic projections are the
equidistant conic projection (`projection='eqdc'`) and the Albers equal-area
projection (`projection='aea'`). Conic projections, like perspective
projections, tend to be good choices for representing small to medium patches
of the globe.

```python
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            lon_0=0, lat_0=50, lat_1=45, lat_2=55,
            width=1.6E7, height=1.2E7)
draw_map(m)
```

### Drawing a map background

The Basemap package contains a range of functions for drawing borders of
physical features, as well as political boundaries.

#### Physical boundaries and bodies of water

`drawcoastlines()`: Draw continental coast lines
`drawlsmask()`: Draw a mask between the land and sea, for use with projecting images on one or the other
`drawmapboundary()`: Draw the map boundary, including the fill color for oceans.
`drawrivers()`: Draw rivers on the map
`fillcontinents()`: Fill the continents with a given color; optionally fill lakes with another color

#### Political boundaries

`drawcountries()`: Draw country boundaries
`drawstates()`: Draw US state boundaries
`drawcounties()`: Draw US county boundaries

#### Map features

`drawgreatcircle()`: Draw a great circle between two points
`drawparallels()`: Draw lines of constant latitude
`drawmeridians()`: Draw lines of constant longitude
`drawmapscale()`: Draw a linear scale on the map

#### Whole-globe images

`bluemarble()`: Project NASA's blue marble image onto the map
`shadedrelief()`: Project a shaded relief image onto the map
`etopo()`: Draw an etopo relief image onto the map
`warpimage()`: Project a user-provided image onto the map

#### Resolution and boundary-based features

For the boundary-based features, you must set the desired resolution when
creating a Basemap image. The `resolution` argument of the `Basemap` class sets
the level of details in boundaries, either `'c'` (crude), `'l'` (low),
`'i'` (intermediate), `'h'` (high), `'f'` (full), or `None` if no boundaries
will be used. This choice is important: setting high-resolution boundaries on
a global map can be slow.

This is an example drawing land/sea boundaries, and shows the effect of the
resolution parameter, creating a low- and high-resolution map of the Isle of
Skye:

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 8))

for i, res in enumerate(['l', 'h']):
    m = Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2,
                width=90000, height=120000, resolution=res, ax=ax[i])
    m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
    m.drawmapboundary(fill_color="#DDEEFF")
    m.drawcoastlines()
    ax[i].set_title("resolution='{0}'".format(res));
```

Low-resolution coastlines are not suitable for this zoom level, but a low
resolution would suit a global view, and be much faster than loading
high-resolution border data for the globe. The best approach is to start with
a fast, low-resolution plot and increase the resolution as needed.

### Plotting data on maps

Basemap allows overplotting a variety of data onto a map background. For simple
plotting and text, any `plt` function works on the map; you can use the
`Basemap` instance to project latitude and longitude coordinates to `(x, y)`
coordinates for plotting with `plt`.

In addition, there are many map-specific functions available as methods of the
`Basemap` instance. These work very similarly to their standard Matplotlib
counterparts, but have an additional Boolean argument `latlon`, which if set
to `True` allows you to pass raw latitudes and longitudes to the method,
instead of projected `(x, y)` coordinates.

Some of these map-specific methods are:

* `contour()`/`contourf()` : Draw contour lines or filled contours
* `imshow()`: Draw an image
* `pcolor()`/`pcolormesh()` : Draw a pseudocolor plot for irregular/regular meshes
* `plot()`: Draw lines and/or markers.
* `scatter()`: Draw points with markers.
* `quiver()`: Draw vectors.
* `barbs()`: Draw wind barbs.
* `drawgreatcircle()`: Draw a great circle.

## Visualisation with Seaborn

Matplotlib is a useful tool, but it leaves much to be desired:

* the defaults are not the best choices. It was based off MATLAB circa
  1999, and this often shows.
* Matplotlib's API is quite low level. Sophisticated visualisation is
  possible, but often requires a lot of boilerplate code.
* Matplotlib predated pandas, and is not designed for use with
  `DataFrame`s. To visualise data from a `DataFrame`, you must extract
  each `Series` and often concatenate them together in the right format.

Matplotlib is addressing this, with the addition of `plt.style` and is
starting to handle pandas data more seamlessly, and with a new default
stylesheet in Matplotlib 2.0.

However, the Seaborn package also answers these problems, providing an
API on top of Matplotlib that offers good choices for plot style and
colour defaults, defines simple high-level functions for common
statistical plot types, and integrates with `DataFrame`s.

### Seaborn versus Matplotlib

A random walk plot:

```python
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd

# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

This plot will contain all the information we want to convey, but not in
an aesthetically pleasing way.

Seaborn has many of its own high-level plotting routines, but can also
overwrite Matplotlib's default parameters and get simple Matplotlib
scripts to produce superior output. We can set the style by calling
Seaborn's `set()` function.

```python
import seaborn as sns
sns.set()

# same plotting code as above!
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
```

### Exploring Seaborn plots

Seaborn provides high-level commands to create a variety of plot types
useful for data exploration, and some even for statistical model
fitting.

Note that the following data plots could be created using Matplotlib,
but Seaborn makes creating these much simpler.

#### Histograms, KDE and densities

Plotting histograms is simple in Matplotlib:

```python
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]],
size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)
```

Rather than a histogram, we can get a smooth estimate of the
distribution using a kernel density estimation, which Seaborn does with
`sns.kdeplot`:

```python
for col in 'xy':
    sns.kdeplot(data[col], shade=True)
```

Histograms and KDE can be combined using `distplot`:

```python
sns.distplot(data['x'])
sns.distplot(data['y']);
```

If we pass the full two dimensional dataset to `kdeplot`, we get a 2D
visualisation of the data.

```python
sns.kdeplot(data);
```

We can see the joint distribution and the marginal distributions
together using `sns.jointplot`.

```python
with sns.axes_style('white'): # Set white background.
    sns.jointplot("x", "y", data, kind='kde');
```

`jointplot` takes other parameters. For instance, we can create a hexagonal
histogram too:

```python
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');
```

#### Pair plots

When you generalise joint plots to datasets of larger dimensions, you
end up with *pair plots*. This is useful for exploring correlations
between multidimensional data, when you'd like to plot all pairs of
values against each other.

Using the iris dataset as an example:

```python
iris = sns.load_dataset("iris")
```

Visualizing the multidimensional relationships among the samples is as
easy as calling `sns.pairplot`:

```python
sns.pairplot(iris, hue='species', size=2.5);
```

#### Faceted histograms

Sometimes the best way to view data is via histograms of subsets.
Seaborn's `FacetGrid` makes this simple. As an example, tip data for
restaurant staff will be examined:

```python
tips = sns.load_dataset('tips')

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15));
```

#### Factor plots

Factor plots let you view the distribution of a parameter within bins
defined by any other parameter.

```python
with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill");
```

#### Joint distributions

Similar to the pairplot, we can use `sns.jointplot` to show the joint
distribution between different datasets, along with the associated
marginal distributions:

```python
with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')
```

The joint plot can even do some automatic kernel density estimation and
regression:

```python
sns.jointplot("total_bill", "tip", data=tips, kind='reg');
```

#### Bar plots

Time series can be plotted using `sns.factorplot`. Using the planets
data:

```python
planets = sns.load_dataset('planets')

with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)
```

We can learn more by looking at the method of discovery:

```python
with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')
```

#### Other notes

Seaborn can do regression too, using e.g. `regplot()` or `lmplot()`.
