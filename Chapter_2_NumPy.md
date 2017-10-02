# Chapter 2: NumPy

## Why NumPy?

Provides efficient storage and operations on homogeneous arrays.

NumPy arrays are homogeneous multidimensional arrays; the basic values (i.e.
values that aren't themselves arrays) inside arrays all have to be the same
type, and arrays can contain arrays.

Standard Python lists are flexible: allow hetereogeneous data to be
stored, but at a cost of more information needed and more indirection
required.

Python 3 has the array module for dense, homogeneous arrays to be stored
efficiently. What NumPy's `ndarray` type gives you over these array
objects is efficient operations too.

## Importing

Normally as:

```python
import numpy as np
```

## Creating arrays

### From Python lists

```python
>>> np.array([1, 4, 2, 5, 3])
array([1, 4, 2, 5, 3])
```

**NumPy arrays contain values of the same type: therefore may get implicit
conversion if possible.**

For example, for a mixed integer, floating point array:

```python
>>> np.array([3.14, 4, 2, 3])
array([ 3.14,  4.  ,  2.  ,  3.  ])
```

Can explicitly set type with `dtype`:

```python
>>> np.array([1, 2, 3, 4], dtype='float32')
array([ 1.,  2.,  3.,  4.], dtype=float32)
```

Can create multidimensional arrays, e.g. with a list of lists:

```python
>>> np.array([range(i, i + 3) for i in [2, 4, 6]])
array([[2, 3, 4],
       [4, 5, 6],
       [6, 7, 8]])
```

Inner lists are rows of the 2D array.

### From NumPy directly

1D array of zeroes:

```python
>>> np.zeros(10, dtype=int)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

2D array, that's 3x5:

```python
>>> np.ones((3, 5), dtype=float)
array([[ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.]])
```

Same again, but with a specified value filling the array:

```python
>>> np.full((3, 5), 3.14)
array([[ 3.14,  3.14,  3.14,  3.14,  3.14],
       [ 3.14,  3.14,  3.14,  3.14,  3.14],
       [ 3.14,  3.14,  3.14,  3.14,  3.14]])
```

Sequence starts at 0, ends at 20 (non-inclusive), steps by 2 (like `range()`):

```python
>>> np.arange(0, 20, 2)
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
```

Five values spaced between 0 and 1 evenly:

```python
>>> np.linspace(0, 1, 5)
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])
```

Random values in 3x3 array:

```python
>>> np.random.random((3, 3))
array([[ 0.99844933,  0.52183819,  0.22421193],
       [ 0.08007488,  0.45429293,  0.20941444],
       [ 0.14360941,  0.96910973,  0.946117  ]])
```

Can also use `np.random.normal()`, `np.random.randint()`.

Identity matrix:

```python
>>> np.eye(3)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
```

Uninitialized array, just contains whatever is currently in memory:

```python
>>> np.empty(3)
array([ 1.,  1.,  1.])
```

## Data types

Can specify these as `dtypes` as strings, e.g. `int16` or as NumPy
objects, e.g. `np.int16`.

Boolean type, various (un)signed integers, floats and complex numbers.

Also compound data types (see later).

## Working with arrays

### Array attributes

`.ndim`:     number of dimensions.  
`.shape`:    size of dimensions.  
`.size`:     total size of array.  
`.dtype`:    data type of array.  
`.itemsize`: size in bytes of each array element.  
`.nbytes`:   size in bytes of the array.  

### Array indexing

Can use Python-like indexing, square brackets, including negative
indices. Counting starts at zero, as usual with Python indexing.

For multidimensional arrays, can use comma-separated tuples of indices,

e.g. `a[0, 0]`.

### Value assignment in arrays

Can assign using e.g. `a[0, 0] = 3`. But, ensure you use the correct
type, e.g. putting a float into a NumPy array with integer type, will
convert it.

### Array slicing

Can slice as in Python, `x[start:stop:step]`.

Omitted values default to:

* `start`, 0;
* `stop`, size of dimension;
* `step`, 1.

Although, if `step` is negative, defaults for start and stop get
reversed. Can use `a[::-1]` to reverse an array.

#### Multidimensional array slicing

Separate slices for each dimension with commas, e.g. `a[:2, :3]` gives the
first two rows, and first three columns for a 2D array.

### Accessing rows or columns of 2D array

Combine indexing and slicing, e.g. `a[:, 0]` gives the first column of
`a` while `a[0, :]` is the first row. Can omit empty slice for a row,
e.g. `a[0]`.

### Slices are views, not copies!

Changing a value of a NumPy array slice changes the original array.

#### Copying arrays

If you really want a copy, can use `.copy()` on the slice.

### Array reshaping

Change a 1D array to a 3x3 grid:

```python
grid = np.arange(1, 10).reshape((3, 3))
print(grid)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

Reshaped array has to have same size as original. May be a view of the
initial array, where possible.

1D array into 2D row or matrix, can use reshape such as here to change
an array into a row vector.

```python
>>> x = np.array([1, 2, 3])
>>> x.reshape((1, 3))
array([[1, 2, 3]])
```

Alternatively, can use `np.newaxis`:

```python
>>> x[np.newaxis, :]
array([[1, 2, 3]])
```

and with columns:

```python
>>> x[:, np.newaxis] # equivalent to x.reshape((3, 1))
array([[1],
       [2],
       [3]])
```

### Array concatenation

`np.concatenate()` takes a tuple or list of arrays as first argument.

Can join multiple arrays.

Also, for multidimensional arrays. By default, it concatenates along the
first axis, but can specify an axis to concatenate along (zero-indexed).

`np.vstack()`, `np.hstack()` and `np.dstack()` can be clearer for arrays of
mixed dimensions.

### Array splitting

`np.split()`, `np.hsplit()`, `np.vsplit()`.

Pass a list of indices giving split points.

## Computation on arrays

### Universal functions (ufuncs)

Computation on arrays can be fast or slow. It is fast when using
vectorised operations, via NumPy's ufuncs.

If handling values individually, may think to use a loop. However, slow
due to overhead of type checking and looking up the correct function to
use for the type. If we knew the type before the code executes, we could
compute this faster as we could skip these steps.

Instead, use ufuncs which are optimised.

Perform operation on array instead, and this is applied to every
element, e.g. do `1.0/myarray`, not iterate through `myarray` and
compute the `1.0/myarray[i]` each time.

Can operate on two arrays, not just an individual value and an array:

```python
>>> np.arange(5) / np.arange(1, 6)
array([ 0.        ,  0.5       ,  0.66666667,  0.75      ,  0.8       ])
```

Also on multidimensional arrays:

```python
>>> x = np.arange(9).reshape((3, 3))
>>> 2 ** x
array([[  1,   2,   4],
       [  8,  16,  32],
       [ 64, 128, 256]])
```

**If looping through an array to compute something, consider whether
there's an appropriate ufunc instead.**

Both unary and binary ufuncs exist, i.e. operating on one or two arrays.

#### Mathematical functions

##### Arithmetic

`+`, `*`, `-`, `/`, `//` (integer division), `-` (negation), `**`
(exponentiation), `%` (modulus).

These are wrappers around NumPy functions, e.g. when you use `+` with an
array, you are really using `np.add()`.

##### Absolute value

Also, `abs()` is really `np.absolute()` or `np.abs()`; also works with
complex numbers.

##### Trigonometric functions

`np.sin()`, `np.cos()`, `np.tan()` and inverse functions: `np.arcsin()`,
`np.arccos()`, `np.arctan()`.

##### Exponents and logarithms

`np.exp()` uses e as base, `np.exp2()` uses 2 as base and
`np.power()` lets you specify a base (or bases, in an array).

`np.log()` is natural logarithm, can also have base 2, `np.log2()` or
base 10, `np.log10()`.

For small inputs, `np.expm1()` and `np.log1p()` to maintain greater
precision.

##### And more

More in NumPy itself, also lots more specialised functions for maths in
`scipy.special`.

#### Specifying array output

Can skip writing array values into a temporary array and then copying
into the target, by directly writing into the target.

```python
>>> x = np.arange(5)
>>> y = np.empty(5)
>>> np.multiply(x, 10, out=y)
```

Can use with array views too, e.g.:

```python
>>> x = np.arange(5)
>>> y = np.zeros(10)
>>> np.power(2, x, out=y[::2])
```

If we'd assigned as `y[::2] = 2 ** x`, this would produce a temporary
array, then copy the values to `y`.

#### Aggregates

##### `reduce`

Applies an operation repeatedly to elements of array until there is a
single result.

```python
>>> x = np.arange(1, 6)
>>> np.add.reduce(x)
15
```

##### `accumulate`

Applies an operation repeatedly to elements of array and stores each
result in turn.

```python
>>> x = np.arange(1, 6)
>>> np.multiply.accumulate(x)
array([  1,   2,   6,  24, 120])
```

However, note that NumPy has `np.sum()`, `np.prod()`, `np.cumsum()`,
`np.cumprod()` for these specific cases.

#### Outer products: `outer`

Computes output of all pairs of two inputs.

```python
>>> x = np.arange(1, 6)
>>> np.multiply.outer(x, x)
array([[ 1,  2,  3,  4,  5],
       [ 2,  4,  6,  8, 10],
       [ 3,  6,  9, 12, 15],
       [ 4,  8, 12, 16, 20],
       [ 5, 10, 15, 20, 25]])
```


### Aggregation functions

#### Summation

Can use Python's `sum()`, but this is slower than NumPy's `np.sum()` and
these two functions behave differently.

#### Minimum and maximum

Again, `np.min()` and `np.max()` are faster than `min()` and `max()` on
NumPy arrays.

Can also use method forms of these on the array itself:
`my_array.sum()`.

#### Multidimensional arrays

By default, aggregation occurs over the whole array.

However, can specify an axis along which to apply aggregation, e.g.
`my_array.min(axis=0)`.

```python
>>> M = np.random.random((3, 4))
>>> M
[[ 0.8967576   0.03783739  0.75952519  0.06682827]
 [ 0.8354065   0.99196818  0.19544769  0.43447084]
 [ 0.66859307  0.15038721  0.37911423  0.6687194 ]]
>>> M.min(axis=0)
array([ 0.66859307,  0.03783739,  0.19544769,  0.06682827])
>>> M.max(axis=1)
array([ 0.8967576 ,  0.99196818,  0.6687194 ])
```

For a 2D array, can think of the rows as a vertical axis, 0, and the columns as
a horizontal axis, 1.

This is slightly confusing terminology though. When you move "along the rows",
along axis 0, you're really moving through columns, aggregating into one row.
Could think of as moving along the rows for each column.

Likewise, when you move "along columns", along axis 1, you're really moving
across through a row, aggregating into one column. Could think of as moving
along the columns for each row.

If we move along the 0 axis, we move down the rows and compute the aggregation
for each column in the row.

If we move along the 1 axis, we move along the columns and compute the
aggregation for each row in the column.

The book describes this keyword `axis` as specifying "the dimension of the
array that will be collapsed". This is a simpler way of thinking about this,
especially for higher dimensional arrays.

Also note that when a dimension is "collapsed" in this way, the resulting array
actually has one fewer dimension; e.g. aggregating along columns for a 2D
array, we don't get a 2D array with just one item in each row; we actually get
an equivalent 1D array containing all of the items instead (because we don't
need any extent along the now removed dimension any longer).

(I guess that this is a nice feature because it means you get a consistent
shape of output for this aggregation regardless of the axis you aggregated
along.)

#### Other functions

Number of aggregation functions:

`np.sum()`, `np.prod()`, `np.mean()`, `np.std()`, `np.var()`,
`np.min()`, `np.max()`, `np.argmin()`, `np.argmax()`, `np.median()`,
`np.percentile()`

also with alternative `NaN` safe versions that ignore missing values.

For evaluating whether elements are true:

`np.any()`, `np.all()`

## Broadcasting

For arrays of the same size, binary operations are applied
element-by-element.

```python
>>> a = np.array([0, 1, 2])
>>> b = np.array([5, 5, 5])
>>> a + b
array([5, 6, 7])
```

Broadcasting extends this idea: it is a set of rules for applying ufuncs
to arrays of different sizes.

e.g. add a scalar to an array:

```python
>>> a + 5
array([5, 6, 7])
```

Can imagine that the value 5 is duplicated into an array [5, 5, 5] and
then added to `a`, though this doesn't really happen (and is therefore
an advantage of how NumPy works).

Can extend this to higher dimension arrays; add a 1D array to a 2D array:

```python
>>> M = np.ones((3, 3))
>>> M + a
array([[ 1.,  2.,  3.],
       [ 1.,  2.,  3.],
       [ 1.,  2.,  3.]])
```

`a` now gets stretched, or broadcast, across the second dimension to match M.

Sometimes *both* arrays can be broadcast this way.

```python
>>> a = np.arange(3)
>>> b = np.arange(3)[:, np.newaxis]
>>> print(a)
[0 1 2]
>>> print(b)
[[0]
 [1]
 [2]]
>>> a + b
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
```

Both `a` and `b` are stretched to a common shape, `a` is stretched along rows,
while `b` is stretched along columns, the 0, 1, 2 values can be thought
of as being replicated (although they aren't). Both are broadcast to arrays
with three rows and columns, and then the operation is applied.

### Broadcasting rules

When arrays interact, broadcasting follows these rules:

1. If two arrays differ in their number of dimensions, the shape of the
   one with fewer dimensions is left padded with ones.
2. If the shape of the two arrays doesn't match in any dimension, the
   array with shape equal to 1 in that dimension is stretched to match
   the other shape.
3. If in any dimension the sizes disagree and neither is equal to 1,
   an error is raised.

Applies to all binary ufuncs.

### Examples

#### Adding 2D array to 1D array

```python
>>> M = np.ones((2, 3))
>>> a = np.arange(3)
```

`M.shape` = `(2, 3)`
`a.shape` = `(3,)`

Rule 1: `a` has fewer dimensions, left pad with 1:

`a.shape` = `(1, 3)`

Rule 2: the first dimensions of `a` and `M` don't match, so stretch `a`
to match `M` as `a` has shape 1 in first dimension.

`a.shape` = `(2, 3)` and now these arrays can be added. Effectively, can
think of `a` getting stretched along rows, so the `[0, 1, 2]` gets
duplicated.

```python
>>> M + a
array([[ 1.,  2.,  3.],
       [ 1.,  2.,  3.]])
```

#### Broadcasting both arrays

```python
>>> a = np.arange(3).reshape((3, 1))
>>> b = np.arange(3)
```

`a.shape = (3, 1)`
`b.shape = (3,)`

Rule 1: `b` has fewer dimensions, left pad with 1:

`b.shape = (1, 3)`

Rule 2: stretch array dimensions that are 1 to match the corresponding
dimension of the other array. Here, *both* arrays are stretched.

`a.shape = (3, 3)`
`b.shape = (3, 3)`

Now we can add them.

```python
>>> a + b
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
```

#### Where broadcasting fails

```python
>>> M = np.ones((3, 2))
>>> a = np.arange(3)
```

`M.shape = (3, 2)`
`a.shape = (3,)`

Rule 1: left pad `a`.

`a.shape = (1, 3)`

Rule 2: stretch first dimension of `a` to match `M`:

`a.shape = (3, 3)`

Rule 3: here, the final shapes of `a` and `M` don't match, so an error occurs
if you try and do `M + a`.

NB: these arrays *would* be compatible if only we padded `a` on the right,
instead of the left (because `a.shape` would be `(3, 1)` which could then be
stretched to `(3, 2)` which matches `m.shape`). However, if you wish to do
this, you have to do it explicitly first; broadcasting's rules don't allow
this:

```python
>>> a[:, np.newaxis].shape
(3, 1)
>>> M + a[:, np.newaxis]
array([[ 1.,  1.],
       [ 2.,  2.],
       [ 3.,  3.]])
```

#### Uses of broadcasting

Examples:

* centring an array of observations, where each contains multiple values.
  Can calculate the mean of these values, which is a 1D array, and subtract
  from our 2D array of observations, and the subtraction occurs via
  broadcasting the 1D array.

* calculating/plotting a 2D function; can create a range of values in a 1D
  array and another range of values in a 2D array with one column per row as a
  series of x and y values to calculate z for, then use broadcasted values to
  calculate this for each combination of x and y.

* likewise, a times table is another example, can have 1-10 in a 1D array,
  1-10 in a 2D array with a single column, and then do the multiplication by
  broadcasting both arrays. (So similar function here to outer product.)

## Comparisons, masks and Boolean logic

### Comparison operators as ufuncs

Comparison operators are implemented as ufuncs.

* `<` as `np.less`
* `>` as `np.greater`
* `<=` as `np.less_equal`
* `>=` as `np.greater_equal`
* `==` as `np.equal`
* `!=` as `np.not_equal`

The output of these is an array with Boolean data type.

```python
>>> x = np.array([1, 2, 3, 4, 5])
>>> x < 3
array([ True,  True, False, False, False], dtype=bool)
```

Can also do element-wise comparison of arrays and use compound
expressions:

```python
>>> (2 * x) == (x ** 2)
array([False,  True, False, False, False], dtype=bool)
```

Work on any size or shape of array.

### Working with Boolean arrays

`np.count_nonzero()` for counting the number of `True` entries in a
Boolean array.

```python
>>> y = array([[5, 0, 3, 3],
               [7, 9, 3, 5],
               [2, 4, 7, 6]])
>>> np.count_nonzero(y < 6)
8
```

Can also use `np.sum()` where `True` is interpreted as `1` and `False`
as `0`. The advantage of this is that you can do this `sum` along an
axis, e.g. to find the counts along rows or columns.

```python
>>> np.sum(y < 6, axis=1)
array([4, 2, 2])
```

This calculates the sum in each row.

For checking whether any or all values are `True`, can use `np.any()` or
`np.all()`.

```python
>>> np.all(y < 10)
True
```

and can use both `np.any()` and `np.all()` along axes.

```python
>>> np.all(y < 8, axis=1)
array([ True, False, True], dtype=bool)
```

**Ensure you use the NumPy `np.sum()`, `np.any()` and `np.all()`
functions, not the Python built-ins `sum()`, `any()` and `all()` as
these may not behave as expected.**

### Boolean operators

Can also use Boolean operators.

Use Python's bitwise logic operators: `&` (bitwise and), `|` (bitwise
or), `^` (bitwise exclusive or), `~` (complement).

These have the equivalent ufuncs:

* `&` `np.bitwise_and()`
* `|` `np.bitwise_or()`
* `^` `np.bitwise_xor()`
* `~` `np.bitwise.not()`

Can combine these on an array representing rainfall data:

```python
>>> np.sum((inches > 0.5) & (inches < 1))
```

Need the parentheses, otherwise `0.5 & inches` gets evaluated first and
results in an error.

This *A AND B* condition is equivalent to:

```python
>>> np.sum(~( (inches <= 0.5) | (inches >= 1) ))
```

(Since this represents *NOT (NOT A OR NOT B)* and *NOT A OR NOT B* is
the same as *NOT (A AND B)*.)

#### Why bitwise operators?

`and`, `or` operate on *entire objects*, while `&` and `|` refer to
*bits wihin an object*.

When we have Boolean values in a NumPy array, effectively this is a
series of bits where `1` is `True` and `0` is `False`. We therefore can
use the bitwise operators to carry out these operations, as these are
applied to individual bits within the array.

On the other hand, the Boolean value of an array with more than one
object is ambiguous.

### Boolean arrays as masks

Can use Boolean arrays as masks, to select subsets of data.

```python
>>> y < 5
array([[False,  True,  True,  True],
       [False, False,  True, False],
       [ True,  True, False, False]], dtype=bool)
```

This gives a Boolean array. We can use this array as an index value; this is
a *masking* operation.

```python
>>> y[y < 5]
array([0, 3, 3, 3, 2, 4])
```

The result is a 1D array that contains the values that meet the
condition, i.e. where the mask array is `True`.

## Fancy indexing

So far, indexing via indices, slices and Boolean masks.

Fancy indexing is where arrays of indices are passed, for accessing or
modifying complicated subsets of array values.

### Simple uses

```python
>>> import numpy as np
>>> rand = np.random.RandomState(42)
>>> x = rand.randint(100, size=10)
[51 92 14 71 60 20 82 86 74 74]
```

Could access different elements by individually accessing indices:

```python
>>> [x[3], x[7], x[4]]
[71, 86, 60]
```

or by passing a list or array of indices:

```python
>>> x[[3, 7, 4]]
array([71, 86, 60])
```

When using fancy indexing, the shape of the result reflects the shape of
the index array, not the shape of the array being indexed:

```python
>>> ind = np.array([[3, 7],
                    [4, 5]])
>>> x[ind]
array([[71, 86],
       [60, 20]])
```

### With multidimensional arrays

```python
>>> X = np.arange(12).reshape((3, 4))
>>> X
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
```

```python
>>> row = np.array([0, 1, 2])
>>> col = np.array([2, 1, 3])
>>> X[row, col]
array([ 2,  5, 11])
```

The first index refers to the row, the second index to the column, so we
effectively get the `X` values `[0, 2]`, `[1, 1]`, `[2, 3]`.

Pairing of indices also follows broadcasting rules.

```python
>>> X[row[:, np.newaxis], col]
array([[ 2,  1,  3],
       [ 6,  5,  7],
       [10,  9, 11]])
```

Each value from `row` gets matched with a value from `col`; both these
arrays are broadcast, with the `row` column getting duplicated left to
right and the `col` rows getting duplicated top to bottom.

So, we get values `[0, 2]`, `[0, 1]`, `[0, 3]`, then `[1, 2]`, `[1, 1]`,
`[1, 3]` etc.

### Combined indexing

Can combine these different ways of indexing.

Fancy and simple indices:

```python
>>> X[2, [2, 0, 1]]
array([10,  8,  9])
```

Fancy indexing and slices:

```python
>>> X[1:, [2, 0, 1]]
array([[ 6,  4,  5],
       [10,  8,  9]])
```

Fancy indexing and masking:

```python
>>> mask = np.array([1, 0, 1, 0], dtype=bool)
>>> X[row[:, np.newaxis], mask]
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10]])
```

### Modifying values

Simple example where you set values of an array according to an array of
indices.

```python
>>> x = np.arange(10)
>>> i = np.array([2, 1, 8, 4])
>>> x[i] = 99
>>> x
[ 0 99 99  3 99  5  6  7 99  9]
```

Can use any assignment-type operator.

```python
>>> x[i] -= 10
>>> x
[ 0 89 89  3 89  5  6  7 89  9]
```

Repeated indices can give unexpected behaviour.

```python
>>> x = np.zeros(10)
>>> x[[0, 0]] = [4, 6]
>>> x
[ 6.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
```

What happens is that the `x[0] = 4` assignment happens first, then `x[0] = 6`.

But:

```python
>>> i = [2, 3, 3, 4, 4, 4]
>>> x[i] += 1
>>> x
array([ 6.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])
```

Incrementing doesn't occur repeatedly.

`x[i] += 1` means `x[i] = x[i] + 1`.

The evaluation of `x[i] + 1` happens first, then the result is assigned
to each index. So, the assignment happens repeatedly, not the increment.

If you do want repeated incrementing, then you can do:

```python
>>> x = np.zeros(10)
>>> np.add.at(x, i, 1)
>>> x
[ 0.  0.  1.  2.  3.  0.  0.  0.  0.  0.]
```

Here, `at()` applies the operator (`add`) at the indices `i` with the
value 1.

## Sorting arrays

`np.sort()` uses quicksort, with mergesort and heapsort available.

To get a new sorted array:

```python
>>> x = np.array([2, 1, 4, 3, 5])
>>> np.sort(x)
array([1, 2, 3, 4, 5])
```

Sort in-place:

```python
>>> x.sort()
```

`np.argsort()` returns indices of sorted elements as an array. Could use
this with fancy indexing to get the sorted array.

```python
>>> x = np.array([2, 1, 4, 3, 5])
>>> i = np.argsort(x)
>>> print(i)
[1 0 3 2 4]
>>> x[i]
array([1, 2, 3, 4, 5])
```

### Sort along rows or columns

Use `axis`.

```python
>>> rand = np.random.RandomState(42)
>>> X = rand.randint(0, 10, (4, 6))
>>> print(X)
[[6 3 7 4 6 9]
 [2 6 7 4 3 7]
 [7 2 5 4 1 7]
 [5 1 4 0 9 5]]
>>> np.sort(X, axis=0)
array([[2, 1, 4, 0, 1, 5],
       [5, 2, 5, 4, 3, 7],
       [6, 3, 7, 4, 6, 7],
       [7, 6, 7, 4, 9, 9]])
```

### Partial sorts by `np.partition`

Finds the smallest `k` values in the array and returns a new array with the
smallest `k` values to the left of the partition and the remaining values to
the right, in arbitrary order:

```python
>>> x = np.array([7, 2, 3, 1, 6, 5, 4])
>>> np.partition(x, 3)
array([2, 1, 3, 4, 6, 5, 7])
```

Can also use along an axis of a multidimensional array.

```python
>>> np.partition(X, 2, axis=1)
array([[3, 4, 6, 7, 6, 9],
       [2, 3, 4, 7, 6, 7],
       [1, 2, 4, 5, 7, 7],
       [0, 1, 4, 5, 9, 5]])
```

Also `np.argpartition()` available and this is analogous to `np.argsort()`.

## Structured arrays

Often data can be represented by homogeneous values. Not always. NumPy
has some support for compound, heterogeneous values in arrays. For
simple cases, can use structured arrays and record arrays. But, for more
complex cases, better to use pandas instead.

Strcutured arrays are arrays with compound data types.

```python
>>> name = ['Alice', 'Bob', 'Cathy', 'Doug']
>>> age = [25, 45, 37, 19]
>>> weight = [55.0, 85.5, 68.0, 61.5]
>>> data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                              'formats':('U10', 'i4', 'f8')})
>>> print(data.dtype)
[('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]
```

`U10`: Unicode string, maximum length 10.  
`i4`: 4-byte integer.
`f8`: 8-byte float.

```python
>>> data['name'] = name
>>> data['age'] = age
>>> data['weight'] = weight
>>> print(data)
[('Alice', 25, 55.0) ('Bob', 45, 85.5) ('Cathy', 37, 68.0)
 ('Doug', 19, 61.5)]
```

Can identify values by index or name:

```python
>>> data['name']
array(['Alice', 'Bob', 'Cathy', 'Doug'], 
      dtype='<U10')
>>> data[0]
('Alice', 25, 55.0)
>>> data[-1]['name']
'Doug'
>>> data[data['age'] < 30]['name'] # Boolean masking
array(['Alice', 'Doug'], 
      dtype='<U10')
```

Can create structured arrays by specifying Python types or NumPy `dtypes` in
formats as above. Can also create them as a list of tuples:

```python
>>> np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
dtype([('name', 'S10'), ('age', '<i4'), ('weight', '<f8')])
```

or as a comma-separated string:

```python
>>> np.dtype('S10,i4,f8')
dtype([('f0', 'S10'), ('f1', '<i4'), ('f2', '<f8')])
```

Optional `<` or `>` indicates little or big endian; the next character
indicates the type of data and the last character represents the size in bytes.

Note that these NumPy structured array `dtypes` map directly to C
structure definitions.

### Record arrays

`np.recarray` are record arrays, like structured arrays but can access fields
as attributes instead of dictionary keys. However, these fields can be slower
to access than in structured arrays, even when using dictionary keys.
