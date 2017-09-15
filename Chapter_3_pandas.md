# Chapter 3: pandas

## Introduction

pandas is built on NumPy. Provides `DataFrame`s that are essentially
multidimensional arrays with row and column labels, and often have
heterogeneous types and/or missing data. `Series` is another important
pandas type and is a 1D array of indexed data.

pandas provides operations that you might expect in database or
spreadsheet tools, and avoids some of the limitations of NumPy in terms
of flexibility (e.g. dealing with missing data), and for operations that
don't work well with element-wide broadcasting (e.g. groupings, pivots).

## Importing

Usually as:

```python
import pandas as pd
```

## pandas objects

Can think of these as enhanced versions of NumPy structured arrays where
the rows and columns are identified with labels instead of indices.

### pandas `Series` object

A pandas `Series` is a 1D array of indexed data.

Can create from a list or array like:

```python
>>> data = pd.Series([0.25, 0.5, 0.75, 1.0])
>>> data
0    0.25
1    0.50
2    0.75
3    1.00
dtype: float64
```

A `Series` wraps a sequence of values and a sequence of indices. Access these
with `values` and `index` attributes.

`values` is just a NumPy array:

```python
>>> data.values
array([ 0.25,  0.5 ,  0.75,  1.  ])
```

`index` is of `pd.Index` type and is an array-like object:

```python
>>> data.index
RangeIndex(start=0, stop=4, step=1)
```

Can access data using usual Python index notation:

```python
>>> data[1]
0.5
>>> data[1:3]
1    0.50
2    0.75
dtype: float64
```

#### `Series` as a generalised NumPy array

`Series` looks a bit like a NumPy 1D array, but a key difference is
presence of an explicitly defined index associated with values. (NumPy
arrays have an implicitly defined integer index.)

So, we can use an index of strings:

```python
>>> data = pd.Series([0.25, 0.5, 0.75, 1.0],
                     index=['a', 'b', 'c', 'd'])
>>> data
a    0.25
b    0.50
c    0.75
d    1.00
dtype: float64
>>> data['b']
0.5
```

Or non-sequential indices:

```python
>>> data = pd.Series([0.25, 0.5, 0.75, 1.0],
                     index=[2, 5, 3, 7])
>>> data
2    0.25
5    0.50
3    0.75
7    1.00
dtype: float64
>>> data[5]
0.5
```

#### `Series` as a specialised dictionary

`Series` can also be considered like a more specialised Python dictionary.

A dictionary maps *arbitrary* keys to *arbitrary* values. A `Series` maps
*typed* keys to *typed* values.

This typing is important for efficiency (see the discussion on this for
NumPy arrays in previous chapter).

In fact, can construct a `Series` from a dictionary:

```python
>>> population_dict = {'California': 38332521,
                       'Texas': 26448193,
                       'New York': 19651127,
                       'Florida': 19552860,
                       'Illinois': 12882135}
>>> population = pd.Series(population_dict)
>>> population
California    38332521
Florida       19552860
Illinois      12882135
New York      19651127
Texas         26448193
dtype: int64
```

Note that by default, here, the index is drawn from the sorted keys.

Data can be accessed just like a dictionary:

```python
>>> population['California']
38332521
```

but can also be sliced like an array:

```python
>>> population['California':'Illinois']
California    38332521
Florida       19552860
Illinois      12882135
dtype: int64
```

#### Constructing `Series` objects

Examples above use some form of:

```python
>>> pd.Series(data, index=index)
```

With `data` as list or NumPy array, `index` defaults to integer
sequence:

```python
>>> pd.Series([2, 4, 6])
0    2
1    4
2    6
dtype: int64
```

`data` can be a scalar, which gets repeated to fill the specified index:

```python
>>> pd.Series(5, index=[100, 200, 300])
100    5
200    5
300    5
dtype: int64
```

`data` can be a dictionary, with `index` defaulting to the sorted keys:

```python
>>> pd.Series({2:'a', 1:'b', 3:'c'})
1    b
2    a
3    c
dtype: object
```

Can set an index too:

```python
>>> pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
3    c
2    a
dtype: object
```

Notice that the `Series` here is populated only with the explicitly stated
keys.

### pandas `DataFrame` object

#### `DataFrame` as a generalised NumPy array

If a `Series` is analogous to a 1D NumPy array with flexible indices, a
`DataFrame` is an analogue of a 2D array with flexible row indices *and*
flexible column names.

You could think of a 2D array as an ordered sequence of aligned 1D
columns, likewise, you can think of a `DataFrame` as a sequence of
aligned `Series` objects; "aligned" being that they share the same
index.

I think it's probably important to not push this analogy too far. Where
this breaks down for me is that a 2D NumPy array is really an ordered
sequence of rows, where the rows contain the columns. Yes, you can
access each column by slicing and indexing, so you could think of the
array as being made up of a series of columns, but that's not really how
you construct the array.

This all implies that you can think of the 2Dness of a general array either as:

* we have stacked columns, each column is a 1D array, and stacking them
  together means that we can also move through columns, which is a second
  dimension, or

* we have stacked rows, each row is a 1D array, and we can move
  along rows vertically or along columns horizontally. Moving through the rows
  is the newly added dimension.

For NumPy arrays, the latter is more applicable; for pandas `DataFrames`, the
former is more applicable.

However, whatever way these structures are made, they are both 2D arrays,
comprising rows and columns, from stacked 1D arrays. And you can move through
both either by row or by column, but the way they're constructed is actually
different (and I think is likely due to their origins, NumPy from matrix
algebra, and pandas from a table-focused data analysis viewpoint).

Though also note that the `index` of a `DataFrame` actually refers to the rows,
not the columns (which are accessed by `columns`).

The author does point out the potential confusion later.

```python
>>> area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
                 'Florida': 170312, 'Illinois': 149995}
>>> area = pd.Series(area_dict)
>>> area
California    423967
Florida       170312
Illinois      149995
New York      141297
Texas         695662
dtype: int64
```

Along with the population `Series` above, we can use a dictionary to
construct a 2D object containing this information:

```python
>>> states = pd.DataFrame({'population': population,
                           'area': area})
>>> states
              area  population
California  423967    38332521
Florida     170312    19552860
Illinois    149995    12882135
New York    141297    19651127
Texas       695662    26448193
```

As with the `Series` object, `DataFrame`s have an `index` attribute that give
access to the index labels:

```python
>>> states.index
Index(['California', 'Florida', 'Illinois', 'New York', 'Texas'], dtype='object')
```

and they also have a `columns` attribute, with an `Index` holding the column
labels:

```python
>>> states.columns
Index(['area', 'population'], dtype='object')
```

So, a `DataFrame` can be thought of as a generalisation of a 2D NumPy array,
where both the rows and columns have a generalised index for accessing the
data.

#### `DataFrame` as a specialised dictionary

Where dictionaries map keys to values, a `DataFrame` maps a column name to a
`Series` of column data, e.g.:

```python
>>> states['area']
California    423967
Florida       170312
Illinois      149995
New York      141297
Texas         695662
Name: area, dtype: int64
```

NB: `data[0]` refers to the first row in a NumPy 2D array, whereas for a
`DataFrame`, `data['col0']` refers to the first column. This is potentially
confusing, and therefore maybe more useful to think of `DataFrames` as
these dictionaries, although both ways of thinking about them can be useful.

####  Constructing `DataFrame` objects

##### From a dictionary of `Series` objects

See above example.

##### From a single `Series` object

A `DataFrame` is a collection of `Series` objects, and a single-column
`DataFrame` can be constructed from a single `Series`:

```python
>>> pd.DataFrame(population, columns=['population'])
            population
California    38332521
Florida       19552860
Illinois      12882135
New York      19651127
Texas         26448193
```

##### From a list of dicts

```python
>>> data = [{'a': i, 'b': 2 * i}
            for i in range(3)]
>>> pd.DataFrame(data)
   a  b
0  0  0
1  1  2
2  2  4
```

If keys are missing, Pandas fills them with `NaN` (not a number) value.

```python
>>> pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
     a  b    c
0  1.0  2  NaN
1  NaN  3  4.0
```
##### From a 2D NumPy array

```python
>>> pd.DataFrame(np.random.rand(3, 2),
                 columns=['foo', 'bar'],
                 index=['a', 'b', 'c'])
        foo       bar
a  0.022492  0.062436
b  0.107422  0.260997
c  0.510114  0.848840
```

##### From a NumPy structured array

```python
>>> A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
>>> pd.DataFrame(A)
   A    B
0  0  0.0
1  0  0.0
2  0  0.0
```

### pandas `Index` object

Both `Series` and `DataFrame` objects contain an explicit index to reference
and modify data. This `Index` object can be thought of as immutable array or
as ordered set (really a multiset, since `Index` objects may contain repeated
values).

```python
>>> ind = pd.Index([2, 3, 5, 7, 11])
>>> ind
Int64Index([2, 3, 5, 7, 11], dtype='int64')
```

#### `Index` as immutable array

Use index notation to retrieve values or slices:

```python
>>> ind[1]
2
>>> ind[::2]
Int64Index([2, 5, 11], dtype='int64')
```

Also features attributes common to NumPy arrays:

```python
>>> print(ind.size, ind.shape, ind.ndim, ind.dtype)
5 (5,) 1 int64
```

However, `Index` objects are immutable, so can't be modified via normal means:

```python
>>> ind[1] = 0
```

causes a `TypeError`.

This means you can share indices more safely between multiple `DataFrames`
and arrays, without the risk of side effects of index modification.

#### `Index` as ordered set

pandas objects are designed to facilitate operations such as joins across
datasets, which depend on many aspects of set arithmetic. The `Index` object
follows many of the conventions used by Python's `set` data type, so that
operations work in a familiar way:

```python
>>> indA = pd.Index([1, 3, 5, 7, 9])
>>> indB = pd.Index([2, 3, 5, 7, 11])
>>> indA & indB  # intersection
Int64Index([3, 5, 7], dtype='int64')
>>> indA | indB  # union
Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')
>>> indA ^ indB  # symmetric difference
Int64Index([1, 2, 9, 11], dtype='int64')
```

These operations can be accessed by object methods, e.g.
`indA.intersection(indB)`.

## Data indexing and selection

Some similarities between working with NumPy arrays, and pandas `Series`
and `DataFrame` objects.

### Data selection in `Series`

As above, can think of a `Series` as acting in many ways like a 1D NumPy
array and also like a Python dictionary.

#### `Series` as dictionary

As above:

```python
>>> data = pd.Series([0.25, 0.5, 0.75, 1.0],
                     index=['a', 'b', 'c', 'd'])
>>> data
a    0.25
b    0.50
c    0.75
d    1.00
dtype: float64
>>> data['b']
0.5
```

Can also use dictionary-like Python expressions and methods:

```python
>>> 'a' in data
True
>>> data.keys()
Index(['a', 'b', 'c', 'd'], dtype='object')
>>> list(data.items())
[('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)]
```

Can extend a `Series` by assigning to a new index value:

```python
>>> data['e'] = 1.25
>>> data
a    0.25
b    0.50
c    0.75
d    1.00
e    1.25
dtype: float64
```

#### `Series` as 1D array

`Series` objects also provide ways to select items much like NumPy
arrays (slicing, masking, fancy indexing):

```python
>>> # slicing by explicit index
>>> # NB: the final index is included in the slice.
>>> data['a':'c']
a    0.25
b    0.50
c    0.75
dtype: float64

>>> # slicing by implicit integer index
>>> # NB: the final index is not included in the slice.
>>> data[0:2]
a    0.25
b    0.50
dtype: float64

>>> # masking
>>> data[(data > 0.3) & (data < 0.8)]
b    0.50
c    0.75
dtype: float64

>>> # fancy indexing
>>> data[['a', 'e']]
a    0.25
e    1.25
dtype: float64
```

#### Indexers: `loc`, `iloc` and `ix`

Slicing and indexing can be confusing. If a `Series` has an explicit
integer index, indexing as `data[1]` uses the *explicit* index, but a
slicing operation like `data[1:3]` will use the *implicit* index.

```python
>>> data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
>>> data
1    a
3    b
5    c
dtype: object

>>> # explicit index when indexing
>>> data[1]
'a'

>>> # implicit index when slicing
>>> data[1:3]
3    b
5    c
dtype: object
```

Because of this, pandas provides special *indexer* attributes to expose
certain indexing schemes.

Useful to use these explicitly to make code easier to read and prevent
bugs due to the different behaviour of indexing and slicing.

##### `loc`

The `loc` attribute references the explicit index for indexing and
slicing (label-location based):

```python
>>> data.loc[1]
'a'
>>> data.loc[1:3]
1    a
3    b
dtype: object
```

##### `iloc`

The `iloc` attribute references the implicit Python-style index for
indexing and slicing (integer-location based):

```python
>>> data.iloc[1]
'b'
>>> data.iloc[1:3]
3    b
5    c
dtype: object
```

##### `ix`

The `ix` attribute is a hybrid of the two, and for `Series` objects is
equivalent to standard `[]`-based indexing. See more below regarding
`DataFrame`s.

(From pandas docs: label-location based, but falls back to
integer-location based unless corresponding axis is of integer type.)

### Data selection in `DataFrame`

A `DataFrame` acts in many ways like a 2D or structured array, and in
other ways like a dictionary of `Series` structures with the same index.

#### `DataFrame` as a dictionary

```python
>>> area = pd.Series({'California': 423967, 'Texas': 695662,
                      'New York': 141297, 'Florida': 170312,
                      'Illinois': 149995})
>>> pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                     'New York': 19651127, 'Florida': 19552860,
                     'Illinois': 12882135})
>>> data = pd.DataFrame({'area':area, 'pop':pop})
>>> data
              area         pop
California  423967    38332521
Florida     170312    19552860
Illinois    149995    12882135
New York    141297    19651127
Texas       695662    26448193
```

Can access each of the `Series` that make up the `DataFrame` columns by
dictionary-style indexing:

```python
>>> data['area']
California    423967
Florida       170312
Illinois      149995
New York      141297
Texas         695662
Name: area, dtype: int64
```

or by attribute access with string column names:

```python
>>> data.area
California    423967
Florida       170312
Illinois      149995
New York      141297
Texas         695662
Name: area, dtype: int64
```

Both of these access the same object:

```python
>>> data.area is data['area']
True
```

NB: doesn't work in all cases:

* if the column names aren't strings.  
* if the column names conflict with names of methods of the `DataFrame`
  (e.g. `data.pop` is the `pop` method, it is not the object `data['pop']`.)

Can modify `DataFrame`s via dictionary-style indexing:

```python
>>> data['density'] = data['pop'] / data['area']
>>> data
              area       pop     density
California  423967  38332521   90.413926
Florida     170312  19552860  114.806121
Illinois    149995  12882135   85.883763
New York    141297  19651127  139.076746
Texas       695662  26448193   38.018740
```

See later for more examples of element-by-element arithmetic.

#### `DataFrame` as 2D array

Examine the underlying array using the `values` attribute:

```python
>>> data.values
array([[  4.23967000e+05,   3.83325210e+07,   9.04139261e+01],
       [  1.70312000e+05,   1.95528600e+07,   1.14806121e+02],
       [  1.49995000e+05,   1.28821350e+07,   8.58837628e+01],
       [  1.41297000e+05,   1.96511270e+07,   1.39076746e+02],
       [  6.95662000e+05,   2.64481930e+07,   3.80187404e+01]])
```

With that in mind, can make array-like observations on the `DataFrame`,
e.g. transpose rows and columns:

```python
>>> data.T
           California       Florida      Illinois      New York         Texas
area     4.239670e+05  1.703120e+05  1.499950e+05  1.412970e+05  6.956620e+05
pop      3.833252e+07  1.955286e+07  1.288214e+07  1.965113e+07  2.644819e+07
density  9.041393e+01  1.148061e+02  8.588376e+01  1.390767e+02  3.801874e+01
```

However, a `DataFrame` can't be treated simply like a NumPy array, e.g.
accessing a single index of an array accesses a row:

```python
>>> data.values[0]
array([  4.23967000e+05,   3.83325210e+07,   9.04139261e+01])
```

but passing a single "index" to a `DataFrame` accesses a column, e.g.
`data['area']`.

(Although confusingly, you can also pass a single slice to a `DataFrame`
and access the rows. Also see below.)

For array-style indexing, we can use the `loc`, `iloc` and `ix`
indexers.

`iloc` allows indexing of the underlying array as if it is a simple NumPy
array, using the implicit Python-style index, but preserving the `DataFrame`
index and column labels.

```python
>>> data.iloc[:3, :2]
              area       pop
California  423967  38332521
Florida     170312  19552860
Illinois    149995  12882135
```

`loc` allows indexing the underlying data in array-style, but using
the explicit index and column names:

```python
>>> data.loc[:'Illinois', :'pop']
              area       pop
California  423967  38332521
Florida     170312  19552860
Illinois    149995  12882135
```

`ix` allows a hybrid of `iloc` and `loc` approaches:

```python
>>> data.ix[:3, :'pop']
              area       pop
California  423967  38332521
Florida     170312  19552860
Illinois    149995  12882135
```

But can be confusing where explicit indices are integers.

Can use the previously discussed NumPy data access patterns with these
indexers; for example, masking and fancy indexing:

```python
>>> data.loc[data.density > 100, ['pop', 'density']]
               pop     density
Florida   19552860  114.806121
New York  19651127  139.076746
```

And can set or modify values using these conventions, e.g.:

```python
>>> data.iloc[0, 2] = 90
>>> data
              area       pop     density
California  423967  38332521   90.000000
Florida     170312  19552860  114.806121
Illinois    149995  12882135   85.883763
New York    141297  19651127  139.076746
Texas       695662  26448193   38.018740
```

### Additional indexing conventions

Indexing refers to columns, while slicing refers to rows.

```python
>>> data['Florida':'Illinois']
            area       pop     density
Florida   170312  19552860  114.806121
Illinois  149995  12882135   85.883763
```

and slices can refer to rows by number instead of index:

```python
>>> data[1:3]
            area       pop     density
Florida   170312  19552860  114.806121
Illinois  149995  12882135   85.883763
```

Likewise, direct masking operations are interpreted row-wise, not column-wise:

```python
>>> data[data.density > 100]
            area       pop     density
Florida   170312  19552860  114.806121
New York  141297  19651127  139.076746
```

These are similar to those on a NumPy array.

## Operating on data in pandas

NumPy provides quick element-wise operations via ufuncs for basic
arithmetic and more complicated operations (e.g. trigonometric
functions). pandas inherits much of this functionality.

Also features a couple of differences:

* for unary operations like negation and trigonometric functions, ufuncs
  preserve index and column labels in the output.  
* for binary operations such as addition and multiplication, pandas will
  automatically align indices when passing the objects to the ufunc.

This means that keeping the context of data and combining data from
different sources is much easier to do without error than with NumPy arrays.

### ufunc: index preservation

Since pandas is designed to work with NumPy, NumPy ufuncs work with
pandas `Series` and `DataFrame` objects.

```python
>>> rng = np.random.RandomState(42)
>>> ser = pd.Series(rng.randint(0, 10, 4))
>>> ser
0    6
1    3
2    7
3    4
dtype: int64
>>> df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                      columns=['A', 'B', 'C', 'D'])
>>> df
   A  B  C  D
0  6  9  2  6
1  7  4  3  7
2  7  2  5  4
```

Applying a NumPy ufunc on these objects produces pandas objects with preserved
indices:

```python
>>> np.exp(ser)
0     403.428793
1      20.085537
2    1096.633158
3      54.598150
dtype: float64

>>> np.sin(df * np.pi / 4)
          A             B         C             D
0 -1.000000  7.071068e-01  1.000000 -1.000000e+00
1 -0.707107  1.224647e-16  0.707107 -7.071068e-01
2 -0.707107  1.000000e+00 -0.707107  1.224647e-16
```

### ufunc: index alignment

#### In `Series`

Combining two data sources:

```python
>>> area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                      'California': 423967}, name='area')
>>> population = pd.Series({'California': 38332521, 'Texas': 26448193,
                            'New York': 19651127}, name='population')
>>> population / area
Alaska              NaN
California    90.413926
New York            NaN
Texas         38.018740
dtype: float64
```

The resulting array contains the union of indices of the two input arrays
which you could also find by:

```python
>>> area.index | population.index
Index(['Alaska', 'California', 'New York', 'Texas'], dtype='object')
```

For items where one or other doesn't have an entry, they get marked with
`NaN` (not a number) in the result.

This matching is implemented for any of Python's built-in arithmetic
expressions: missing values are filled in with `NaN` by default:

```python
>>> A = pd.Series([2, 4, 6], index=[0, 1, 2])
>>> B = pd.Series([1, 3, 5], index=[1, 2, 3])
>>> A + B
0    NaN
1    5.0
2    9.0
3    NaN
dtype: float64
```

Can change this fill value by using arithmetic methods instead of the
operators, e.g.

```python
>>> A.add(B, fill_value=0)
0    2.0
1    5.0
2    9.0
3    5.0
dtype: float64
```

### In `DataFrame`

Get a similar type of alignment for both columns and indices when performing
operations on `DataFrame`s.

```python
>>> A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                     columns=list('AB'))
>>> A
   A   B
0  1  11
1  5   1
>>> B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                     columns=list('BAC'))
>>> B
   B  A  C
0  4  0  9
1  5  8  0
2  9  2  6
>>> A + B
      A     B   C
0   1.0  15.0 NaN
1  13.0   6.0 NaN
2   NaN   NaN NaN
```

Indices are aligned correctly irrespective of their order in the two objects.
Indices in the result are sorted.

As for `Series`, we can use arithmetic methods with fill values instead of
ending up with `NaN`:

```python
>>> fill = A.stack().mean() # Stack rows of A first; mean is 4.5.
>>> A.add(B, fill_value=fill)
      A     B     C
0   1.0  15.0  13.5
1  13.0   6.0   4.5
2   6.5  13.5  10.5
```

Python operators and equivalent pandas methods:

* `+`  `add()`  
* `-`  `sub()`, `subtract()`  
* `*`  `mul()`, `multiply()`  
* `/`  `truediv()`, `div()`, `divide()`  
* `//` `floordiv()`  
* `%`  `mod()`  
* `**` `pow()`

### Ufuncs: operations between `DataFrame` and `Series`

Index and column alignment is similarly maintained here too.

Operations between a `DataFrame` and a `Series` are similar to operations
between a 2D and a 1D NumPy array.

Consider finding difference of a 2D array and one of its rows:

```python
>>> A = rng.randint(10, size=(3, 4))
>>> A
array([[3, 8, 2, 4],
       [2, 6, 4, 8],
       [6, 1, 3, 8]])
>>> A - A[0]
array([[ 0,  0,  0,  0],
       [-1, -2,  2,  4],
       [ 3, -7,  1,  4]])
```

This proceeds according to NumPy's broadcasting rules, the
subtraction between a 2D array and one of its rows occurs row-wise.

With pandas, the analogous operation operates row-wise too:

```python
>>> df = pd.DataFrame(A, columns=list('QRST'))
>>> df - df.iloc[0]
  Q  R  S  T
0  0  0  0  0
1 -1 -2  2  4
2  3 -7  1  4
```

To operate column-wise, use object methods and specify an `axis`:

```python
>>> df.subtract(df['R'], axis=0)
   Q  R  S  T
0 -5  0 -6 -4
1 -4  0 -2  2
2  5  0  2  7
```

(Here, we're doing the subtraction along the rows, i.e. apply the
subtraction down or along the rows for each column in turn, specifically
where there is a match of `df` row index to `df['R']` series index.  So,
really you're applying against aligned indices. "Along" rows is perhaps
a better descriptor here, since we only operate on aligned matches.)

These operations on `DataFrames` with `Series` will also align indices
between the two elements:

```python
>>> halfrow = df.iloc[0, ::2]
>>> halfrow
Q    3
S    2
Name: 0, dtype: int64
>>> df - halfrow
     Q   R    S   T
0  0.0 NaN  0.0 NaN
1 -1.0 NaN  2.0 NaN
2  3.0 NaN  1.0 NaN
```

## Handling missing data

Many datasets will have some amount of data missing, and may indicate
this in different ways.

Here, missing data is referred to as *null*, *NaN* or *NA* values.

### Trade-offs in missing data conventions

Different schemes for indicating missing data in a table or `DataFrame`.
Generally, one of two strategies: use a *mask* that globally indicates
missing values, or use a *sentinel value* that indicates a missing
entry.

A mask could be a separate Boolean array, or use one bit in the data
representation to locally indicate the null status of a value.

A sentinel value could be a data-specific convention, e.g. indicate a
missing integer with -9999 or some rare bit pattern, or a more global
convention such as indicating a missing floating-point value with NaN
(Not a Number), a special value part of the IEEE floating-point
specification.

There are trade-offs for both. A mask array requires an extra Boolean
array, which adds overhead in storage and computation. A sentinel value
reduces the range of valid values that can be represented, and may
require extra (often non-optimised) logic in CPU and GPU arithmetic.
Common special values like NaN are not available for all data types.

Because of this, different languages and systems use different
conventions.

### Missing data in pandas

The way pandas deals with this is constrained by its reliance on NumPy,
which has no built-in notion of NA values for non-floating-point data
types.

pandas could have followed R in specifying bit patterns for each
individual data type to indicate nullness, but unwieldy since NumPy
supports far more data types (R has one integer type, NumPy has
fourteen). Reserving a bit pattern for all available NumPy types would
require lots of overhead in special-casing various operations for
various types. And, for smaller data types (e.g. 8-bit integers), using
a bit as a mask reduces the range of representable values.

NumPy has support for masked arrays, which pandas could have derived
from, but overhead in storage, computation and code maintenance makes
that unattractive.

So, pandas chose sentinels for missing data, and uses two existing
Python null values, the special floating-point `NaN` value, and the
Python `None` object. This choice has side-effects, but is a good
compromise in most cases.

#### `None`: Pythonic missing data

The first sentinel value used by pandas is `None`, a Python singleton
object that is often used for missing data in Python code. Because it is
a Python object, `None` cannot be used in an arbitrary NumPy/pandas
array, but only in arrays with data type `object` (arrays of Python
objects):

```python
>>> import numpy as np
>>> import pandas as pd
>>> vals1 = np.array([1, None, 3, 4])
>>> vals1
array([1, None, 3, 4], dtype=object)
```

This `dtype=object` means that the best common type representation that
NumPy could infer for the array contents is that they are Python
objects. This kind of object array might be useful for some purposes,
but any operations on the data are done at Python level, with much more
overhead than the typically fast operations for arrays with native
types:

```python
>>> for dtype in ['object', 'int']:
        print("dtype =", dtype)
        %timeit np.arange(1E6, dtype=dtype).sum()
        print()

dtype = object
10 loops, best of 3: 78.2 ms per loop

dtype = int
100 loops, best of 3: 3.06 ms per loop
```

The use of Python objects in an array means that if you perform
aggregation like `sum()` or `min()` across an array with a `None` value,
you will usually get an error, e.g. because addition between an integer
and `None` is undefined.

#### `NaN`: missing numerical data

The other missing data representation, `NaN`, is different. It is a
special floating-point value recognised by all systems that use the
standard IEEE floating-point representation:

```python
>>> vals2 = np.array([1, np.nan, 3, 4]) 
>>> vals2.dtype
dtype('float64')
```

NumPy chose a floating-point type for this array. It supports fast
operations pushed into compiled code.

`NaN` "infects" any other object it touches. Regardless of the
operation, arithmetic with `NaN` results in another `NaN`:

```python
>>> 1 + np.nan
nan
>>> 0 * np.nan
nan
```

This means that aggregates over values are well defined (no error), but
not always useful:

```python
>>> vals2.sum(), vals2.min(), vals2.max()
(nan, nan, nan)
```

NumPy has special aggregations that ignore these missing values:

```python
>>> np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
(8.0, 1.0, 4.0)
```

`NaN` is a floating-point value: there is no equivalent NaN value for
integers, strings or other types.

#### `NaN` and `None` in pandas

Both `NaN` and `None` have their place. pandas handles these
interchangeably, converting between them where appropriate:

```python
>>> pd.Series([1, np.nan, 2, None])
0    1.0
1    NaN
2    2.0
3    NaN
dtype: float64
```

For types that don't have an available sentinel value, pandas
automatically type-casts when NA values are present. For example, if we
set a value in an integer array to `np.nan`, it automatically gets cast
to a floating-point type to accommodate the NA:

```python
>>> x = pd.Series(range(2), dtype=int)
>>> x
0    0
1    1
dtype: int64
>>> x[0] = None
>>> x
0    NaN
1    1.0
dtype: float64
```

As well as casting the integer array to floating point, pandas converts
the `None` to a `NaN` value.

When NA values are introduced:

* `floating` type, no type conversion, `np.nan` sentinel value
* `object` type, no type conversion, `None` or `np.nan` sentinel value
* `integer` type, cast to `float64`, `np.nan` sentinel value
* `boolean` type, cast to `object`, `None` or `np.nan` sentinel value

NB: string data is always stored with an `object` type in pandas.

### Operating on null values

Several useful methods for detecting, removing and replacing null
(`None` or `NaN`) values in pandas data structures.

* `isnull()`: generate a Boolean mask indicating missing values
* `notnull()`: opposite of `isnull()`
* `dropna()`: return a filtered version of the data
* `fillna()`: return a copy of the data with missing values filled or
  imputed.

#### Detecting null values

```python
>>> data = pd.Series([1, np.nan, 'hello', None])
>>> data.isnull()
0    False
1     True
2    False
3     True
dtype: bool
```

Can use a Boolean mask as an index to a `Series` or `DataFrame`:

```python
>>> data[data.notnull()]
0        1
2    hello
dtype: object
```

The `isnull()` and `notnull()` methods produce similar Boolean results
for `DataFrames`.

#### Dropping null values

As well as using the masking as above, can use `dropna()` (removes NA
values) and `fillna()` (fills in NA values).

For a `Series`, the result is straightforward:

```python
>>> data.dropna()
0        1
2    hello
dtype: object
```

For a `DataFrame`, have more options.

```python
>>> df = pd.DataFrame([[1,      np.nan, 2],
                       [2,      3,      5],
                       [np.nan, 4,      6]])
>>> df
     0    1  2
0  1.0  NaN  2
1  2.0  3.0  5
2  NaN  4.0  6
```

Can't drop single values from a `DataFrame`, only full rows or columns.

`dropna()` gives a number of options for doing so. By default, it drops all
rows where any null value is present.

```python
>>> df.dropna()
     0    1  2
1  2.0  3.0  5
```

Can also drop NA values along a different axis; `axis=1` drops columns:

```python
>>> df.dropna(axis='columns')
   2
0  2
1  5
2  6
```

But this drops good data too. You might want to drop rows or columns with all
NA values, or a majority of NA values. Can be specified by `how` or `thresh`
parameters, which allow fine control of the number of nulls to allow through.

The default is `how='any'` so any row or column (depending on `axis`) with a
null value will be dropped. Can also specify `how='all'` to drop only rows or
columns that are all null values.

```python
>>> df[3] = np.nan
>>> df
     0    1  2   3
0  1.0  NaN  2 NaN
1  2.0  3.0  5 NaN
2  NaN  4.0  6 NaN
>>> df.dropna(axis='columns', how='all')
     0    1  2
0  1.0  NaN  2
1  2.0  3.0  5
2  NaN  4.0  6
```

`thresh` lets you specify a minimum number of non-null values for the row or
column to be kept.

```python
>>> df.dropna(axis='rows', thresh=3)
     0    1  2   3
1  2.0  3.0  5 NaN
```

NB: the use of axis can be a little confusing here. "`rows`" means
`axis=0`, but usually think of `axis=0` as going down the rows for each
column. The difference here is that, from the official docs, `dropna()`
will "return object with labels on given axis omitted where alternately
any or all of the data are missing", so can think of it as going through
the *index* for that axis and dropping the labels.

Someone else has the same confusion as me in [this Stack Overflow
question](https://stackoverflow.com/questions/39290667/confused-with-the-use-of-axis-in-pandas-python).

And label is one of the axis labels, i.e. one of the identifiers used to
for accessing a value in a dimension by indexing, i.e. ultimately inside
an array. These axis labels are used to identify a position (and a
value) on that axis.

#### Filling null values

Sometimes you might want to replace NA values with a valid value instead of
dropping them. Could be a number like zero or some kind of imputation or
interpolation from the good values.

Could do this using `isnull()` as a mask, but because it is a common
operation, pandas has `fillna()` which returns a copy of the array with
the null values replaced.

```python
>>> data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
a    1.0
b    NaN
c    2.0
d    NaN
e    3.0
dtype: float64
>>> data.fillna(0) # replace NA with single value
a    1.0
b    0.0
c    2.0
d    0.0
e    3.0
dtype: float64
>>> data.fillna(method='ffill') # forward-fill
a    1.0
b    1.0
c    2.0
d    2.0
e    3.0
dtype: float64
>>> data.fillna(method='bfill') # back-fill
a    1.0
b    2.0
c    2.0
d    3.0
e    3.0
dtype: float64
```

For `DataFrames`, have similar options, but can specify the axis along which
fills take place:

```python
>>> df
     0    1  2   3
0  1.0  NaN  2 NaN
1  2.0  3.0  5 NaN
2  NaN  4.0  6 NaN
>>> df.fillna(method='ffill', axis=1)
     0    1    2    3
0  1.0  1.0  2.0  2.0
1  2.0  3.0  5.0  5.0
2  NaN  4.0  6.0  6.0
```

(So here you could use `ffill`, then `bfill` to fill all NA values.)

## Hierarchical indexing

So far focused on 1D and 2D data stored in `Series` and `DataFrame`
objects respectively. Often useful to store higher-dimensional data:
data indexed by more than one or two keys. pandas does provide `Panel`
and `Panel4D` objects to handle 3D and 4D data, but it is more common to
use hierarchical indexing (or multi-indexing) to incorporate multiple
index *levels* within a single index. Can then represent higher
dimensional data within the familiar `Series` and `DataFrame` objects.

*levels* as in multiple hierarchical levels.

### A multiply indexed `Series`

Consider representing 2D data with a 1D series. Here, consider a series
of data where each point has a string and numerical key.

#### The bad way

Use Python tuples as keys:

```python
>>> index = [('California', 2000), ('California', 2010),
             ('New York', 2000), ('New York', 2010),
             ('Texas', 2000), ('Texas', 2010)]
>>> populations = [33871648, 37253956,
                   18976457, 19378102,
                   20851820, 25145561]
>>> pop = pd.Series(populations, index=index)
>>> pop
(California, 2000)    33871648
(California, 2010)    37253956
(New York, 2000)      18976457
(New York, 2010)      19378102
(Texas, 2000)         20851820
(Texas, 2010)         25145561
dtype: int64
```

Can index or slice based on this multiple index:

```python
>>> pop[('California', 2010):('Texas', 2000)]
(California, 2010)    37253956
(New York, 2000)      18976457
(New York, 2010)      19378102
(Texas, 2000)         20851820
dtype: int64
```

But convenience ends there. You can't easily, for example, select all values
from 2010:

```python
>>> pop[[i for i in pop.index if i[1] == 2010]]
(California, 2010)    37253956
(New York, 2010)      19378102
(Texas, 2010)         25145561
dtype: int64
```

Not as clean, or efficient, as the usual pandas syntax.

#### The better way: pandas `MultiIndex`

This tuple-based indexing is essentially a rudimentary multi-index, and a
pandas `MultiIndex` gives us the types of operations we need.

```python
>>> index = pd.MultiIndex.from_tuples(index)
>>> index
MultiIndex(levels=[['California', 'New York', 'Texas'], [2000, 2010]],
           labels=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
```

This `MultiIndex` contains multiple levels of indexing: here, the state names
and years, as well as multiple labels for each data point which encode these
levels.

Confusingly, pandas documentation describes `levels` as "the unique labels for
each level", and `labels` as "integers for each level designating which label
at each location". In practice, I think that grouping the integers found inside
`labels` together (at the same index position in each array inside `labels`)
can be used to obtain the corresponding labels for that data item, e.g. `0, 0`
is California, 2000, while `2, 1` is Texas, 2010. So this is presumably how
they are encoded. So, `labels` are the indices of the `levels` for each data
item.

Another way to consider this: with a normal `Index`, have a single index
value corresponding to something (e.g. a row, column or data value), and
have an array of these index values to give the order. With a
`MultiIndex`, each position in a sequence has multiple index values, one
at one level, represented in one array, and one at another level,
represented in a different array. The indices inside `labels` map to the
level labels.

If we re-index our series with this `MultiIndex`, we see the hierarchical
representation of the data:

```python
>>> pop = pop.reindex(index)
>>> pop
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
```

The first two columns of the `Series` representation show the multiple index
values, while the third column shows the data. Some entries are missing in the
first column: for this multi-index representation, a blank line indicates the
same value as the line above.

Now we can use pandas slicing notation:

```python
>>> pop[:, 2010]
California    37253956
New York      19378102
Texas         25145561
dtype: int64
```

This gives us a singly indexed `Series` with the keys we're interested in, and
is much more convenient and efficient than the homebrew tuple multiindexing
solution above.

#### `MultiIndex` as extra dimension

We could have easily stored the same data using a `DataFrame` with index
and column labels. pandas is built with this equivalence in mind. The
`unstack()` method will convert a multiply indexed `Series` into a
conventionally indexed `DataFrame`.

```python
>>> pop_df = pop.unstack()
>>> pop_df
                2000      2010
California  33871648  37253956
New York    18976457  19378102
Texas       20851820  25145561
```

`stack()` does the opposite:

```python
>>> pop_df.stack()
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
```

So, why bother with hierarchical indexing? Because as we can use a `MultiIndex`
to represent 2D data in a 1D `Series`, we can use it to represent data of
higher dimensions (3D or more) in a `Series` or `DataFrame`.

Each extra level in a multi-index represents an extra data dimension: this
gives us much more flexibility in the types of data we can represent.

As an example, might want to add another column of demographic data for
each state at each year (e.g. population under 18 years of age). With a
`MultiIndex`, we can just add another column to the `DataFrame`:

```python
>>> pop_df = pd.DataFrame({'total': pop,
                           'under18': [9267089, 9284094,
                                       4687374, 4318033,
                                       5906301, 6879014]})
>>> pop_df
                    total  under18
California 2000  33871648  9267089
           2010  37253956  9284094
New York   2000  18976457  4687374
           2010  19378102  4318033
Texas      2000  20851820  5906301
           2010  25145561  6879014
```

In addition, all the ufuncs and other functionality work with hierarchical
indices as well.

```python
>>> f_u18 = pop_df['under18'] / pop_df['total']
>>> f_u18.unstack()
                2000      2010
California  0.273594  0.249211
New York    0.247010  0.222831
Texas       0.283251  0.273568
```

(So, this computes a `MultiIndex` `Series` for each row of the `DataFrame`,
and then unstacks the `MultiIndex` into a new `DataFrame`.)

### Methods of `MultiIndex` creation

The simplest way is to pass a list of two or more index arrays to the
constructor of a `Series` or `DataFrame`:

```python
>>> df = pd.DataFrame(np.random.rand(4, 2),
                      index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                      columns=['data1', 'data2'])
>>> df
        data1     data2
a 1  0.039466  0.075460
  2  0.211717  0.641098
b 1  0.586914  0.237623
  2  0.472170  0.130114
```

If you pass a dictionary with tuples as keys, pandas will recognise this and
use a `MultiIndex` by default:

```python
>>> data = {('California', 2000): 33871648,
            ('California', 2010): 37253956,
            ('Texas', 2000): 20851820,
            ('Texas', 2010): 25145561,
            ('New York', 2000): 18976457,
            ('New York', 2010): 19378102}
>>> pd.Series(data)
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
```

#### Explicit `MultiIndex` constructors

Sometimes useful to create a `MultiIndex` explicitly too.

For more flexibility, you can use `MultiIndex` class methods for
instantiation.

Can use arrays as before, giving the index values within each level:

```python
>>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
MultiIndex(levels=[['a', 'b'], [1, 2]],
           labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
```

Can construct from a list of tuples, giving the multiple index values of
each point:

```python
>>> pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
MultiIndex(levels=[['a', 'b'], [1, 2]],
           labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
```

Can construct from a Cartesian product of single indices (Cartesian
product: set of all ordered pairs from multiple sets):

```python
>>> pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
MultiIndex(levels=[['a', 'b'], [1, 2]],
           labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
```

Can construct the `MultiIndex` directly using its internal encoding by
passing `levels` (a list of lists containing available index values for
each level) and `labels` (a list of lists that reference these labels):

```python
>>> pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
                  labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
MultiIndex(levels=[['a', 'b'], [1, 2]],
           labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
```

A `MultiIndex` can be passed as the `index` argument when creating a
`Series` or `DataFrame`, or passed to the `reindex` method of an
existing `Series` or `DataFrame`.

#### `MultiIndex` level names

Sometimes convenient to name the levels of the `MultiIndex` to track their
meaning.

Can pass `names` argument to a `MultiIndex` constructor, or by setting `names`
attribute of the index:

```python
>>> pop.index.names = ['state', 'year']
>>> pop
state       year
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
```

#### `MultiIndex` for columns

In a `DataFrame`, the rows and columns are completely symmetric. Just as rows
can have multiple index levels, so can the columns.

```python
>>> # hierarchical indices and columns
>>> index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                       names=['year', 'visit'])
>>> columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                         names=['subject', 'type'])
>>> # mock some data
>>> data = np.round(np.random.randn(4, 6), 1)
>>> data[:, ::2] *= 10
>>> data += 37
>>> # create the DataFrame
>>> health_data = pd.DataFrame(data, index=index, columns=columns)
>>> health_data
subject      Bob       Guido         Sue      
type          HR  Temp    HR  Temp    HR  Temp
year visit                                    
2013 1      46.0  37.4  31.0  34.4  34.0  35.4
     2      27.0  37.5  45.0  36.2  56.0  36.1
2014 1      50.0  35.9  35.0  37.1  27.0  34.8
     2      32.0  35.7  38.0  36.7  39.0  37.6
```

This represents 4D data within a `DataFrame`. Can index the top-level column
by the person's name and get a `DataFrame` containing that person's data:

```python
>>> health_data['Guido']
type          HR  Temp
year visit            
2013 1      31.0  34.4
     2      45.0  36.2
2014 1      35.0  37.1
     2      38.0  36.7
```

For complicated records containing multiple labelled measurements across
multiple times for many subjects (people, countries, cities etc.), use of
hierarchical rows and columns can be very convenient.

### Indexing and slicing a `MultiIndex`

Designed to be intuitive. Helps to think about the indices as added dimensions.

#### Multiply indexed `Series`

Consider the `pop` `Series` earlier.

```python
>>> pop
state       year
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
```

Can access single elements by indexing with multiple terms:

```python
>>> pop['California', 2000]
33871648
```

`MultiIndex` also supports partial indexing: indexing just one level in the
index. This results in another `Series`, with the lower-level indices
maintained.

```python
>>> pop['California']
year
2000    33871648
2010    37253956
dtype: int64
```

Partial slicing is available as well, as long as the `MultiIndex` is sorted.
(See below.)

```python
>>> pop.loc['California':'New York']
state       year
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
dtype: int64
```

With sorted indices, can perform partial indexing on lower levels by passing
an empty slice in the first index:

```python
>>> pop[:, 2000]
state
California    33871648
New York      18976457
Texas         20851820
dtype: int64
```

Other types of indexing and selection work too, e.g. with Boolean masks:

```python
>>> pop[pop > 22000000]
state       year
California  2000    33871648
            2010    37253956
Texas       2010    25145561
dtype: int64
```

Selection based on fancy indexing also works:

```python
>>> pop[['California', 'Texas']]
state       year
California  2000    33871648
            2010    37253956
Texas       2000    20851820
            2010    25145561
dtype: int64
```

#### Multiply indexed `DataFrames`

Consider previous data:

```python
>>> health_data
subject      Bob       Guido         Sue
type          HR  Temp    HR  Temp    HR  Temp
year visit
2013 1      46.0  37.4  31.0  34.4  34.0  35.4
     2      27.0  37.5  45.0  36.2  56.0  36.1
2014 1      50.0  35.9  35.0  37.1  27.0  34.8
     2      32.0  35.7  38.0  36.7  39.0  37.6
```

Remember that columns are primary in a `DataFrame`, and syntax for multiply
indexed `Series` applies to the columns.

```python
>>> health_data['Guido', 'HR']
year  visit
2013  1        31.0
      2        45.0
2014  1        35.0
      2        38.0
Name: (Guido, HR), dtype: float64
```

As with the single-index case, can use `loc`, `iloc` and `ix` indexers.

```python
>>> health_data.iloc[:2, :2]
subject      Bob      
type          HR  Temp
year visit            
2013 1      46.0  37.4
     2      27.0  37.5
```

These provide an array-like view of the underlying 2D data, but each index in
`loc` or `iloc` can be passed a tuple of multiple indices:

```python
>>> health_data.loc[:, ('Bob', 'HR')]
year  visit
2013  1        46.0
      2        27.0
2014  1        50.0
      2        32.0
Name: (Bob, HR), dtype: float64
```

Working with slices within these index tuples is not convenient: trying to
create a slice within a tuple will lead to a syntax error:

```python
>>> health_data.loc[(:, 1), (:, 'HR')]
```

Could build this using Python's `slice()` function, but better here to use
a pandas `IndexSlice` object:

```python
>>> idx = pd.IndexSlice
>>> health_data.loc[idx[:, 1], idx[:, 'HR']]
subject      Bob Guido   Sue
type          HR    HR    HR
year visit                  
2013 1      46.0  31.0  34.0
2014 1      50.0  35.0  27.0
```

### Rearranging Multi-Indices

A key to working with multiply indexed data is knowing how to effectively
transform the data. There are a number of operations that will preserve all the
information in the dataset, but rearrange it for the purposes of various
computations.

`stack()` and `unstack()` are examples of this, but there are more too.

#### Sorted and unsorted indices

Many `MultiIndex` slicing operations fail if the index is not sorted.

As an example, here is some multiply indexed data where indices are not
lexicographically sorted:

```python
>>> index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
>>> data = pd.Series(np.random.rand(6), index=index)
>>> data.index.names = ['char', 'int']
>>> data
char  int
a     1      0.003001
      2      0.164974
c     1      0.741650
      2      0.569264
b     1      0.001693
      2      0.526226
dtype: float64
```

If we try to take a partial slice of this index, it fails:

```python
>>> try:
        data['a':'b']
    except KeyError as e:
        print(type(e))
        print(e)
<class 'KeyError'>
'Key length (1) was greater than MultiIndex lexsort depth (0)'
```

The error message isn't clear, but this is a result of the `MultiIndex` not
being sorted. Partial slices and other similar operations require
lexicographically sorted levels. pandas provides a number of convenience
routines to do this sorting, e.g. `sort_index()` and `sortlevel()` methods of
`DataFrame`s.

```python
>>> data = data.sort_index()
>>> data
char  int
a     1      0.003001
      2      0.164974
b     1      0.001693
      2      0.526226
c     1      0.741650
      2      0.569264
dtype: float64
```

With the index sorted, partial slicing works:

```python
>>> data['a':'b']
char  int
a     1      0.003001
      2      0.164974
b     1      0.001693
      2      0.526226
dtype: float64
```

#### Stacking and unstacking indices

Can convert a dataset from a stacked multi-index to a 2D representation,
optionally specifying the level to use (which becomes the added dimension in
the `DataFrame`).

```python
>>> pop
state       year
California  2000    33871648
            2010    37253956
New York    2000    18976457
            2010    19378102
Texas       2000    20851820
            2010    25145561
dtype: int64
>>> pop.unstack(level=0)
state  California  New York     Texas
year                                 
2000     33871648  18976457  20851820
2010     37253956  19378102  25145561
>>> pop.unstack(level=1)
year            2000      2010
state                         
California  33871648  37253956
New York    18976457  19378102
Texas       20851820  25145561
```

The opposite of `unstack()` is `stack()` which can be used to recover the
original series.

#### Index setting and resetting

Another way to rearrange hierarchical data is to turn the index labels into
columns with the `reset_index()` method. For clarity, we can specify the name
of the data for the column representation:

```python
>>> pop_flat = pop.reset_index(name='population')
>>> pop_flat
        state  year  population
0  California  2000    33871648
1  California  2010    37253956
2    New York  2000    18976457
3    New York  2010    19378102
4       Texas  2000    20851820
5       Texas  2010    25145561
```

Often, the raw data looks like that, and it's useful to build a `MultiIndex`
from the column values, which can be achieved via `set_index()` method of a
`DataFrame`, returning a multiply indexed `DataFrame`:

```python
>>> pop_flat.set_index(['state', 'year'])
                 population
state      year            
California 2000    33871648
           2010    37253956
New York   2000    18976457
           2010    19378102
Texas      2000    20851820
           2010    25145561
```

### Data aggregations on multi-indices

Already seen that pandas has aggregation methods, e.g. `mean()`, `sum()` and
`max()`. For hierarchically indexed data, these can be passed a `level`
parameter to control which subset of the data the aggregate is computed on.

```python
>>> health_data
subject      Bob       Guido         Sue      
type          HR  Temp    HR  Temp    HR  Temp
year visit                                    
2013 1      46.0  37.4  31.0  34.4  34.0  35.4
     2      27.0  37.5  45.0  36.2  56.0  36.1
2014 1      50.0  35.9  35.0  37.1  27.0  34.8
     2      32.0  35.7  38.0  36.7  39.0  37.6
```

If we want to average the measurements in the two visits each year, we can
name the index level we want to explore, the year:

```python
>>> data_mean = health_data.mean(level='year')
>>> data_mean
subject   Bob        Guido         Sue       
type       HR   Temp    HR  Temp    HR   Temp
year                                         
2013     36.5  37.45  38.0  35.3  45.0  35.75
2014     41.0  35.80  36.5  36.9  33.0  36.20
```

Can take mean among levels on columns by using the `axis` keyword:

```python
>>> data_mean.mean(axis=1, level='type') # NB: using the previous output.
type         HR       Temp
year                      
2013  39.833333  36.166667
2014  36.833333  36.300000
```

So, can find the average heart rate and temperature among all subjects in all
visits each year in just two lines. This syntax is a shortcut to the `GroupBy`
functionality (see later).

### Panel data

pandas has `Panel` and `Panel4D` objects that can be thought of as 3D and 4D
generalisations respectively of `Series` and `DataFrame` structures.

However, multi-indexing is often more useful and conceptually simpler
representation of higher dimensional data. Also, panel data is a dense data
representation, while multi-indexing is sparse, and the dense representation
can become inefficient for real-world datasets. For occasional specialised
applications, these panel objects can be useful, but are not discussed more
here.

## Combining datasets: Concat and Append

Combining different data sources often of interest. Can be
straightforward concatenation of different datasets, to more complicated
database-style joins and merges that handle overlaps between the
datasets. `Series` and `DataFrame`s are built with this type of
operation in mind, and pandas includes functions and methods to make
this sort of wrangling fast and straightforward.

Here, look at `pd.concat` for concatenating `Series` and `DataFrame`s.

Use this convenience function:

```python
>>> def make_df(cols, ind):
        """Quickly make a DataFrame"""
        data = {c: [str(c) + str(i) for i in ind]
                for c in cols}
        return pd.DataFrame(data, ind)
>>> # example DataFrame
>>> make_df('ABC', range(3))
    A   B   C
0  A0  B0  C0
1  A1  B1  C1
2  A2  B2  C2
```

### Concatenation of NumPy arrays

Concatenation of `Series` and `DataFrame`s is very similar to concatenating
NumPy arrays, and those can be concatenated via the `np.concatenate` function.
(See Chapter 2.)

```python
>>> x = [1, 2, 3]
>>> y = [4, 5, 6]
>>> z = [7, 8, 9]
>>> np.concatenate([x, y, z])
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

The first argument is a list or tuple of arrays to concatenate. Can also
specify `axis` along which the result will be concatenated:

```python
>>> x = [[1, 2],
         [3, 4]]
>>> np.concatenate([x, x], axis=1)
array([[1, 2, 1, 2],
       [3, 4, 3, 4]])
```

### Simple concatenation with `pd.concat`

pandas' `pd.concat` function has a similar syntax to `np.concatenate` but
contains other options too:

```python
# Signature in Pandas v0.18
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)
```

Can use `pd.concat()` for simple concatenation of `Series` or `DataFrame`
objects, just like `np.concatenate()` can concatenate NumPy arrays.

```python
>>> ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
>>> ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
>>> pd.concat([ser1, ser2])
1    A
2    B
3    C
4    D
5    E
6    F
dtype: object
```

Works for higher dimensional objects too:

```python
>>> df1 = make_df('AB', [1, 2])
>>> df2 = make_df('AB', [3, 4])
>>> df1
    A   B
1  A1  B1
2  A2  B2
>>> df2
    A   B
3  A3  B3
4  A4  B4
>>> pd.concat([df1, df2])
    A   B
1  A1  B1
2  A2  B2
3  A3  B3
4  A4  B4
```

By default, the concatenation takes place row-wise in the `DataFrame`
(`axis=0`). Like `np.concatenate`, `pd.concat` allows specification of an
axis along which concatenation will take place:

```python
>>> df3 = make_df('AB', [0, 1])
>>> df3
    A   B
0  A0  B0
1  A1  B1
>>> df4
    C   D
0  C0  D0
1  C1  D1
>>> df4 = make_df('CD', [0, 1])
>>> pd.concat([df3, df4], axis='columns')
    A   B   C   D
0  A0  B0  C0  D0
1  A1  B1  C1  D1
```

`axis='columns'` is the same as `axis=1`.

concat: think of as go down axis and adding each column on for along axis=0, or
go along columns and add row for each column, for along axis=1, or can
just think about as moving to the end of that axis and sticking the
whole extra item there.

### Duplicate indices

Unlike `np.concatenate`, pandas' `pd.concat` preserves indices, even if the
result has duplicate indices!

```python
>>> x = make_df('AB', [0, 1])
>>> x
    A   B
0  A0  B0
1  A1  B1
>>> y = make_df('AB', [2, 3])
>>> y
    A   B
2  A2  B2
3  A3  B3
>>> y.index = x.index  # make duplicate indices!
>>> pd.concat([x, y])
    A   B
0  A0  B0
1  A1  B1
0  A2  B2
1  A3  B3
```

Duplicate indices are valid within `DataFrame`s but often undesirable.

`pd.concat()` gives us a few ways to handle it.

#### Catching repeats as an error

To verify that the indices in the result of `pd.concat()` do not overlap,
you can specify the `verify_integrity` flag:

```python
>>> try:
        pd.concat([x, y], verify_integrity=True)
    except ValueError as e:
        print("ValueError:", e)
ValueError: Indexes have overlapping values: [0, 1]
```

#### Ignoring the index

Sometimes the index doesn't matter and you may want to ignore it. Can do so
using the `ignore_index` flag:

```python
>>> pd.concat([x, y], ignore_index=True)
    A   B
0  A0  B0
1  A1  B1
2  A2  B2
3  A3  B3
```

#### Adding `MultiIndex` keys

Can specify `keys` to specify a label for the data source, giving a
hierarchically indexed series containing the data:

```python
>>> pd.concat([x, y], keys=['x', 'y'])
      A   B
x 0  A0  B0
  1  A1  B1
y 0  A2  B2
  1  A3  B3
```

Can then transform this as required using approaches discussed above.

### Concatenation with joins

In examples above, mainly concatenating `DataFrame`s with shared column
names. In practice, data from different sources might have different sets
of column names, and `pd.concat` offers several options in this case.

For example, concatenating two `DataFrame`s with some, but not all, columns
in common:

```python
>>> df5 = make_df('ABC', [1, 2])
>>> df5
    A   B   C
1  A1  B1  C1
2  A2  B2  C2
>>> df6 = make_df('BCD', [3, 4])
>>> df6
    B   C   D
3  B3  C3  D3
4  B4  C4  D4
>>> pd.concat([df5, df6])
     A   B   C    D
1   A1  B1  C1  NaN
2   A2  B2  C2  NaN
3  NaN  B3  C3   D3
4  NaN  B4  C4   D4
```

By default, entries for which no data is available are filled with NA values.
To change this, specify one of several options for `join` and `join_axes`
parameters of `pd.concat()`. By default, the join is a union of the input
columns (`join='outer'`), but can change it to an intersection of the columns
using `join='inner'`:

```python
>>> pd.concat([df5, df6], join='inner')
    B   C
1  B1  C1
2  B2  C2
3  B3  C3
4  B4  C4
```

Another option is specify the index of the remaining columns using the
`join_axes` argument, which takes a list of index objects:

```python
>>> pd.concat([df5, df6], join_axes=[df5.columns])
     A   B   C
1   A1  B1  C1
2   A2  B2  C2
3  NaN  B3  C3
4  NaN  B4  C4
```

### The append() method

Because array concatenation is common, `Series` and `DataFrame` objects have
an `append` method that does the same thing in fewer characters, e.g.
instead of `pd.concat([df1, df2])`, can call `df1.append(df2)`:

```python
>>> df1
    A   B
1  A1  B1
2  A2  B2
>>> df2
    A   B
3  A3  B3
4  A4  B4
>>> df1.append(df2)
    A   B
1  A1  B1
2  A2  B2
3  A3  B3
4  A4  B4
```

Unlike `append()` and `extend()` of Python lists, `append()` method here
creates a new object with the combined data. Not very efficient because it
involves creation of a new index and data buffer. For multiple `append`s, it
is better to do them at once, building a list of `DataFrame`s and passing them
to `pd.concat()`.

## Combining datasets: merge and join

pandas offers high performance, in-memory join and merge operations.

Main interface for this functionality is `pd.merge()`.

### Relational algebra

The behaviour implemented in `pd.merge()` is a subset of what is known
as *relational algebra*, which is a formal set of rules for manipulating
relational data, and forms the conceptual foundation of operations
available in most databases. The strength of the relational algebra
approach is that it proposes several primitive operations, which become
the building blocks of more complicated operations on any dataset.

pandas implements several of these fundamental building blocks in the
`pd.merge()` function and the related `join()` method of `Series` and
`DataFrames`. These let you efficiently link data from different
sources.

### Categories of joins

`pd.merge()` implements a number of types of joins: the *one-to-one*,
*many-to-one*, and *many-to-many* joins. All three are accessed via
identical calls to `pd.merge()`; the type of join performed depends on
the form of the input data.

#### One-to-one joins

Perhaps the simplest type of merge is the one-to-one join, which is
similar to the column-wise concatenation seen above.

Consider this example:

```python
>>> df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                        'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
>>> df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                        'hire_date': [2004, 2008, 2012, 2014]})
>>> df1
  employee        group
0      Bob   Accounting
1     Jake  Engineering
2     Lisa  Engineering
3      Sue           HR
>>> df2
  employee  hire_date
0     Lisa       2004
1      Bob       2008
2     Jake       2012
3      Sue       2014
```

To combine these into a single `DataFrame`, we use `pd.merge()`:

```python
>>> df3 = pd.merge(df1, df2)
>>> df3
  employee        group  hire_date
0      Bob   Accounting       2008
1     Jake  Engineering       2012
2     Lisa  Engineering       2004
3      Sue           HR       2014
```

`pd.merge()` recognises that each `DataFrame` has an employee column, and
automatically joins using this column as a key. The result of this merge is a
new `DataFrame` combining the information from the two inputs. The order of
entries isn't necessarily maintained: `df1` and `df2`  have different ordering
of the employee column, but `pd.merge()` correctly accounts for this.
Additionally, the merge in general discards the index, except in the special
case of merges by index (see `left_index` and `right_index` below).

NB: I think the use of key is a little less strict than in a database context.
Here it is more like columns containing shared values, used to combine
different datasets. This is what database keys are often used for, but database
keys may have additional constraints. These are more like foreign keys: i.e.
you use them to link to keys in another table, but they are not necessarily
unique in either table.

#### Many-to-one joins

Many-to-one joins are joins in which one of the two key columns contains
duplicate entries. The resulting `DataFrame` will preserve those duplicate
entries as appropriate.

```python
>>> df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                        'supervisor': ['Carly', 'Guido', 'Steve']})
>>> df4
         group supervisor
0   Accounting      Carly
1  Engineering      Guido
2           HR      Steve
>>> pd.merge(df3, df4)
  employee        group  hire_date supervisor
0      Bob   Accounting       2008      Carly
1     Jake  Engineering       2012      Guido
2     Lisa  Engineering       2004      Guido
3      Sue           HR       2014      Steve
```

The resulting `DataFrame` has an additional supervisor column, where the
information is repeated as required by the inputs.

#### Many-to-many joins

Many-to-many joins can be confusing conceptually, but are well defined.
If the key column in both the left and right array contains duplicates,
then the result is a many-to-many merge.

```python
>>> df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                                  'Engineering', 'Engineering', 'HR', 'HR'],
                        'skills': ['math', 'spreadsheets', 'coding', 'linux',
                                  'spreadsheets', 'organization']})
>>> df5
         group        skills
0   Accounting          math
1   Accounting  spreadsheets
2  Engineering        coding
3  Engineering         linux
4           HR  spreadsheets
5           HR  organization
>>> pd.merge(df1, df5)
  employee        group        skills
0      Bob   Accounting          math
1      Bob   Accounting  spreadsheets
2     Jake  Engineering        coding
3     Jake  Engineering         linux
4     Lisa  Engineering        coding
5     Lisa  Engineering         linux
6      Sue           HR  spreadsheets
7      Sue           HR  organization
```

These three joins can be used to implement a wide array of functionality.
There are further options provided by `pd.merge()` that enable customisation
of how the join operations work.

### Specification of the merge key

#### The `on` keyword

You can specify the name of the key column, using `on`, which takes a column
name or a list of column names.

```python
>>> pd.merge(df1, df2, on='employee')
  employee        group  hire_date
0      Bob   Accounting       2008
1     Jake  Engineering       2012
2     Lisa  Engineering       2004
3      Sue           HR       2014
```

This option works only if both the left and right `DataFrame`s have the
specified column names.

#### The `left_on` and `right_on` keywords

May want to merge two datasets with different column names. For example,
the employee name is labelled as name instead of employee.

```python
>>> df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                        'salary': [70000, 80000, 120000, 90000]})
>>> df3
   name  salary
0   Bob   70000
1  Jake   80000
2  Lisa  120000
3   Sue   90000
>>> pd.merge(df1, df3, left_on="employee", right_on="name")
  employee        group  name  salary
0      Bob   Accounting   Bob   70000
1     Jake  Engineering  Jake   80000
2     Lisa  Engineering  Lisa  120000
3      Sue           HR   Sue   90000
```

The result has a redundant column that we can drop if desired, e.g.

```python
>>> pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
  employee        group  salary
0      Bob   Accounting   70000
1     Jake  Engineering   80000
2     Lisa  Engineering  120000
3      Sue           HR   90000
```

#### The `left_index` and `right_index` keywords

Sometimes, rather than merging on a column, you would like to merge on an
index.

```python
>>> df1a = df1.set_index('employee')
>>> df2a = df2.set_index('employee')
>>> df1a
                group
employee             
Bob        Accounting
Jake      Engineering
Lisa      Engineering
Sue                HR
>>> df2a
          hire_date
employee           
Lisa           2004
Bob            2008
Jake           2012
Sue            2014
>>> pd.merge(df1a, df2a, left_index=True, right_index=True)
                group  hire_date
employee                        
Bob        Accounting       2008
Jake      Engineering       2012
Lisa      Engineering       2004
Sue                HR       2014
```

For convenience, `DataFrame`s implement the `join()` method, which is a merge
that defaults to joining on indices:

```python
>>> df1a.join(df2a)
                group  hire_date
employee                        
Bob        Accounting       2008
Jake      Engineering       2012
Lisa      Engineering       2004
Sue                HR       2014
```

If you'd like to mix indices and columns, you can combine `left_index` with
`right_on` (or `right_index` with `left_on`) to get the desired behaviour:

```python
>>> pd.merge(df1a, df3, left_index=True, right_on='name')
         group  name  salary
0   Accounting   Bob   70000
1  Engineering  Jake   80000
2  Engineering  Lisa  120000
3           HR   Sue   90000
```

All of these options work with multiple indices and/or multiple columns. Also
see the pandas documentation for more.

### Specifying set arithmetic for joins

We have not yet specified the set arithmetic for the join. This arises when
a value in one key column is not present in the other.

```python
>>> df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                        'food': ['fish', 'beans', 'bread']},
                        columns=['name', 'food'])
>>> df6
    name   food
0  Peter   fish
1   Paul  beans
2   Mary  bread
>>> df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                        'drink': ['wine', 'beer']},
                        columns=['name', 'drink'])
>>> df7
     name drink
0    Mary  wine
1  Joseph  beer
>>> pd.merge(df6, df7)
   name   food drink
0  Mary  bread  wine
```

Here we merged two datasets with a single name entry in common. By default,
the result contains the intersection of the two sets of inputs: an inner join.
We can specify this explicitly using the `how` keyword, which defaults to
`inner`.

```python
>>> pd.merge(df6, df7, how='inner')
   name   food drink
0  Mary  bread  wine
```

Other options for `how` are `outer`, `left` and `right`. An outer join returns
a join over the union of the input columns, and fills in all missing values
with NAs:

```python
>>> pd.merge(df6, df7, how='outer')
     name   food drink
0   Peter   fish   NaN
1    Paul  beans   NaN
2    Mary  bread  wine
3  Joseph    NaN  beer
```

The *left join* and *right join* return joins over the left entries and right
entries, respectively.

```python
>>> pd.merge(df6, df7, how='left')
    name   food drink
0  Peter   fish   NaN
1   Paul  beans   NaN
2   Mary  bread  wine
```

Output rows correspond to the entries in the left input. Using `how='right'`
works in a similar manner.

Can apply all of these options to any of the preceding join types.

### Overlapping column names: the `suffixes` keyword

May end up where your two input `DataFrame`s have conflicting column names.

```python
>>> df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                        'rank': [1, 2, 3, 4]})
>>> df8
   name  rank
0   Bob     1
1  Jake     2
2  Lisa     3
3   Sue     4
>>> df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                        'rank': [3, 1, 4, 2]})
>>> df9
   name  rank
0   Bob     3
1  Jake     1
2  Lisa     4
3   Sue     2
>>> pd.merge(df8, df9, on="name")
   name  rank_x  rank_y
0   Bob       1       3
1  Jake       2       1
2  Lisa       3       4
3   Sue       4       2
```

Because the output would have conflicting column names, the merge function
automatically appends a suffix `_x` or `_y` to make the output columns unique.
Can change these defaults with the `suffixes` keyword:

```python
>>> pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])')
   name  rank_L  rank_R
0   Bob       1       3
1  Jake       2       1
2  Lisa       3       4
3   Sue       4       2
```

These suffixes work in any of the possible join patterns and work if there
are multiple overlapping columns.

### Notes from worked example

Using `.isnull().any()` is a useful check to see if merged data contains
missing matches.

## Aggregation and grouping

An essential piece of analysis of large data is efficient summarisation:
computing aggregations like `sum()`, `mean()`, `median()`, `min()`, and
`max()`, in which a single number gives insight into the nature of a
dataset.

### Planets data

These examples will use Seaborn's planets dataset, which gives
information on planets discovered around other stars (exoplanets).

```python
>>> import seaborn as sns
>>> planets = sns.load_dataset('planets')
>>> planets.shape
(1035, 6)
>>> planets.head()
            method  number  orbital_period   mass  distance  year
0  Radial Velocity       1         269.300   7.10     77.40  2006
1  Radial Velocity       1         874.774   2.21     56.95  2008
2  Radial Velocity       1         763.000   2.60     19.84  2011
3  Radial Velocity       1         326.030  19.40    110.62  2007
4  Radial Velocity       1         516.220  10.50    119.47  2009
```

### Simple aggregation in pandas

Previously, we explored some of the data aggregation available for NumPy
arrays.

As with a 1D NumPy array, aggregates of a pandas `Series` return a single
value:

```python
>>> rng = np.random.RandomState(42)
>>> ser = pd.Series(rng.rand(5))
>>> ser
0    0.374540
1    0.950714
2    0.731994
3    0.598658
4    0.156019
dtype: float64
>>> ser.sum()
2.8119254917081569
>>> ser.mean()
0.56238509834163142
```

For a `DataFrame`, by default the aggregates return results within each
column.

```python
>>> df = pd.DataFrame({'A': rng.rand(5),
                       'B': rng.rand(5)})
>>> df
          A         B
0  0.155995  0.020584
1  0.058084  0.969910
2  0.866176  0.832443
3  0.601115  0.212339
4  0.708073  0.181825
>>> df.mean()
A    0.477888
B    0.443420
dtype: float64
```

By specifying `axis`, you can aggregrate within each row:

```python
>>> df.mean(axis='columns')
0    0.088290
1    0.513997
2    0.849309
3    0.406727
4    0.444949
dtype: float64
```

pandas `Series` and `DataFrames` include the common NumPy aggregates. There is
also a convenience method `describe()` that computes several common aggregates
for each column and returns the result. This can be useful to start
understanding the overall properties of a dataset.

```python
>>> planets.dropna().describe() # drop missing values
          number  orbital_period        mass    distance         year
count  498.00000      498.000000  498.000000  498.000000   498.000000
mean     1.73494      835.778671    2.509320   52.068213  2007.377510
std      1.17572     1469.128259    3.636274   46.596041     4.167284
min      1.00000        1.328300    0.003600    1.350000  1989.000000
25%      1.00000       38.272250    0.212500   24.497500  2005.000000
50%      1.00000      357.000000    1.245000   39.940000  2009.000000
75%      2.00000      999.600000    2.867500   59.332500  2011.000000
max      6.00000    17337.500000   25.000000  354.000000  2014.000000
```

For example, can see from `year` column that although exoplanets were
discovered as far back as 1989, half of them were not discovered until 2010 or
later, largely thanks to Kepler space telescope mission.

Built-in pandas aggregations, that are all methods of `DataFrame` and `Series`
objects:

* `count()`, total number of items;
* `first()`, `last()`, first and last item;
* `mean()`, `median(), mean and median;
* `min()`, `max()`, minimum and maximum;
* `std()`, `var()`, standard deviation and variance;
* `mad()`, mean absolute deviation;
* `prod()`, product of all items;
* `sum()`, sum of all items.

To go deeper into the data, simple aggregates are often not enough. The next
level of data summarisation is the `groupby` operation, which allows quick
and efficient computation of aggregates on subsets of data.

### GroupBy: Split, Apply, Combine

Simple aggregations can give a flavour of a dataset, but often we would prefer
to aggregate conditionally on some label or index. This is implemented in
the `groupby` operation. The name "group by" comes from SQL, but perhaps more
helpful to think of it as: split, apply, combine.

#### Split, apply, combine

An example of a split-apply-combine operation, where the "apply" is a
summation aggregation is shown below:

```
Input          Split         Apply         Combine     
                             (sum)
                                          
               key data  ->  key data  ->  
               A   1         A   5
key data  ->   A   4
A   1                                      key data
B   2          key data  ->  key data  ->  A   5
C   3     ->   B   2         B   7         B   7
A   4          B   5                       C   9
B   5
C   6     ->   key data  ->  key data  ->
               C   3         C   9
               C   6
```

What `groupby` does:

* The *split* step breaks up and groups a `DataFrame` depending on the value of 
  the specified key.
* The *apply* step computes some function, usually an aggregate, transformation
  or filtering within the individual groups.
* The *combine* step merges the results of these operations into an output
  array.

(Key here just seems to be a column that identifies some pieces of data as
belonging to some group, and the value that you're using as the basis for some
operation.)

While this could be done manually using some combination of the masking,
aggregation, and merging commands covered earlier, an important realisation is
that the intermediate splits do not need to be explicitly instantiated.

The GroupBy can (often) do this in a single pass over the data, updating the
sum, mean, count, min or other aggregate for each group along the way. The
power of the GroupBy is that it abstracts away these steps: the user doesn't
need to think about how the computation is done, only about the operation as
a whole.

Calculating the example above using pandas:

```python
>>> df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                       'data': range(6)}, columns=['key', 'data'])
>>> df
  key  data
0   A     0
1   B     1
2   C     2
3   A     3
4   B     4
5   C     5
```

The most basic split-apply-combine operation can be computed with the
`groupby()` method of `DataFrame`s, passing the name of the desired key column:

```python
>>> df.groupby('key')
<pandas.core.groupby.DataFrameGroupBy object at 0x7f98dbc8d320>
```

Notice that a `DataFrameGroupBy` object is returned, not a set of `DataFrame`s.
This object is where the magic is: you can think of it as a special view of the
`DataFrame` (like a view of a database), which is poised to dig into the groups
but does not actual computation until the aggregation is applied. This
"lazy evaluation" approach means that common aggregates can be implemented
very efficiently in a way almost transparent to the user.

To produce a result, we can apply an aggregate to this `DataFrameGroupBy`
object, which will perform the appropriate apply/combine steps to produce the
desired result:

```python
>>> df.groupby('key').sum()
     data
key      
A       3
B       5
C       7
```

The `sum()` method is just one possibility. Virtually any common pandas or
NumPy aggregation function, as well as virtually any valid `DataFrame`
operation can be applied.

#### The `GroupBy` object

The `GroupBy` object is a flexible abstraction. In many ways, you can treat it
as if it's a collection of `DataFrame`s and it does the difficult things behind
the scenes.

Perhaps the most important operations made available by a `GroupBy` are 
*aggregate*, *filter*, *transform* and *apply*. These will be discussed further
below. First, other functionality will be introduced.

##### Column indexing

The `GroupBy` object supports column indexing in the same way as the
`DataFrame` and returns a modified `GroupBy` object.

```python
>>> planets.groupby('method')
<pandas.core.groupby.DataFrameGroupBy object at 0x7f98dbc8db38>
>>> planets.groupby('method')['orbital_period']
<pandas.core.groupby.SeriesGroupBy object at 0x7f98dbc8d8d0>
```

Here we've selected a particular `Series` group from the original `DataFrame`
group by reference to its column name. As with the `GroupBy` object, no
computation is done until we call some aggregate on the object:

```python
>>> planets.groupby('method')['orbital_period'].median()
method
Astrometry                         631.180000
Eclipse Timing Variations         4343.500000
Imaging                          27500.000000
Microlensing                      3300.000000
Orbital Brightness Modulation        0.342887
Pulsar Timing                       66.541900
Pulsation Timing Variations       1170.000000
Radial Velocity                    360.200000
Transit                              5.714932
Transit Timing Variations           57.011000
Name: orbital_period, dtype: float64
```

This gives an idea of the general scale of orbital periods that each method is
sensitive to.

##### Iteration over groups

The `GroupBy` object supports direct iteration over the groups, returning
each group as a `Series` or `DataFrame`:

```python
>>> for (method, group) in planets.groupby('method'):
        print("{0:30s} shape={1}".format(method, group.shape))
Astrometry                     shape=(2, 6)
Eclipse Timing Variations      shape=(9, 6)
Imaging                        shape=(38, 6)
Microlensing                   shape=(23, 6)
Orbital Brightness Modulation  shape=(3, 6)
Pulsar Timing                  shape=(5, 6)
Pulsation Timing Variations    shape=(1, 6)
Radial Velocity                shape=(553, 6)
Transit                        shape=(397, 6)
Transit Timing Variations      shape=(4, 6)
```

This can be useful for doing certain things manually, though it is often
much faster to use the built-in `apply` functionality, discussed below.

##### Dispatch methods

Through some Python class magic, any method not explicitly implemented by
the `GroupBy` object will be passed through and called on the groups, whether
they are `DataFrame` or `Series` objects. For example, you can use the
`describe()` method of `DataFrame`s to perform a set of aggregations that
describe each group in the data.

```python
>>> planets.groupby('method')['year'].describe()
                               count         mean       std     min      25%  \
method                                                                         
Astrometry                       2.0  2011.500000  2.121320  2010.0  2010.75   
Eclipse Timing Variations        9.0  2010.000000  1.414214  2008.0  2009.00   
Imaging                         38.0  2009.131579  2.781901  2004.0  2008.00   
Microlensing                    23.0  2009.782609  2.859697  2004.0  2008.00   
Orbital Brightness Modulation    3.0  2011.666667  1.154701  2011.0  2011.00   
Pulsar Timing                    5.0  1998.400000  8.384510  1992.0  1992.00   
Pulsation Timing Variations      1.0  2007.000000       NaN  2007.0  2007.00   
Radial Velocity                553.0  2007.518987  4.249052  1989.0  2005.00   
Transit                        397.0  2011.236776  2.077867  2002.0  2010.00   
Transit Timing Variations        4.0  2012.500000  1.290994  2011.0  2011.75   

                                  50%      75%     max  
method                                                  
Astrometry                     2011.5  2012.25  2013.0  
Eclipse Timing Variations      2010.0  2011.00  2012.0  
Imaging                        2009.0  2011.00  2013.0  
Microlensing                   2010.0  2012.00  2013.0  
Orbital Brightness Modulation  2011.0  2012.00  2013.0  
Pulsar Timing                  1994.0  2003.00  2011.0  
Pulsation Timing Variations    2007.0  2007.00  2007.0  
Radial Velocity                2009.0  2011.00  2014.0  
Transit                        2012.0  2013.00  2014.0  
Transit Timing Variations      2012.5  2013.25  2014.0
```

This table helps us to better understand the data. For example, the vast
majority of planets have been discovered by radial velocity or transits methods.
The newest methods seem to be orbital brightness modulation and transit
timing variations, which didn't discover a new planet until 2011.

This is one example of the utility of dispatch methods. They are applied to
each individual group, and the results are then combined within `GroupBy` and
returned. Again, any valid `DataFrame`/`Series` method can be used on the
corresponding `GroupBy` object, which makes for very flexible and powerful
operations.

#### Aggregate, filter, transform, apply

The above discussion focused on aggregation for the combine operation, but
there are more options available. In particular, `GroupBy` objects have
`aggregate()`, `filter()`, `transform()` and `apply()` methods that efficiently
implement a variety of useful operations before combining the grouped data.

```python
>>> rng = np.random.RandomState(0)
>>> df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                       'data1': range(6),
                       'data2': rng.randint(0, 10, 6)},
                       columns = ['key', 'data1', 'data2'])
>>> df
  key  data1  data2
0   A      0      5
1   B      1      0
2   C      2      3
3   A      3      3
4   B      4      7
5   C      5      9
```

##### Aggregation

The `aggregate()` method offers more flexibility than `sum()`, `median()` and
other aggregations shown above. It can take a string, a function or a list
thereof, and compute all the aggregates at once.

```python
>>> df.groupby('key').aggregate(['min', np.median, max])
    data1            data2           
      min median max   min median max
key                                  
A       0    1.5   3     3    4.0   5
B       1    2.5   4     0    3.5   7
C       2    3.5   5     3    6.0   9
```

Another useful pattern is to pass a dictionary mapping column names to
operations to be applied on that column.

```python
>>> df.groupby('key').aggregate({'data1': 'min',
                                 'data2': 'max'})
     data1  data2
key              
A        0      5
B        1      7
C        2      9
```

##### Filtering

A filtering operation allows you to drop data based on the group properties.
For example, we might want to keep all groups where the standard deviation
is larger than some value:

```python
>>> def filter_func(x):
        return x['data2'].std() > 4
>>> df.groupby('key').std()
       data1     data2
key                   
A    2.12132  1.414214
B    2.12132  4.949747
C    2.12132  4.242641
>>> df.groupby('key').filter(filter_func)
  key  data1  data2
1   B      1      0
2   C      2      3
4   B      4      7
5   C      5      9
```

The filter function should return a Boolean value specifying whether the group
passes the filtering. Group A does not have a standard deviation greater than
4, so it is dropped.

##### Transformation

While aggregation returns a reduced version of the data, transformation can
return some transformed version of the full data to recombine. For such a
transformation, the output is the same shape as the input.

A common example is centring the data by subtracting the group-wise mean:

```python
>>> df.groupby('key').transform(lambda x: x - x.mean())
   data1  data2
0   -1.5    1.0
1   -1.5   -3.5
2   -1.5   -3.0
3    1.5   -1.0
4    1.5    3.5
5    1.5    3.0
```

##### The `apply()` method

The `apply()` method lets you apply an arbitrary function to the group results.

The function should take a `DataFrame` and return either a pandas object, or a
scalar. The combine operation will be tailored to the type of object returned.

```python
>>> def norm_by_data2(x):
        # x is a DataFrame of group values
	x['data1'] /= x['data2'].sum()
	return x
>>> df.groupby('key').apply(norm_by_data2)
  key     data1  data2
0   A  0.000000      5
1   B  0.142857      0
2   C  0.166667      3
3   A  0.375000      3
4   B  0.571429      7
5   C  0.416667      9
```

(Here, we group by key, then compute the value of `data1` based on the value
of the sum of the `data2` for that key. `data2` in the output is then left
unchanged.)

#### Specifying the split key

In the previous examples, we split the `DataFrame` on a single column name.
There are other options to define the groups.

##### A list, array, series or index providing the grouping keys

The key can be any series or list with a length matching that of the
`DataFrame`. For example:

```python
>>> L = [0, 1, 0, 1, 2, 0]
>>> df
  key  data1  data2
0   A      0      5
1   B      1      0
2   C      2      3
3   A      3      3
4   B      4      7
5   C      5      9
>>> df.groupby(L).sum()
   data1  data2
0      7     17
1      4      3
2      4      7
```

(Here, the groups are assigned by number in L corresponding to each row of
`df`, e.g. row `0` in the output corresponds to rows with indices `0`, `2`, `5`
in the input.)

This means there is a more verbose way of accomplishing `df.groupby('key')`
from before: `df.groupby(df['key']).sum()`.

##### A dictionary or series mapping index to group

Another method is to provide a dictionary that maps index values to the group
keys:

```python
>>> df2 = df.set_index('key')
>>> mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
>>> df2.groupby(mapping).sum()
           data1  data2
consonant     12     19
vowel          3      8
```

##### Any Python function

Similar to mapping, you can pass any Python function that will input the index
value and output the group:

```python
>>> df2.groupby(str.lower).mean()
   data1  data2
a    1.5    4.0
b    2.5    3.5
c    3.5    6.0
```

##### A list of valid keys

Further, any of the preceding key choices can be combined to group on a
multi-index:

```python
>>> df2.groupby([str.lower, mapping]).mean()
             data1  data2
a vowel        1.5    4.0
b consonant    2.5    3.5
c consonant    3.5    6.0
```

#### Grouping example

We can put these together and count discovered planets by method and by
decade:

```python
>>> decade = 10 * (planets['year'] // 10)
>>> decade = decade.astype(str) + 's'
>>> decade.name = 'decade'
>>> planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
decade                         1980s  1990s  2000s  2010s
method                                                   
Astrometry                       0.0    0.0    0.0    2.0
Eclipse Timing Variations        0.0    0.0    5.0   10.0
Imaging                          0.0    0.0   29.0   21.0
Microlensing                     0.0    0.0   12.0   15.0
Orbital Brightness Modulation    0.0    0.0    0.0    5.0
Pulsar Timing                    0.0    9.0    1.0    1.0
Pulsation Timing Variations      0.0    0.0    1.0    0.0
Radial Velocity                  1.0   52.0  475.0  424.0
Transit                          0.0    0.0   64.0  712.0
Transit Timing Variations        0.0    0.0    0.0    9.0
```

(Here, `decade` is a `Series` with one entry per planet, that is used as a
second grouping key after `method`. So, this gives a `MultiIndex` to do the
`sum()` over, so for each method, we find the sum for each decade. `unstack()
then changes the `decade` part of the `MultiIndex` to columns.)

## Pivot tables

We have seen how the `GroupBy` abstraction lets us explore relationships
within a dataset. A *pivot table* is a similar operation that is
commonly seen in spreadsheets and other programs that operate on tabular
data. The pivot table takes simple column-wise data as input, and groups
the entries into a two-dimensional table that provides a
multidimensional summarisation of the data. The difference between pivot
tables and `GroupBy` can sometimes cause confusion; it can help to think
of pivot tables as a *multidimensional* version of `GroupBy`
aggregation. That is, you split-apply-combine, but both the split and
the combine happen across not a 1D index, but a 2D grid.

### Motivating pivot tables

We'll use the *Titanic* passenger database, available through Seaborn,
for these examples. It contains a lot of information on each passenger.

```python
>>> import numpy as np
>>> import pandas as pd
>>> import seaborn as sns
>>> titanic = sns.load_dataset('titanic')
>>> titanic.head()
   survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
0         0       3    male  22.0      1      0   7.2500        S  Third   
1         1       1  female  38.0      1      0  71.2833        C  First   
2         1       3  female  26.0      0      0   7.9250        S  Third   
3         1       1  female  35.0      1      0  53.1000        S  First   
4         0       3    male  35.0      0      0   8.0500        S  Third   

     who  adult_male deck  embark_town alive  alone  
0    man        True  NaN  Southampton    no  False  
1  woman       False    C    Cherbourg   yes  False  
2  woman       False  NaN  Southampton   yes   True  
3  woman       False    C  Southampton   yes  False  
4    man        True  NaN  Southampton    no   True
```

### Pivot tables by hand

To learn more about this data, we might group according to gender, survival
status or some combination thereof. You could try a `GroupBy` operation,
for example, to look at survival rate by gender.

```python
>>> titanic.groupby('sex')[['survived']].mean()
        survived
sex             
female  0.742038
male    0.188908
```

Overall, three of every four women on board survived, but only one in five men
did.

We might want to go further and look at survival by sex and class. Using
`GroupBy`, we might proceed like this: *group* by class and gender, *select*
survival, *apply* a mean aggregate, *combine* the resulting groups, and then
*unstack* the hierarchical index to reveal the hidden multidimensionality.

```python
>>> titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
class      First    Second     Third
sex                                 
female  0.968085  0.921053  0.500000
male    0.368852  0.157407  0.135447
```

This gives us the answer we want, but the code is starting to look confusing.
However, pandas includes a convenience routine, `pivot_table`, which handles
this type of multi-dimensional aggregation.

### Pivot table syntax

Here is the equivalent operation using the `pivot_table` method of
`DataFrame`s:

```python
>>> titanic.pivot_table('survived', index='sex', columns='class')
class      First    Second     Third
sex                                 
female  0.968085  0.921053  0.500000
male    0.368852  0.157407  0.135447
```

This is much more readable.

#### Multi-level pivot tables

As in the `GroupBy`, pivot table grouping can be specified with multiple levels
and via a number of options. For example, we might be interested in age as
a third dimension. Here, we can bin the age using `pd.cut()`.

```python
>>> age = pd.cut(titanic['age'], [0, 18, 80])
>>> titanic.pivot_table('survived', ['sex', age], 'class')
class               First    Second     Third
sex    age                                   
female (0, 18]   0.909091  1.000000  0.511628
       (18, 80]  0.972973  0.900000  0.423729
male   (0, 18]   0.800000  0.600000  0.215686
       (18, 80]  0.375000  0.071429  0.133663
```

We can do the same with the columns too. Let's add info on the fare using
`pd.qcut()` to compute quartiles.

```python
>>> fare = pd.qcut(titanic['fare'], 2)
>>> titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
fare            (-0.001, 14.454]                     (14.454, 512.329]  \
class                      First    Second     Third             First   
sex    age                                                               
female (0, 18]               NaN  1.000000  0.714286          0.909091   
       (18, 80]              NaN  0.880000  0.444444          0.972973   
male   (0, 18]               NaN  0.000000  0.260870          0.800000   
       (18, 80]              0.0  0.098039  0.125000          0.391304   

fare                                 
class              Second     Third  
sex    age                           
female (0, 18]   1.000000  0.318182  
       (18, 80]  0.914286  0.391304  
male   (0, 18]   0.818182  0.178571  
       (18, 80]  0.030303  0.192308 
```

The result is a four-dimensional aggregation with hierarchical indices, shown
in a grid demonstrating the relationship between the values.

#### Additional pivot table options

The full call signature of the `pivot_table` method is:

```python
# call signature as of Pandas 0.18
DataFrame.pivot_table(data, values=None, index=None, columns=None,
                      aggfunc='mean', fill_value=None, margins=False,
                      dropna=True, margins_name='All')
```

We've already seen examples of the first three arguments; in this section,
we'll look at the remaining ones.

Two of the options, `fill_value` and `dropna`, have to do with missing data,
and are straightforward.

`aggfunc` controls what type of aggregation is applies, which is a mean by
default. As in `GroupBy`, the aggregation specification can be a string
representing one of several common choices (e.g. `'sum'`, `'mean'`, `'count'`,
`'min'`, `'max'` etc.) or a function that implements an aggregation, (e.g.
`np.sum()`, `np.min()`, `sum()` etc.). Additionally, it can be specified as
a dictionary mapping a column to any of the above desired options:

```python
>>> titanic.pivot_table(index='sex', columns='class',
                        aggfunc={'survived':sum, 'fare':'mean'})
              fare                       survived             
class        First     Second      Third    First Second Third
sex                                                           
female  106.125798  21.970121  16.118810       91     70    72
male     67.226127  19.741782  12.661633       45     17    47
```

The `values` keyword is not needed here: it is determined automatically when
specifying a mapping for `aggfunc`.

At times, it's useful to compute totals along each grouping. This can be done
via the `margins` keyword:

```python
>>> titanic.pivot_table('survived', index='sex', columns='class', margins=True)
class      First    Second     Third       All
sex                                           
female  0.968085  0.921053  0.500000  0.742038
male    0.368852  0.157407  0.135447  0.188908
All     0.629630  0.472826  0.242363  0.383838
```

This give us information about the class-agnostic survival rate by gender, the
gender-agnostic survival rate by class, and the overall survival rate. The
margin label can be specified with the `margins_name` keyword, which defaults
to `"All"`.

## Vectorised string operations

One strength of Python is its ease in handling and manipulating strings. pandas
builds on this and provides a comprehensive set of vectorised string operations
that become an essential tool when cleaning real-world data.

(NB: A vectorised operation is one that is applied to an entire array, instead
of individual elements.)

### Introducing pandas string operations

We saw previously how tools like NumPy and pandas generalise arithmetic
operations so that we can easily and quickly perform the same operation on
many array elements, such as in this example:

```python
>>> import numpy as np
>>> x = np.array([2, 3, 5, 7, 11, 13])
>>> x * 2
array([ 4,  6, 10, 14, 22, 26])
```

This vectorisation of operations simplifies the syntax of operations on arrays
of data: we no longer have to worry about the size of shape of the array, but
just about what operation we want done. For arrays of strings, NumPy does not
provide such simple access, and you're stuck using a more verbose loop syntax:

```python
>>> data = ['peter', 'Paul', 'MARY', 'gUIDO']
>>> [s.capitalize() for s in data]
['Peter', 'Paul', 'Mary', 'Guido']
```

This can work with some data, but will break if there are any missing values.

```python
>>> data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
>>> [s.capitalize() for s in data]
```

pandas includes features to address this need for vectorised string operations
and for correctly handling missing data via the `str` attribute of pandas
`Series` and `Index` objects containing strings.

Suppose we create a pandas `Series` with this data:

```python
>>> import pandas as pd
>>> names = pd.Series(data)
>>> names
0    peter
1     Paul
2     None
3     MARY
4    gUIDO
dtype: object 
```

A single method call capitalises all entries and skips missing values:

```python
>>> names.str.capitalize()
0    Peter
1     Paul
2     None
3     Mary
4    Guido
dtype: object
```

### Tables of pandas string methods

If you are familiar with Pyhon string manipulation, most pandas string syntax
is intuitive enough to list available methods.

The examples below use the following `Series` of names:

```python
>>> monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                       'Eric Idle', 'Terry Jones', 'Michael Palin'])
```

#### Methods similar to Python string methods

Here are a list of pandas vectorised `str` methods that mirror Python string
methods:

```
* len()
* lower()
* translate()
* islower()
* ljust()
* upper()
* startswith()
* isupper()
* rjust()
* find()
* endswith()
* isnumeric()
* center()
* rfind()
* isalnum()
* isdecimal()
* zfill()
* index()
* isalpha()
* split()
* strip()
* rindex()
* isdigit()
* rsplit()
* rstrip()
* capitalize()
* isspace()
* partition()
* lstrip()
* swapcase()
* istitle()
* rpartition()
```

These have various return values. Some, e.g. `lower()` return a `Series` of
strings:

```python
>>> monte.str.lower()
0    graham chapman
1       john cleese
2     terry gilliam
3         eric idle
4       terry jones
5     michael palin
dtype: object
```

Others return numbers:

```python
>>> monte.str.len()
0    14
1    11
2    13
3     9
4    11
5    13
dtype: int64
```

Or Boolean values:

```python
>>> monte.str.startswith('T')
0    False
1    False
2     True
3    False
4     True
5    False
dtype: bool
```

Others return lists or compound values for each element:

```python
>>> monte.str.split()
0    [Graham, Chapman]
1       [John, Cleese]
2     [Terry, Gilliam]
3         [Eric, Idle]
4       [Terry, Jones]
5     [Michael, Palin]
dtype: object
```

#### Methods using regular expressions

There are several methods that accept regular expressions to examine the
content of each string element, and follow some of the API conventions of
Python's built-in `re` module:

* `match()` Call `re.match()` on each element, returning a boolean.
* `extract()` Call re.match() on each element, returning matched groups as strings.
* `findall()` Call re.findall() on each element.
* `replace()` Replace occurrences of pattern with some other string.
* `contains()` Call re.search() on each element, returning a boolean.
* `count()` Count occurrences of pattern.
* `split()` Equivalent to str.split(), but accepts regexps.
* `rsplit()` Equivalent to str.rsplit(), but accepts regexps.

With these, you can do a wide range of interesting operations. For example,
we can extract the first name from each by asking for a contiguous group of
characters at the beginning of each element:

```python
>>> monte.str.extract('([A-Za-z]+)', expand=False)
0     Graham
1       John
2      Terry
3       Eric
4      Terry
5    Michael
dtype: object
```

Or do something more complicated, like finding all names that start and end
with a consonant, making use of the start-of-string (`^`) and end-of-string
(`$`) regular expression characters:

```python
>>> monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
0    [Graham Chapman]
1                  []
2     [Terry Gilliam]
3                  []
4       [Terry Jones]
5     [Michael Palin]
dtype: object
```

The ability to concisely apply regular expressions across `Series` or
`DataFrame` entries opens up many possibilities for analysis and cleaning of
data.

#### Miscellaneous methods

There are miscellaneous methods that enable other convenient operations:

* `get()` Index each element.
* `slice()` Slice each element.
* `slice_replace()` Replace slice in each element with passed value.
* `cat()` Concatenate strings.
* `repeat()` Repeat values.
* `normalize()` Return Unicode form of string.
* `pad()` Add whitespace to left, right, or both sides of strings.
* `wrap()` Split long strings into lines with length less than a given width.
* `join()` Join strings in each element of the `Series` with passed separator.
* `get_dummies()` Extract dummy variables as a `DataFrame`.

##### Vectorized item access and slicing

The `get()` and `slice()` operations, in particular, enable vectorized element
access from each array. For example, we can get a slice of the first three
characters of each array using `str.slice(0, 3)`. This is also available
through Python's normal indexing syntax: `df.str.slice(0, 3)` is equivalent to
`df.str[0:3]`:

```python
>>> monte.str[0:3]
0    Gra
1    Joh
2    Ter
3    Eri
4    Ter
5    Mic
dtype: object
```

Indexing via `df.str.get(i)` and `df.str[i]` is likewise similar.

These `get()` and `slice()` methods also let you access elements of arrays
returned by `split()`. For example, to extract the last name of each entry, we
can combine `split()` and `get()`:

```python
>>> monte.str.split().str.get(-1)
0    Chapman
1     Cleese
2    Gilliam
3       Idle
4      Jones
5      Palin
dtype: object
```

##### Indicator variables

The `get_dummies()` method may also require explanation. This is useful when
your data has a column containing some sort of coded indicator. For example, we
might have a dataset that contains information in the form of codes, such as
A="born in America," B="born in the United Kingdom," C="likes cheese," D="likes
spam":

```python
>>> full_monte = pd.DataFrame({'name': monte,
                               'info': ['B|C|D', 'B|D', 'A|C',
                                        'B|D', 'B|C', 'B|C|D']})
>>> full_monte
    info            name
0  B|C|D  Graham Chapman
1    B|D     John Cleese
2    A|C   Terry Gilliam
3    B|D       Eric Idle
4    B|C     Terry Jones
5  B|C|D   Michael Palin
```

`get_dummies()` lets you split out these indicator variables into a
`DataFrame`:

```python
>>> full_monte['info'].str.get_dummies('|')
   A  B  C  D
0  0  1  1  1
1  0  1  0  1
2  1  0  1  0
3  0  1  0  1
4  0  1  1  0
5  0  1  1  1
```

## Working with time series

pandas was developed in the context of financial modelling; it contains an
extensive set of tools for working with dates, times and time-indexed data.

Date and time data comes in a few forms:

* *Time stamps* reference particular moments in time (e.g. July 4th, 2015
  at 7:00 AM).
* *Time intervals* and *periods* reference a length of time between a
  particular beginning and end point; for example, the year 2015. Periods
  usually reference a special case of time intervals in which each interval
  is of uniform length and does not overlap (e.g. 24 hour long periods
  comprising days).

  (The difference between these and time deltas is the specific start and end.)
* *Time deltas* or *durations* reference an exact length of time (e.g. a
  duration of 22.56 seconds).

### Dates and times in Python

Python itself has available representations of dates, times, deltas, and
timespans. While the time series tools provided by pandas tend to be most
useful for data science applications, it is useful to see their relationship to
other packages used in Python.

#### Native Python dates and times: `datetime` and `dateutil`

Python's basic objects for working with dates and times are in the `datetime`
module. Along with the third-party `dateutil` module, you can use it to perform
a host of useful functionalities on dates and times.

You can manually build a date using the `datetime` type:

```python
>>> from datetime import datetime
>>> datetime(year=2015, month=7, day=4)
datetime.datetime(2015, 7, 4, 0, 0)
```

Or, using `dateutil`, you can parse dates from a variety of string formats:

```python
>>> from dateutil import parser
>>> date = parser.parse("4th of July, 2015")
>>> date
datetime.datetime(2015, 7, 4, 0, 0)
```

Once you have a `datetime` object, you can do things like print the day of the
week:

```python
>>> date.strftime('%A')
'Saturday'
```

A related package to be aware of is `pytz` which contains tools for working
with timezones, which are often problematic.

The power of `datetime` and `dateutil` lie in their flexibility and easy
syntax: you can use these objects and their methods to easily perform almost
any operation you may be interested in. Where they break down is when you wish
to work with large arrays of dates and times. Just as lists of Python numerical
variables are suboptimal compared with NumPy-style typed numerical arrays,
lists of Python datetime objects are suboptimal compared to typed arrays of
encoded dates.

#### Typed arrays of times: NumPy's `datetime64`

The weaknesses of Python's datetime format inspired the NumPy developers to add
a set of native time series data type to NumPy. The `datetime64` dtype encodes
dates as 64-bit integers, and allows arrays of dates to be represented very
compactly. The `datetime64` requires a very specific input format:

```python
>>> import numpy as np
>>> date = np.array('2015-07-04', dtype=np.datetime64)
>>> date # NB: 0-dimensional array.
array(datetime.date(2015, 7, 4), dtype='datetime64[D]')
```

Once we have this date formatted, we can quickly do vectorised operations on
it:

```python
>>> date + np.arange(12)
array(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
       '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
       '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'], dtype='datetime64[D]')
```

Because of the uniform type in NumPy `datetime64` arrays, this type of
operation can be accomplished much more quickly than if we were working
directly with Python's `datetime` objects, especially as arrays get large.

One detail of `datetime64` and `timedelta64` objects is that they are built on
a *fundamental time unit*. Because the `datetime64` object is limited to 64-bit
precision, the range of encodable times is 2^64 times this fundamental unit. In
other words, `datetime64` imposes a trade-off between time resolution and
maximum time span.

For example, if you want a time resolution of one nanosecond, you only have
enough information to encode a range of 2^64 nanoseconds, or just under 600
years. NumPy will infer the desired unit from the input, e.g. here is a
day-based datetime:

```python
>>> np.datetime64('2015-07-04')
numpy.datetime64('2015-07-04')
```

and a minute-based datetime:

```python
>>> np.datetime64('2015-07-04 12:00')
numpy.datetime64('2015-07-04T12:00')
```

NB: the time zone is automatically set to the local time on the computer
running the code.

You can force any desired fundamental unit using a format code, e.g. forcing a
nanosecond-based time:

```python
>>> np.datetime64('2015-07-04 12:59:59.50', 'ns')
numpy.datetime64('2015-07-04T12:59:59.500000000')
```

See the NumPy documentation. For real world data, `datetime64[ns]` is a useful
default as it can encode a useful range of modern dates (1678 AD to 2262 AD
absolute, or ±292 years relative) with a suitably find precision.

NB: `datetime64` does address some deficiencies of Python's `datetime` type,
but lacks many of the convenient methods and functions provided by both
`datetime` and `dateutil`.

#### Dates and times in pandas: best of both worlds

pandas builds upon all these tools to provide a `Timestamp` object. This
combines the ease of use of `datetime` and `dateutil` with the efficient
storage and vectorised interface of `datetime64`. From a group of these
`Timestamp`s, pandas can construct a `DatetimeIndex` that can be used to index
data in a `Series` or `DataFrame`; we'll see many examples of this below.

For example, we can use repeat the demonstration from above using pandas:

```python
>>> import pandas as pd
>>> date = pd.to_datetime("4th of July, 2015")
>>> date
Timestamp('2015-07-04 00:00:00')
>>> date.strftime('%A')
'Saturday'
>>> date + pd.to_timedelta(np.arange(12), 'D') # Vectorised operation.
DatetimeIndex(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
               '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
               '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'],
              dtype='datetime64[ns]', freq=None)
```

(The last step gives a `TimedeltaIndex` which then gets combined with the
`date`.)

### pandas time series: indexing by time

Where the pandas time series tools become useful is when you begin to index
data by timestamps. For example, we can construct a `Series` object that has
time indexed data:

```python
>>> index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                              '2015-07-04', '2015-08-04'])
>>> data = pd.Series([0, 1, 2, 3], index=index)
>>> data
2014-07-04    0
2014-08-04    1
2015-07-04    2
2015-08-04    3
dtype: int64
```

Now that we have the data in a `Series`, we can make use of any of the `Series`
indexing patterns we discussed previously, passing values that can be coerced
into dates:

```python
>>> data['2014-07-04':'2015-07-04']
2014-07-04    0
2014-08-04    1
2015-07-04    2
dtype: int64
```

There are additional special date-only indexing operations, such as passing a
year to obtain a slice of all data from that year:

```python
>>> data['2015']
2015-07-04    2
2015-08-04    3
dtype: int64
```

### pandas time series data structures

The fundamental pandas data structures for working with time series data:

* For *time stamps*, pandas provides the `Timestamp` type. As mentioned
  previously, it is essentially a replacement for Python's native
  `datetime`, but is based on the efficient `numpy.datetime64` data type.
  The associated index structure is `DatetimeIndex`.
* For *time periods*, pandas provides the `Period` type. This encodes a
  fixed-frequency interval based on `numpy.datetime64`. The associated index
  structure is `PeriodIndex`.
* For *time deltas* or *durations*, pandas provides the `Timedelta` type.
  `Timedelta` is a more efficient replacement for Python's `datetime.timedelta`
  type, and is based on `numpy.timedelta64`. The associated index structure is
  `TimedeltaIndex`.

The most fundamental of these are the `Timestamp` and `DatetimeIndex` objects.
While these class objects can be invoked directly, it is more common to use the
`pd.to_datetime()` function, which can parse a wide variety of formats. Passing
a single date to `pd.to_datetime()` yields a `Timestamp`; passing a series of
dates by default yields a `DatetimeIndex`:

```python
>>> dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                           '2015-Jul-6', '07-07-2015', '20150708'])
>>> dates
DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-06', '2015-07-07',
               '2015-07-08'],
              dtype='datetime64[ns]', freq=None)
```

Any `DatetimeIndex` can be converted to a `PeriodIndex` with the `to_period()`
function with the addition of a frequency code; here we'll use `'D'` to
indicate daily frequency:

```python
>>> dates.to_period('D')
PeriodIndex(['2015-07-03', '2015-07-04', '2015-07-06', '2015-07-07',
             '2015-07-08'],
            dtype='int64', freq='D')
```

A `TimedeltaIndex` is created, for example, when a date is subtracted from
another:

```python
>>> dates - dates[0]
TimedeltaIndex(['0 days', '1 days', '3 days', '4 days', '5 days'], dtype='timedelta64[ns]', freq=None)
```

#### Regular sequences: `pd.date_range()`

To make the creation of regular date sequences more convenient, pandas offers a
few functions for this purpose: `pd.date_range()` for timestamps,
`pd.period_range()` for periods, `pd.timedelta_range()` for time deltas. We've
seen that Python's `range()` and NumPy's `np.arange()` turn a start point,
end point and optional step size into a sequence. Similarly, `pd.date_range()`
accepts a start date, end date and an optional frequency code to create a
regular sequence of dates. By default, the frequency is one day.

```python
>>> dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
                           '2015-Jul-6', '07-07-2015', '20150708'])
>>> dates
DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-06', '2015-07-07',
               '2015-07-08'],
              dtype='datetime64[ns]', freq=None)
```

Alternatively, the date range can be specified not with a start and end point,
but with a start point and number of periods:

```python
>>> pd.date_range('2015-07-03', periods=8)
DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
               '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
              dtype='datetime64[ns]', freq='D')
```

The spacing can be modified by altering the `freq` argument, which defaults to
`D`. For example, for hourly timestamps:

```python
>>> pd.date_range('2015-07-03', periods=8, freq='H')
DatetimeIndex(['2015-07-03 00:00:00', '2015-07-03 01:00:00',
               '2015-07-03 02:00:00', '2015-07-03 03:00:00',
               '2015-07-03 04:00:00', '2015-07-03 05:00:00',
               '2015-07-03 06:00:00', '2015-07-03 07:00:00'],
              dtype='datetime64[ns]', freq='H')
```

To create regular sequences of `Period` or `Timedelta` values, the very similar
`pd.period_range()` and `pd.timedelta_range()` functions are useful. Here are
some monthly periods:

```python
>>> pd.period_range('2015-07', periods=8, freq='M')
PeriodIndex(['2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
             '2016-01', '2016-02'],
            dtype='int64', freq='M')
```

And a sequence of durations increasing by an hour:

```python
>>> pd.timedelta_range(0, periods=10, freq='H')
TimedeltaIndex(['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00',
                '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00'],
               dtype='timedelta64[ns]', freq='H')
```

All of these require an understanding of pandas frequency codes.

### Frequencies and offsets

Fundamental to these pandas time series tools is the concept of a frequency or
data offset. Just as we saw the `D` (day) and `H` (hour) codes, we can use such
codes to specify any desired frequency spacing. The following are the main
codes available:

`D` Calendar day  
`B` Business day  
`W` Weekly  
`M` Month end  
`BM` Business month end  
`Q` Quarter end  
`BQ` Business quarter end  
`A` Year end  
`BA` Business year end  
`H` Hours  
`BH` Business hours  
`T` Minutes  
`S` Seconds  
`L` Milliseonds  
`U` Microseconds  
`N` Nanoseconds  

The monthly, quarterly and annual frequencies are all marked at the end of the
specified period. By adding an `S` suffix to any of these, they instead will be
marked at the beginning:

`MS` Month start  
`BMS` Business month start
`QS` Quarter start  
`BQS` Business quarter start
`AS` Year start  
`BAS` Business year start

Additionally, you can change the month used to mark any quarterly or annual
code by adding a three-letter month code as a suffix:

* `Q-JAN`, `BQ-FEB`, `QS-MAR`, `BQS-APR`, etc.  
* `A-JAN`, `BA-FEB`, `AS-MAR`, `BAS-APR`, etc.

In the same way, the split point of the weekly frequency can be modified by
adding a three-letter weekday code:

* `W-SUN`, `W-MON`, `W-TUE`, `W-WED`, etc.

On top of this, codes can be combined with numbers to specify other
frequencies. For example, for a frequency of 2 hours 30 minutes, we can combine
the hour (`H`) and minute (`T`) codes as follows:

```python
>>> pd.timedelta_range(0, periods=9, freq="2H30T")
TimedeltaIndex(['00:00:00', '02:30:00', '05:00:00', '07:30:00', '10:00:00',
                '12:30:00', '15:00:00', '17:30:00', '20:00:00'],
               dtype='timedelta64[ns]', freq='150T')
```

All of these short codes refer to specific instances of pandas time series
offsets, found in the `pd.tseries.offsets` module. For example, we can create
a business day offset directly as follows:

```python
>>> from pandas.tseries.offsets import BDay
>>> pd.date_range('2015-07-01', periods=5, freq=BDay())
DatetimeIndex(['2015-07-01', '2015-07-02', '2015-07-03', '2015-07-06',
               '2015-07-07'],
              dtype='datetime64[ns]', freq='B')
```

### Resampling, shifting and windowing

#### Resampling and converting frequencies

Commonly need to resample time series data at a higher or lower frequency. This
can be done with the `resample()` method, or the simpler `asfreq()` method.
`resample()` is fundamentally a *data aggregation*, while `asfreq()` is
fundamentally a *data selection*.

For example, in the worked through case in the book, `resample` reports
the average of the previous year, while `asfreq` reports the value at the end
of the year.

For upsampling, `resample()` and `asfreq()` are largely equivalent, though
`resample()` has many more options available. In this case, the default for
both methods is to leave the upsampled points empty, that is, filled with NA
values. Just as with `pd.fillna()`, `asfreq()` accepts a `method` argument to
specify how values are imputed, e.g. `bfill` or `ffill`.

In the book's example, stock data is filled in at weekends with `bfill` or
`ffill`, while left blank when using just `asfreq` with no method.

#### Time shifting

pandas has two closely related methods for computing time shifts: `shift()` and
`tshift()`. `shift()` shifts the data, while `tshift()` shifts the index. In
both cases, the shift is specified in multiples of the frequency (which
defaults to days).

This can be useful when computing differences over time, e.g. use `tshift()` to
find data from some number of days ago by shifting the index, and match to the
current data.

#### Rolling windows

Rolling statistics can be computed via the `rolling()` attribute of `Series`
and `DataFrame` objects, returning a view similar to `groupby`. This rolling
view makes a number of aggregation operations available by default, e.g.
`rolling(365, center=True).mean()`, `rolling(365, center=True).std()`.

As with group by operations, can use `aggregate()` and `apply()` for custom
rolling computations.


## High-performance pandas: `eval()` and `query()`

The power of the PyData stack is based on the ability of NumPy and
pandas to push basic operations into C code via an intuitive syntax,
e.g. vectorized/broadcasted operations in NumPy, and grouping-type
operations in pandas. While these abstractions are efficient and
effective for many common use cases, they often rely on the creation of
temporary intermediate objects, which can cause undue overhead in
computational time and memory use.

pandas also includes tools that allows you to directly access C-speed
operations without costly allocation of intermediate arrays. These are
the `eval()` and `query()` functions, which rely on the Numexpr package.

### Motivating `eval()` and `query()`: compound expressions

We've previously seen that NumPy and pandas support fast vectorised
operations; for example, when adding the elements of two arrays:

```python
>>> import numpy as np
>>> rng = np.random.RandomState(42)
>>> x = rng.rand(1000000)
>>> y = rng.rand(1000000)
>>> %timeit x + y
100 loops, best of 3: 3.39 ms per loop
```

This is much faster than doing the addition via a Python loop or
comprehension:

```python
>>> %timeit np.fromiter((xi + yi for xi, yi in zip(x, y)), dtype=x.dtype,
                        count=len(x))
1 loop, best of 3: 266 ms per loop
```

But this abstraction can become less efficient when computing compound
expressions. For example, consider the following expression:

```python
>>> mask = (x > 0.5) & (y < 0.5)
```

Because NumPy evaluates each subexpression, this is roughly equivalent
to:

```python
>>> tmp1 = (x > 0.5)
>>> tmp2 = (y < 0.5)
>>> mask = tmp1 & tmp2
```

Every intermediate step is explicitly allocated in memory. If the arrays
are very large, this can lead to significant memory and computational
overhead. The Numexpr library gives you the ability to compute this type
of compound expression element by element, without the need to allocate
full intermediate arrays. The library accepts a string giving the
NumPy-style expression you wish to compute:

```python
>>> import numexpr
>>> mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
>>> np.allclose(mask, mask_numexpr)
True
```

For NumPy, using Numexpr can be more efficient. Likewise, for pandas,
using `eval()` and `query()` can be more efficient too; these depend on
Numexpr.

(NB: the pandas documentation actually suggests only using `pd.eval()` when
working with large `DataFrame`s, more than 10,000 rows.)

### `pandas.eval()` for efficient operations

The `eval()` function in pandas uses string operations to efficiently
compute operations using `DataFrame`s. For example:

```python
>>> import pandas as pd
>>> nrows, ncols = 100000, 100
>>> rng = np.random.RandomState(42)
>>> df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                          for i in range(4))
```

To compute the sum of all four `DataFrame`s using the typical pandas
approach, we can write the sum directly:

```python
>>> %timeit df1 + df2 + df3 + df4
10 loops, best of 3: 87.1 ms per loop
```

The same result can be computed via `pd.eval()` by constructing the expression
as a string:

```python
>>> %timeit pd.eval('df1 + df2 + df3 + df4')
10 loops, best of 3: 42.2 ms per loop
```

The `eval()` version is about 50% faster (and uses much less memory), while
giving the same result:

```python
>>> np.allclose(df1 + df2 + df3 + df4,
                pd.eval('df1 + df2 + df3 + df4'))
True
```

#### Operations supported by `pd.eval()`

`pd.eval()` supports a wide range of operations. We'll demonstrate using
the following `DataFrame`s:

```python
>>> df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3)))
                               for i in range(5))
```

##### Arithmetic operators

`pd.eval()` supports all arithmetic operators.

```python
>>> result1 = -df1 * df2 / (df3 + df4) - df5
>>> result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
>>> np.allclose(result1, result2)
True
```

##### Comparison operators

`pd.eval()` supports all comparison operators, including chained expressions:

```python
>>> result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
>>> result2 = pd.eval('df1 < df2 <= df3 != df4')
>>> np.allclose(result1, result2)
True
```

##### Bitwise operators

`pd.eval()` supports bitwise operators:

```python
>>> result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
>>> result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
>>> np.allclose(result1, result2)
True
```

It also supports Boolean operators:

```python
>>> result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
>>> np.allclose(result1, result3)
True
```

##### Object attributes and indices

`pd.eval()` supports access to object attributes via the `obj.attr` syntax, and
indexes via the `obj[index]` syntax:

```python
>>> result1 = df2.T[0] + df3.iloc[1]
>>> result2 = pd.eval('df2.T[0] + df3.iloc[1]')
>>> np.allclose(result1, result2)
True
```

##### Other operations

Other operations such as functional calls, conditional statements, loops and
more involved constructs are not implemented in `pd.eval()`. To execute these
more complicated expressions, you can use the Numexpr library itself.

### `DataFrame.eval()` for column-wise operations

Just as pandas has `pd.eval()`, `DataFrame`s have an `eval()` method that works
in similar ways. The benefit of this method is that columns can be referred to
by name.

```python
>>> df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
>>> df.head()
          A         B         C
0  0.615875  0.525167  0.047354
1  0.330858  0.412879  0.441564
2  0.689047  0.559068  0.230350
3  0.290486  0.695479  0.852587
4  0.424280  0.534344  0.245216
```

Using `pd.eval()` as above, we can compute expressions with the three columns
like this:

```python
>>> result1 = (df['A'] + df['B']) / (df['C'] - 1)
>>> result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
>>> np.allclose(result1, result2)
True
```

The `DataFrame.eval()` method allows more concise evaluation of expressions
with the columns:

```python
>>> result3 = df.eval('(A + B) / (C - 1)')
>>> np.allclose(result1, result3)
```

Column names are treated as variables in the evaluated expression.

#### Assignment in `DataFrame.eval()`

We can also use `DataFrame.eval()` to assign to any column.

```python
>>> df.eval('D = (A + B) / C', inplace=True)
>>> df.head()
          A         B         C          D
0  0.615875  0.525167  0.047354  24.095868
1  0.330858  0.412879  0.441564   1.684325
2  0.689047  0.559068  0.230350   5.418335
3  0.290486  0.695479  0.852587   1.156439
4  0.424280  0.534344  0.245216   3.909296
```

Existing columns can be modified too:

```python
>>> df.eval('D = (A - B) / C', inplace=True)
>>> df.head()
          A         B         C         D
0  0.615875  0.525167  0.047354  1.915527
1  0.330858  0.412879  0.441564 -0.185752
2  0.689047  0.559068  0.230350  0.564268
3  0.290486  0.695479  0.852587 -0.475016
4  0.424280  0.534344  0.245216 -0.448844
```

#### Local variables in `DataFrame.eval()`

The `DataFrame.eval()` method supports an additional syntax that lets it work
with local Python variables. Consider the following:

```python
>>> column_mean = df.mean(1)
>>> result1 = df['A'] + column_mean
>>> result2 = df.eval('A + @column_mean')
>>> np.allclose(result1, result2)
True
```

The `@` character here marks a variable name, and lets you efficiently evaluate
expressions involving the two "namespaces": that of columns, and that of Python
objects. `@` is only supported by `DataFrame.eval()`, not `pandas.eval()`
because `pandas.eval()` only has access to the one (Python) namespace.

### `DataFrame.query()` method

The `DataFrame` has another method based on evaluated strings: `query()`.

Consider:

```python
>>> result1 = df[(df.A < 0.5) & (df.B < 0.5)]
>>> result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
>>> np.allclose(result1, result2)
True
``` 

This, again, is an expression involving columns of the `DataFrame`. It cannot
be expressed directly using `DataFrame.eval()`.

(NB: I'm not sure about this; can you not use `@df`?)

Instead, you can use `query()`:

```python
>>> result2 = df.query('A < 0.5 and B < 0.5')
>>> np.allclose(result1, result2)
True
```

As well as being more efficient, compared to the masking expression, this is
easier to read and understand.

```python
>>> Cmean = df['C'].mean()
>>> result1 = df[(df.A < Cmean) & (df.B < Cmean)]
>>> result2 = df.query('A < @Cmean and B < @Cmean')
>>> np.allclose(result1, result2)
True
```

### Performance: when to use these functions

There are two considerations: computation time and memory use.

Memory use is the most predictable aspect. Every compound expression involving
NumPy arrays or pandas `DataFrame`s will result in implicit creation of
temporary arrays.

If the size of temporary `DataFrame`s is significant compared with your
available system memory, then using `eval()` or `query()` is a good idea.

You can check the approximate size of an array in bytes via `df.values.nbytes`.

As for performance, `eval()` can be faster even when not using all available
system memory. The issue is how temporary `DataFrame`s compare with the size
of the system's CPU L1 or L2 caches. If they are much bigger, then `eval()`
can avoid some slow movement of values between the different memory caches. In
practice, it can be that the difference in performance is not that much, and
the "traditional" methods are faster than `eval`/`query` for smaller arrays.
The main benefit of using `eval`/`query` is saved memory, and sometimes cleaner
syntax.
