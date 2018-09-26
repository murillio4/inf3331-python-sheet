# Some tips for assignment 4

## Some numpy magic
```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], np.int32)

b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], np.int32)

print(a*b)

#Result:
[[ 1  4  9]
 [16 25 36]
 [49 64 81]]
```

```python
a = np.array([[2, 3, 2],
              [3, 3, 3],
              [2, 3, 2]], np.float64)

b = a < 3

print(b.type)

#result:
[[ True False  True]
 [False False False]
 [ True False  True]]
```

```python
a = np.array([[True,  False,  True],
              [False, False,  False],
              [True,  False,  True]], np.bool)

b = np.array([[2, 2, 2],
              [2, 2, 2],
              [2, 2, 2]], np.float64)
b[a] += 2

print(b)

#result:
[[4. 2. 4.]
 [2. 2. 2.]
 [4. 2. 4.]]
```

```python
a = np.array([[True,  False,  True],
              [False, False,  False],
              [True,  False,  True]], np.bool)

a = np.logical_not(a)

print(a)

#result:
[[False  True False]
 [ True  True  True]
 [False  True False]]
```

```python
a = np.array([[2, 2, 2],
              [2, 2, 2],
              [2, 2, 2]], dtype=float)

b = np.full(c.shape, 1000, dtype=int)

print(b)

#result
[[1000 1000 1000]
 [1000 1000 1000]
 [1000 1000 1000]]
```

## Numba

```python
@jit
def square(x):
    return x ** 2
```

```python
#return and parameter type defined
@jit(int32(int32))
def square(x):
    return x ** 2
```

```python
#automatic parallelization
@jit(float32[:](float32[:], float32[:]), nopython=True, parallel=True)
def mul(arr1, arr2):
    return arr1*arr2

mul(np.random.rand(1000,1000), np.random.rand(3,2))
```