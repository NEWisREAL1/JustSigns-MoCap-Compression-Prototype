# Motion Approximation with LSPIA on B-Spline

## Iterative Steps

### 1. At iteration $k$, calculate current curve:

$$
C^{(k)}(t) = \sum_{j=1}^{n} P^{(k)}_j N_{j,p}(t), \quad\text{or}\quad \hat{Q}^{(k)} = BP^{(k)}.
$$

### 2. Calculate the error vector:

$$
E^{(k)}_i = Q_i - C^{(k)}(t_i), \quad\text{or}\quad E^{(k)} = Q - \hat{Q}^{(k)} = Q - BP^{(k)}. 
$$

### 3. Calculate the local update:

The local update vector $\Delta P^{(k)}$ consisting of entries $\Delta P^{(k)}_i$ that gather and compute the weighted average of the errors of all data points that fall within the support of the basis function $N_{i,p}(t)$:

$$
\Delta P^{(k)}_j = \frac{\sum_{i=1}^m N_{j,p}(t_i) E^{(k)}_i}{\sum_{i=1}^m N_{j,p}(t_i)}
$$

or in the matrix form:

$$
\Delta P^{(k)} = D^{-1} B^T E^{(k)} = D^{-1} B^T (Q - BP^{(k)})
$$

### 4. Updating the control points according to the local update:

$$
P^{(k+1)}_j = P^{(k)}_j + \mu \Delta P^{(k)}_j
$$

**vital**: the relaxation factor $\mu$ cannot exceed $2$ or else it will cause the process to diverge.

### Summary of matrix form implementation:

The updating equation can be written in matrix form as:

$$
P^{(k+1)} = P^{{k}} + \mu D^{-1} B^T (Q - BP^{(k)}).
$$

where $B$ is the collocation matrix given by

$$
B_{i,j} = N_{j,p}(t_i)
$$

and $D$ is the diagonal normalization matrix given by

$$
D_{j,j} = \sum_{i=1}^{m} B_{i,j} = B^T \mathbb{1}_m
$$

for caching, we can utilize the weight matrix $W$:

$$
P^{(k+1)} = P^{{k}} + \mu W (Q - BP^{(k)}),\quad W = D^{-1} B^T.
$$

