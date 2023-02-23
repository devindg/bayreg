import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, polynomial_kernel


class KernelTrick:
    def __init__(self,
                 u: np.ndarray,
                 v: np.ndarray = None,
                 scaler_type: str = 'standard'):
        self.u = u
        self.v = v
        self.scaler_type = scaler_type
        self.num_cols = u.shape[1]

        if v is not None:
            if v.shape[1] != self.num_cols:
                raise ValueError("Array v was provided, but its column length "
                                 "does not match the column length of array u. "
                                 "The number of columns must match across arrays.")

        if scaler_type not in ('standard', 'min_max', 'max_abs'):
            raise ValueError("scaler_type must be 'standard', 'min_max', or 'max_abs'.")
        else:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'min_max':
                scaler = MinMaxScaler()
            else:
                scaler = MaxAbsScaler()

        self.scaler_u = scaler
        self.scaler_u.fit(u)
        self.u_scaled = self.scaler_u.transform(u)

        if self.v is not None:
            self.scaler_v = scaler
            self.scaler_v.fit(v)
            self.v_scaled = self.scaler_v.transform(v)
        else:
            self.scaler_v = None
            self.v_scaled = None

    def rbf_kernel_matrix(self, gamma=None):
        if gamma is None:
            gamma = 1 / self.num_cols
        return rbf_kernel(X=self.u_scaled, Y=self.v_scaled, gamma=gamma)

    def cos_kernel_matrix(self):
        return cosine_similarity(X=self.u_scaled, Y=self.v_scaled)

    def poly_kernel_matrix(self, degree=None):
        if degree is None:
            degree = 3
        return polynomial_kernel(X=self.u_scaled, Y=self.v_scaled, degree=degree)