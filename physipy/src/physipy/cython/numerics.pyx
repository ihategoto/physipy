import numpy as np
cimport numpy as cnp

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cpdef _integrate_numerov_cython(double psi_0, double psi_1, double h, double renorm_threshold, double renorm_factor, double[:] psi, double[:] k_squared):
    cdef Py_ssize_t i, j, n_steps
    cdef double h_squared, temp
    
    n_steps = psi.shape[0]
    h_squared = h * h

    psi[0] = psi_0
    psi[1] = psi_1

    for i in range(0, n_steps - 2):
        temp = (2 * (1 - (5 * h_squared * k_squared[i + 1] / 12)) * psi[i + 1] - (1 + h_squared * k_squared[i] / 12) * psi[i]) / (1 + h_squared * k_squared[i + 2] / 12)
        psi[i + 2] = temp

        if np.abs(temp) > renorm_threshold:
            for j in range(i + 3):
                psi[j] *= renorm_factor

    return np.asarray(psi)


    