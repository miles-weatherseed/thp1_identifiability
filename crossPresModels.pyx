cimport cython
import numpy as np
cimport numpy as np

cdef double removalP = 0.00529
cdef double gM = 5.10
cdef double gT = 51.0
cdef double dM = 0.000079892
cdef double dT = 0.001725968
cdef double dMe = 0.00009329349
cdef double bM = 0.00000000093731353
cdef double bT = 0.00000004905
cdef double c = 0.000002449376
cdef double e = 0.1141804
cdef double uSelf = 0.0001
cdef double uT = 0.000001184643
cdef double v = 936.3137
cdef double q = 21035.0
cdef double uA1 = 0.0000963
cdef double uA2 = 0.0000214
cdef double uA3 = 0.0000642
cdef double uA24 = 0.0000214
cdef double uB7 = 0.000128
cdef double delay = 30.0*60.0
cdef double duration = 300.0*60.0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate normal python division checking
cpdef list no_erap_rhs_ivp(double t, double [:] y, double selfSupply, double E1, double kSelf, double KM_self, double k_A1, double k_A2, double k_A3, double k_A24, double k_B7, double A1_s, double A2_s, double A3_s, double A24_s, double B7_s):
    cdef bint tOver = (t < delay + duration)*(t > delay)
    cdef double d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27
    d0 = A1_s*tOver + uA1*y[9] + q*uA1*y[15] - (removalP + bM*y[6] + c*y[8])*y[0]
    d1 = A2_s*tOver + uA2*y[10] + q*uA2*y[16] - (removalP + bM*y[6] + c*y[8])*y[1]
    d2 = A3_s*tOver + uA3*y[11] + q*uA3*y[17] - (removalP + bM*y[6] + c*y[8])*y[2]
    d3 = A24_s*tOver + uA24*y[12] + q*uA24*y[18] - (removalP + bM*y[6] + c*y[8])*y[3]
    d4 = B7_s*tOver + uB7*y[13] + q*uB7*y[19] - (removalP + bM*y[6] + c*y[8])*y[4]
    d5 = selfSupply + uSelf*y[14] + q*uSelf*y[20] - (removalP + bM*y[6] + c*y[8])*y[5]
    d6 = uA1*y[9] + uA2*y[10] + uA3*y[11] + uA24*y[12] + uB7*y[13] + uSelf*y[14] + gM + uT*y[8] - (dM + bM*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + bT*y[7])*y[6]
    d7 = uT*y[8] + gT + uT*v*(y[15] + y[16] + y[17] + y[18] + y[19] + y[20]) - (bT*y[6] + dT)*y[7]
    d8 = bT*y[7]*y[6] + q*(uA1*y[15] + uA2*y[16] + uA3*y[17] + uA24*y[18] + uB7*y[19] + uSelf*y[20]) - (c*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + uT)*y[8]
    d9 = bM*y[6]*y[0] + uT*v*y[15] - (e + uA1)*y[9]
    d10 = bM*y[6]*y[1] + uT*v*y[16] - (e + uA2)*y[10]
    d11 = bM*y[6]*y[2] + uT*v*y[17] - (e + uA3)*y[11]
    d12 = bM*y[6]*y[3] + uT*v*y[18] - (e + uA24)*y[12]
    d13 = bM*y[6]*y[4] + uT*v*y[19] - (e + uB7)*y[13]
    d14 = bM*y[6]*y[5] + uT*v*y[20] - (e + uSelf)*y[14]
    d15 = c*y[8]*y[0] - (q*uA1 + uT*v)*y[15]
    d16 = c*y[8]*y[1] - (q*uA2 + uT*v)*y[16]
    d17 = c*y[8]*y[2] - (q*uA3 + uT*v)*y[17]
    d18 = c*y[8]*y[3] - (q*uA24 + uT*v)*y[18]
    d19 = c*y[8]*y[4] - (q*uB7 + uT*v)*y[19]
    d20 = c*y[8]*y[5] - (q*uSelf + uT*v)*y[20]
    d21 = e*y[9] - uA1*y[21]
    d22 = e*y[10] - uA2*y[22]
    d23 = e*y[11] - uA3*y[23]
    d24 = e*y[12] - uA24*y[24]
    d25 = e*y[13] - uB7*y[25]
    d26 = e*y[14] - uSelf*y[26]
    d27 = uA1*y[21] + uA2*y[22] + uA3*y[23] + uA24*y[24] + uB7*y[25] + uSelf*y[26] - dMe*y[27]
    return [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate normal python division checking
cpdef list no_erap_jac_ivp(double t, double[:] y, double selfSupply, double E1, double kSelf, double KM_self, double k_A1, double k_A2, double k_A3, double k_A24, double k_B7, double A1_s, double A2_s, double A3_s, double A24_s, double B7_s):
    return [[- (removalP + bM*y[6] + c*y[8]), 0, 0, 0, 0, 0, -bM*y[0], 0, -c*y[0], uA1, 0, 0, 0, 0, 0, q*uA1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, - (removalP + bM*y[6] + c*y[8]), 0, 0, 0, 0, -bM*y[1], 0, -c*y[1], 0, uA2, 0, 0, 0, 0, 0, q*uA2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, - (removalP + bM*y[6] + c*y[8]), 0, 0, 0, -bM*y[2], 0, -c*y[2], 0, 0, uA3, 0, 0, 0, 0, 0, q*uA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, - (removalP + bM*y[6] + c*y[8]), 0, 0, -bM*y[3], 0, -c*y[3], 0, 0, 0, uA24, 0, 0, 0, 0, 0, q*uA24, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, - (removalP + bM*y[6] + c*y[8]), 0, -bM*y[4], 0, -c*y[4], 0, 0, 0, 0, uB7, 0, 0, 0, 0, 0, q*uB7, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, - (removalP + bM*y[6] + c*y[8]), -bM*y[5], 0, -c*y[5], 0, 0, 0, 0, 0, uSelf, 0, 0, 0, 0, 0, q*uSelf, 0, 0, 0, 0, 0, 0, 0],
            [-bM*y[6], -bM*y[6], -bM*y[6], -bM*y[6], -bM*y[6], -bM*y[6], - (dM + bM*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + bT*y[7]), -bT*y[6], uT, uA1, uA2, uA3, uA24, uB7, uSelf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -bT*y[7], - (bT*y[6] + dT), uT, 0, 0, 0, 0, 0, 0, uT*v, uT*v, uT*v, uT*v, uT*v, uT*v, 0, 0, 0, 0, 0, 0, 0],
            [-c*y[8], -c*y[8], -c*y[8], -c*y[8], -c*y[8], -c*y[8], bT*y[7], bT*y[6], -(c*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + uT), 0, 0, 0, 0, 0, 0, q*uA1, q*uA2, q*uA3, q*uA24, q*uB7, q*uSelf, 0, 0, 0, 0, 0, 0, 0],
            [bM*y[6], 0, 0, 0, 0, 0, bM*y[0], 0, 0, -(e + uA1), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, bM*y[6], 0, 0, 0, 0, bM*y[0], 0, 0, 0, -(e + uA2), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, bM*y[6], 0, 0, 0, bM*y[0], 0, 0, 0, 0, -(e + uA3), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, bM*y[6], 0, 0, bM*y[0], 0, 0, 0, 0, 0, -(e + uA24), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, bM*y[6], 0, bM*y[0], 0, 0, 0, 0, 0, 0, -(e + uB7), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, bM*y[6], bM*y[0], 0, 0, 0, 0, 0, 0, 0, -(e + uSelf), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0],
            [c*y[8], 0, 0, 0, 0, 0, 0, 0, c*y[0], 0, 0, 0, 0, 0, 0, - (q*uA1 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, c*y[8], 0, 0, 0, 0, 0, 0, c*y[1], 0, 0, 0, 0, 0, 0, 0, - (q*uA2 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, c*y[8], 0, 0, 0, 0, 0, c*y[2], 0, 0, 0, 0, 0, 0, 0, 0, - (q*uA3 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, c*y[8], 0, 0, 0, 0, c*y[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, - (q*uA24 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, c*y[8], 0, 0, 0, c*y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, - (q*uB7 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, c*y[8], 0, 0, c*y[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, - (q*uSelf + uT*v), 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA24, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uB7, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uSelf, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, uA1, uA2, uA3, uA24, uB7, uSelf, -dMe]]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate normal python division checking
cpdef list erap_rhs_ivp(double t, double [:] y, double selfSupply, double E1, double kSelf, double KM_self, double k_A1, double k_A2, double k_A3, double k_A24, double k_B7, double A1_s, double A2_s, double A3_s, double A24_s, double B7_s):
    cdef double d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27
    cdef bint tOver = (t < delay + duration)*(t > delay)
    d0 = A1_s*tOver + uA1*y[9] + q*uA1*y[15] - (removalP + bM*y[6] + c*y[8])*y[0] - E1*k_A1*KM_self*y[0]/(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]))
    d1 = A2_s*tOver + uA2*y[10] + q*uA2*y[16] - (removalP + bM*y[6] + c*y[8])*y[1] - E1*k_A2*KM_self*y[1]/(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]))
    d2 = A3_s*tOver + uA3*y[11] + q*uA3*y[17] - (removalP + bM*y[6] + c*y[8])*y[2] - E1*k_A3*KM_self*y[2]/(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]))
    d3 = A24_s*tOver + uA24*y[12] + q*uA24*y[18] - (removalP + bM*y[6] + c*y[8])*y[3] - E1*k_A24*KM_self*y[3]/(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]))
    d4 = B7_s*tOver + uB7*y[13] + q*uB7*y[19] - (removalP + bM*y[6] + c*y[8])*y[4] - E1*k_B7*KM_self*y[4]/(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]))
    d5 = selfSupply + uSelf*y[14] + q*uSelf*y[20] - (removalP + bM*y[6] + c*y[8])*y[5] - E1*KM_self*(kSelf*y[5] - k_A1*y[0] - k_A2*y[1] - k_A3*y[2] - k_A24*y[3] - k_B7*y[4])/(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]))
    d6 = uA1*y[9] + uA2*y[10] + uA3*y[11] + uA24*y[12] + uB7*y[13] + uSelf*y[14] + gM + uT*y[8] - (dM + bM*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + bT*y[7])*y[6]
    d7 = uT*y[8] + gT + uT*v*(y[15] + y[16] + y[17] + y[18] + y[19] + y[20]) - (bT*y[6] + dT)*y[7]
    d8 = bT*y[7]*y[6] + q*(uA1*y[15] + uA2*y[16] + uA3*y[17] + uA24*y[18] + uB7*y[19] + uSelf*y[20]) - (c*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + uT)*y[8]
    d9 = bM*y[6]*y[0] + uT*v*y[15] - (e + uA1)*y[9]
    d10 = bM*y[6]*y[1] + uT*v*y[16] - (e + uA2)*y[10]
    d11 = bM*y[6]*y[2] + uT*v*y[17] - (e + uA3)*y[11]
    d12 = bM*y[6]*y[3] + uT*v*y[18] - (e + uA24)*y[12]
    d13 = bM*y[6]*y[4] + uT*v*y[19] - (e + uB7)*y[13]
    d14 = bM*y[6]*y[5] + uT*v*y[20] - (e + uSelf)*y[14]
    d15 = c*y[8]*y[0] - (q*uA1 + uT*v)*y[15]
    d16 = c*y[8]*y[1] - (q*uA2 + uT*v)*y[16]
    d17 = c*y[8]*y[2] - (q*uA3 + uT*v)*y[17]
    d18 = c*y[8]*y[3] - (q*uA24 + uT*v)*y[18]
    d19 = c*y[8]*y[4] - (q*uB7 + uT*v)*y[19]
    d20 = c*y[8]*y[5] - (q*uSelf + uT*v)*y[20]
    d21 = e*y[9] - uA1*y[21]
    d22 = e*y[10] - uA2*y[22]
    d23 = e*y[11] - uA3*y[23]
    d24 = e*y[12] - uA24*y[24]
    d25 = e*y[13] - uB7*y[25]
    d26 = e*y[14] - uSelf*y[26]
    d27 = uA1*y[21] + uA2*y[22] + uA3*y[23] + uA24*y[24] + uB7*y[25] + uSelf*y[26] - dMe*y[27]
    return [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate normal python division checking
cpdef list erap_jac_ivp(double t, double[:] y, double selfSupply, double E1, double kSelf, double KM_self, double k_A1, double k_A2, double k_A3, double k_A24, double k_B7, double A1_s, double A2_s, double A3_s, double A24_s, double B7_s):
    return [[- (removalP + bM*y[6] + c*y[8]) -k_A1*KM_self*E1*(1 + KM_self*(y[1] + y[2] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A1*KM_self*KM_self*E1*y[0]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A1*KM_self*KM_self*E1*y[0]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A1*KM_self*KM_self*E1*y[0]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A1*KM_self*KM_self*E1*y[0]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A1*KM_self*KM_self*E1*y[0]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), -bM*y[0], 0, -c*y[0], uA1, 0, 0, 0, 0, 0, q*uA1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [k_A2*KM_self*KM_self*E1*y[1]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), - (removalP + bM*y[6] + c*y[8]) - k_A2*KM_self*E1*(1 + KM_self*(y[0] + y[2] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A2*KM_self*KM_self*E1*y[1]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A2*KM_self*KM_self*E1*y[1]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A2*KM_self*KM_self*E1*y[1]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A2*KM_self*KM_self*E1*y[1]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), -bM*y[1], 0, -c*y[1], 0, uA2, 0, 0, 0, 0, 0, q*uA2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [k_A3*KM_self*KM_self*E1*y[2]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A3*KM_self*KM_self*E1*y[2]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), - (removalP + bM*y[6] + c*y[8]) - k_A3*KM_self*E1*(1 + KM_self*(y[0] + y[1] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A3*KM_self*KM_self*E1*y[2]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A3*KM_self*KM_self*E1*y[2]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A3*KM_self*KM_self*E1*y[2]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), -bM*y[2], 0, -c*y[2], 0, 0, uA3, 0, 0, 0, 0, 0, q*uA3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [k_A24*KM_self*KM_self*E1*y[3]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A24*KM_self*KM_self*E1*y[3]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A24*KM_self*KM_self*E1*y[3]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), - (removalP + bM*y[6] + c*y[8]) - k_A24*KM_self*E1*(1 + KM_self*(y[0] + y[1] + y[2] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A24*KM_self*KM_self*E1*y[3]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A24*KM_self*KM_self*E1*y[3]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), -bM*y[3], 0, -c*y[3], 0, 0, 0, uA24, 0, 0, 0, 0, 0, q*uA24, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [k_B7*KM_self*KM_self*E1*y[4]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_B7*KM_self*KM_self*E1*y[4]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_B7*KM_self*KM_self*E1*y[4]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_B7*KM_self*KM_self*E1*y[4]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), - (removalP + bM*y[6] + c*y[8]) - k_B7*KM_self*E1*(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_B7*KM_self*KM_self*E1*y[4]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), -bM*y[4], 0, -c*y[4], 0, 0, 0, 0, uB7, 0, 0, 0, 0, 0, q*uB7, 0, 0, 0, 0, 0, 0, 0, 0],
            [k_A1*KM_self*E1*(1 + KM_self*(y[1] + y[2] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)) + kSelf*KM_self*KM_self*E1*y[5]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A2*KM_self*E1*(1 + KM_self*(y[0] + y[2] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)) + kSelf*KM_self*KM_self*E1*y[5]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A3*KM_self*E1*(1 + KM_self*(y[1] + y[0] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)) + kSelf*KM_self*KM_self*E1*y[5]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_A24*KM_self*E1*(1 + KM_self*(y[1] + y[2] + y[0] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)) + kSelf*KM_self*KM_self*E1*y[5]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), k_B7*KM_self*E1*(1 + KM_self*(y[1] + y[2] + y[3] + y[4] + y[5]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[0] + y[5])**2)) + kSelf*KM_self*KM_self*E1*y[5]/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), - (removalP + bM*y[6] + c*y[8]) - kSelf*KM_self*E1*(1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4]))/((1 + KM_self*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5])**2)), -bM*y[5], 0, -c*y[5], 0, 0, 0, 0, 0, uSelf, 0, 0, 0, 0, 0, q*uSelf, 0, 0, 0, 0, 0, 0, 0],
            [-bM*y[6], -bM*y[6], -bM*y[6], -bM*y[6], -bM*y[6], -bM*y[6], - (dM + bM*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + bT*y[7]), -bT*y[6], uT, uA1, uA2, uA3, uA24, uB7, uSelf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -bT*y[7], - (bT*y[6] + dT), uT, 0, 0, 0, 0, 0, 0, uT*v, uT*v, uT*v, uT*v, uT*v, uT*v, 0, 0, 0, 0, 0, 0, 0],
            [-c*y[8], -c*y[8], -c*y[8], -c*y[8], -c*y[8], -c*y[8], bT*y[7], bT*y[6], -(c*(y[0] + y[1] + y[2] + y[3] + y[4] + y[5]) + uT), 0, 0, 0, 0, 0, 0, q*uA1, q*uA2, q*uA3, q*uA24, q*uB7, q*uSelf, 0, 0, 0, 0, 0, 0, 0],
            [bM*y[6], 0, 0, 0, 0, 0, bM*y[0], 0, 0, -(e + uA1), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, bM*y[6], 0, 0, 0, 0, bM*y[0], 0, 0, 0, -(e + uA2), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, bM*y[6], 0, 0, 0, bM*y[0], 0, 0, 0, 0, -(e + uA3), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, bM*y[6], 0, 0, bM*y[0], 0, 0, 0, 0, 0, -(e + uA24), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, bM*y[6], 0, bM*y[0], 0, 0, 0, 0, 0, 0, -(e + uB7), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, bM*y[6], bM*y[0], 0, 0, 0, 0, 0, 0, 0, -(e + uSelf), 0, 0, 0, 0, 0, uT*v, 0, 0, 0, 0, 0, 0, 0],
            [c*y[8], 0, 0, 0, 0, 0, 0, 0, c*y[0], 0, 0, 0, 0, 0, 0, - (q*uA1 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, c*y[8], 0, 0, 0, 0, 0, 0, c*y[1], 0, 0, 0, 0, 0, 0, 0, - (q*uA2 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, c*y[8], 0, 0, 0, 0, 0, c*y[2], 0, 0, 0, 0, 0, 0, 0, 0, - (q*uA3 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, c*y[8], 0, 0, 0, 0, c*y[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, - (q*uA24 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, c*y[8], 0, 0, 0, c*y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, - (q*uB7 + uT*v), 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, c*y[8], 0, 0, c*y[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, - (q*uSelf + uT*v), 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uA24, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uB7, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -uSelf, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, e, 0, 0, 0, 0, 0, 0, uA1, uA2, uA3, uA24, uB7, uSelf, -dMe]]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Deactivate normal python division checking
cpdef list no_erap_ss(double t, double [:] y, double selfSupply, double E1, double kSelf, double KM_self, double k_A1, double k_A2, double k_A3, double k_A24, double k_B7, double A1_s, double A2_s, double A3_s, double A24_s, double B7_s):
    cdef double d0, d1, d2, d3, d4, d5, d6, d7
    d0 = selfSupply + uSelf*y[4] + q*uSelf*y[5] - (removalP + bM*y[1] + c*y[3])*y[0]
    d1 = uSelf*y[4] + gM + uT*y[3] - (dM + bM*y[0] + bT*y[2])*y[1]
    d2 = uT*y[3] + gT + uT*v*y[5] - (bT*y[1] + dT)*y[2]
    d3 = bT*y[2]*y[1] + q*uSelf*y[5] - (c*y[0] + uT)*y[3]
    d4 = bM*y[1]*y[0] + uT*v*y[5] - (e + uSelf)*y[4]
    d5 = c*y[0]*y[3] - (q*uSelf + uT*v)*y[5]
    d6 = e*y[5] - uSelf*y[6]
    d7 = uSelf*y[6] - dMe*y[7]
    return [d0, d1, d2, d3, d4, d5, d6, d7]