##########################################
# FILE INFORMATION:
# Purpose: all the DNN subfunctions 
# Created: 20260324
# Last changed: 20260324
##########################################

print(f"[INFO]: Script began running!")

##########################################
# Importing Python Libraries
##########################################

import numpy as np
import tensorflow as tf

print("[INFO]: Libraries imported!")

##########################################
# Tensorflow Configuration
##########################################

_NUMBER_NODES_HIDDEN_1 = 10
_NUMBER_NODES_HIDDEN_2 = 10
_NUMBER_NODES_HIDDEN_3 = 10
_NUMBER_NODES_HIDDEN_4 = 10

print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_1} nodes in 1st hidden layer")
print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_2} nodes in 2nd hidden layer")
print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_3} nodes in 3rd hidden layer")
print(f"[INFO]: Replica DNN has {_NUMBER_NODES_HIDDEN_4} nodes in 4th hidden layer")

### USE THIS FOR TENSORFLOW ONLY ###
_CONVERSION_GEV6_GEV4NB = tf.constant(.389379 * 1000000.)
_MASS_OF_PROTON_IN_GEV = tf.constant(0.93827208816)
_QED_FINE_STRUCTURE = tf.constant(1./137.035999177)
_ELECTRIC_FORM_FACTOR_CONSTANT = tf.constant(0.710649)
_PROTON_MAGNETIC_MOMENT = tf.constant(2.79284734463)

def compute_fe(t):
    return 1./ (1. - (t / _ELECTRIC_FORM_FACTOR_CONSTANT))**2

def compute_fg(fe):
    return _PROTON_MAGNETIC_MOMENT * fe

def compute_f2(t, fe, fg):
    tau = -1. * t/ (4. * _MASS_OF_PROTON_IN_GEV**2)
    numerator = fg - fe
    denominator = 1. + tau
    return numerator / denominator

def compute_f1(fg, f2):
    return fg - f2

def compute_epsilon(xb, q_squared):
    return 2. * xb * _MASS_OF_PROTON_IN_GEV / tf.sqrt(q_squared)

def compute_y(k_beam, q_squared, ep):
    return tf.sqrt(q_squared) / (ep * k_beam)

def compute_skewness(xb, t, q_squared):
    return xb * (1. + (t / (2. * q_squared))) / (2. - xb + (xb * t / q_squared))

def compute_t_min(xb, q_squared, ep):
    return -1. * q_squared * ((2. * (1. - xb) * (1. - tf.sqrt(1. + ep**2))) + ep**2) / ((4. * xb * (1. - xb)) + ep**2)

def compute_t_prime(t, tmin):
    return (t-tmin)

def compute_k_tilde(xb, q_squared, t, tmin, ep):
    return tf.sqrt(tmin - t) * tf.sqrt(((1. - xb) * tf.sqrt((1. + ep**2))) + (((tmin - t) * (ep**2 + (4. * (1. - xb) * xb))) / (4. * q_squared)))

def compute_k(q_squared, y_lep, ep, k_tilde):
    return tf.sqrt(((1. - y_lep + (ep**2 * y_lep**2 / 4.)) / q_squared)) * k_tilde

def compute_k_dot_delta(q_squared, xb, t, phi_azi, ep, y_lep, k):
    return (-1.*q_squared / (2.*y_lep*(1.+ep**2))) * (1. + ((2.*k*tf.cos(tf.constant(np.pi)- phi_azi)) - ((t / q_squared)*(1.-(xb * (2. - y_lep)) + (y_lep * ep**2 / 2.))) + (y_lep * ep**2 / 2.)))

def prop_1(q_squared, kdd):
    return (1. + (2. * (kdd / q_squared)))

def prop_2(q_squared, t, kdd):
    return ((-2. * (kdd / q_squared)) + (t / q_squared))

def bh_unp_c0(
    q_sq: float, xb: float, t: float, ep: float,
    y: float, k: float, f1: float, f2: float):
    first_line = 8. * k**2 * (((2. + 3. * ep**2) * (f1**2 - (t * f2**2 / (4. * _MASS_OF_PROTON_IN_GEV**2))) / (t / q_sq)) + (2. * xb**2 * (f1 + f2)**2))
    second_line_first_part = (2. + ep**2) * ((4. * xb**2 * _MASS_OF_PROTON_IN_GEV**2 / t) * (1. + (t / q_sq))**2 + 4. * (1 - xb) * (1. + (xb * (t / q_sq)))) * (f1**2 - (t * f2**2 / (4. * _MASS_OF_PROTON_IN_GEV**2)))
    second_line_second_part = 4. * xb**2 * (xb + (1. - xb + (ep**2 / 2.)) * (1 - (t / q_sq))**2 - xb * (1. - 2. * xb) * (t / q_sq)**2) * (f1 + f2)**2
    second_line = (2. - y)**2 * (second_line_first_part + second_line_second_part)
    third_line = 8. * (1. + ep**2) * (1. - y - (ep**2 * y**2 / 4.)) * (2. * ep**2 * (1 - (t / (4. * _MASS_OF_PROTON_IN_GEV**2))) * (f1**2 - (t * f2**2 / (4. * _MASS_OF_PROTON_IN_GEV**2))) - xb**2 * (1 - (t / q_sq))**2 * (f1 + f2)**2)
    c0_unpolarized_bh = first_line + second_line + third_line
    return c0_unpolarized_bh

def bh_unp_c1(
    q_sq: float, xb: float, t: float, ep: float,
    y: float, k: float, f1: float, f2: float) -> float:
    addition_of_form_factors_squared = (f1 + f2)**2
    weighted_combination_of_form_factors = f1**2 - ((t / (4. * _MASS_OF_PROTON_IN_GEV**2)) * f2**2)
    first_line_first_part = ((4. * xb**2 * _MASS_OF_PROTON_IN_GEV**2 / t) - 2. * xb - ep**2) * weighted_combination_of_form_factors
    first_line_second_part = 2. * xb**2 * (1. - (1. - 2. * xb) * (t / q_sq)) * addition_of_form_factors_squared
    c1_unpolarized_bh = 8. * k * (2. - y) * (first_line_first_part + first_line_second_part)
    return c1_unpolarized_bh

def bh_unp_c2( 
    xb: float, t: float, k: float, f1: float, f2: float) -> float:
    addition_of_form_factors_squared = (f1 + f2)**2
    weighted_combination_of_form_factors = f1**2 - ((t/ (4. * _MASS_OF_PROTON_IN_GEV**2)) * f2**2)
    first_part_of_contribution = (4. * _MASS_OF_PROTON_IN_GEV**2 / t) * weighted_combination_of_form_factors
    c2_unpolarized_bh = 8. * xb**2 * k**2 * (first_part_of_contribution + 2. * addition_of_form_factors_squared)
    return c2_unpolarized_bh

def bh_lp_c0(
    lep_helicity: float,target_polar: float,
    q_sq: float, xb: float, t: float,ep: float,y: float, f1: float, f2: float) -> float:
    sum_of_form_factors = (f1 + f2)
    t_over_four_mp_squared = t / (4. * _MASS_OF_PROTON_IN_GEV**2)
    weighted_sum_of_form_factors = f1 + t_over_four_mp_squared * f2
    one_minus_xb = 1. - xb
    t_over_Q_squared = t / q_sq
    one_minus_t_over_Q_squared = 1. - t_over_Q_squared
    first_term_first_bracket = 0.5 * xb * (one_minus_t_over_Q_squared) - t_over_four_mp_squared
    first_term_second_bracket = 2. - xb - (2. * (one_minus_xb)**2 * t_over_Q_squared) + (ep**2 * one_minus_t_over_Q_squared) - (xb * (1. - 2. * xb) * t_over_Q_squared**2)
    first_term = 0.5 * sum_of_form_factors * first_term_first_bracket * first_term_second_bracket
    second_term_first_bracket = xb**2 * (1. + t_over_Q_squared)**2 / (4. * t_over_four_mp_squared) + ((1. - xb) * (1. + xb * t_over_Q_squared))
    second_term = (1. - (1. - xb) * t_over_Q_squared) * weighted_sum_of_form_factors * second_term_first_bracket
    prefactor = 8. * float(lep_helicity) * float(target_polar) * xb * (2. - y) * y * tf.sqrt(1. + ep**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
    c0LP_BH = prefactor * (first_term + second_term)
    return c0LP_BH

def bh_lp_c1(
    lep_helicity: float,target_polar: float,
    q_sq: float, xb: float, t: float,ep: float,y: float,shorthand_k: float,f1: float, f2: float) -> float:
    sum_of_form_factors = (f1 + f2)
    t_over_four_mp_squared = t / (4. * _MASS_OF_PROTON_IN_GEV**2)
    weighted_sum_of_form_factors = f1 + t_over_four_mp_squared * f2
    t_over_Q_squared = t / q_sq
    first_term = ((2. * t_over_four_mp_squared) - (xb * (1. - t_over_Q_squared))) * ((1. - xb + (xb * t_over_Q_squared))) * sum_of_form_factors
    second_term_bracket_term = 1. + xb - ((3. - 2. * xb) * (1. + xb * t_over_Q_squared)) - (xb**2 * (1. + t_over_Q_squared**2) / t_over_four_mp_squared)
    second_term = weighted_sum_of_form_factors * second_term_bracket_term
    prefactor = -8. * lep_helicity * target_polar * xb * y * shorthand_k * tf.sqrt(1. + ep**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
    c1LP_BH = prefactor * (first_term + second_term)
    return c1LP_BH

def bh_squared(lep_helicity, target_polar, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2):

    bh_c0 = bh_unp_c0(q_sq, xb, t, ep, y, k, f1, f2)
    bh_c1 = bh_unp_c1(q_sq, xb, t, ep, y, k, f1, f2)
    bh_c2 = bh_unp_c2(xb, t, k, f1, f2)

    bh_c0_lp = bh_lp_c0(lep_helicity, target_polar, q_sq, xb, t, ep, y, f1, f2)
    bh_c1_lp = bh_lp_c1(lep_helicity, target_polar, q_sq, xb, t, ep, y, k, f1, f2)

    return ((
        (bh_c0 + bh_c0_lp) * tf.cos(0.* (tf.constant(np.pi) - phi)) +
        (bh_c1 + bh_c1_lp) * tf.cos(1.* (tf.constant(np.pi) - phi)) + 
        (bh_c2 + 0.0) * tf.cos(2.* (tf.constant(np.pi) - phi))) / (xb * xb * y * y * (1.+ep**2)**2 * t * p1 * p2))

def f_eff(xi: float, cff: complex, use_ww: bool = True):
    if use_ww:
        cff_effective = 2. * cff / (1. + xi)
    else:
        cff_effective = -2. * xi * cff / (1. + xi)
    return cff_effective

def curly_c_real(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):
    
    first_line = (4.*(1.-xb)*(cff_re_h*cff_re_h_star - cff_im_h*cff_im_h_star)) + (4.*(1.-xb + 0.25*((2.*q_sq + t)*ep**2)/(q_sq + xb*t))*(cff_re_ht * cff_re_ht_star - cff_im_ht * cff_im_ht_star))
    next_line = -xb**2*(q_sq+t)**2*(cff_re_h*cff_re_e_star - cff_im_e*cff_im_h_star + cff_re_e*cff_re_h_star - cff_im_h*cff_im_e_star)/(q_sq*(q_sq+xb*t)) - (xb**2*q_sq*(cff_re_ht*cff_re_et_star - cff_im_et *cff_im_ht_star + cff_re_et*cff_re_ht_star - cff_im_ht*cff_im_et_star)/(q_sq+xb*t))
    final_line = -1.*(xb**2*(q_sq+t)**2/(q_sq*(q_sq+xb*t)) + 0.25*((2.-xb)*q_sq+xb*t)**2*t/(q_sq*_MASS_OF_PROTON_IN_GEV**2*(q_sq+xb*t)))*(cff_re_e*cff_re_e_star - cff_im_e*cff_im_e_star) -0.25*xb**2*q_sq*t*(cff_re_et*cff_re_et_star - cff_im_et*cff_im_et_star)/((q_sq+xb*t)*_MASS_OF_PROTON_IN_GEV**2)

    return ((first_line + next_line + final_line)*q_sq*(q_sq+xb*t)/((2.-xb)*q_sq+xb*t)**2)

def curly_c_imag(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):
    
    first_line = (4.*(1.-xb)*(cff_im_h*cff_re_h_star + cff_re_h*cff_im_h_star)) + (4.*(1.-xb + 0.25*(2.*q_sq + t)*ep**2/(q_sq + xb*t))*(cff_im_ht * cff_re_ht_star + cff_re_ht * cff_im_ht_star))
    next_line = -xb**2*(q_sq+t)**2*(cff_im_h*cff_re_e_star + cff_re_e*cff_im_h_star + cff_im_e*cff_re_h_star + cff_re_h*cff_im_e_star)/(q_sq*(q_sq+xb*t)) - (xb**2*q_sq*(cff_im_ht*cff_re_et_star + cff_re_et*cff_im_ht_star + cff_im_et*cff_re_ht_star + cff_re_ht*cff_im_et_star)/(q_sq+xb*t))
    final_line = -1.*(xb**2*(q_sq+t)**2/(q_sq*(q_sq+xb*t)) + 0.25*((2.-xb)*q_sq+xb*t)**2*t/(q_sq*_MASS_OF_PROTON_IN_GEV**2*(q_sq+xb*t)))*(cff_im_e*cff_re_e_star + cff_re_e*cff_im_e_star) -0.25*xb**2*q_sq*t*(cff_im_et*cff_re_et_star + cff_re_et*cff_im_et_star)/((q_sq+xb*t)*_MASS_OF_PROTON_IN_GEV**2)

    return ((first_line + next_line + final_line)*q_sq*(q_sq+xb*t)/((2.-xb)*q_sq+xb*t)**2)

def curly_c_real_lp(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):

    first_line=(4.*(1.-xb+(((3.-2.*xb)*q_sq + t))*ep*ep/(4.*(q_sq+xb*t)))*(cff_re_h*cff_re_ht_star-cff_im_ht*cff_im_h_star+cff_re_ht*cff_re_h_star-cff_im_h*cff_im_ht_star))
    second_line=-xb*xb*(q_sq-xb*t*(1.-2.*xb))*(cff_re_h*cff_re_et_star-cff_im_et*cff_im_h_star+cff_re_et*cff_re_h_star-cff_im_h*cff_im_et_star+cff_re_ht*cff_re_e_star-cff_im_e*cff_im_ht_star+cff_re_e*cff_re_ht_star-cff_im_ht*cff_im_e_star)/(q_sq+xb*t)
    third_line=-xb*((4.*(1.-xb)*(q_sq+xb*t)*t+ep*ep*(q_sq+t)**2)*(cff_re_ht*cff_re_e_star-cff_im_e*cff_im_ht_star+cff_re_e*cff_re_ht_star-cff_im_ht*cff_im_e_star))/(2.*q_sq*(q_sq+xb*t))
    fourth_line=-xb*((q_sq*(2.-xb)+xb*t)/(q_sq+xb*t))*((xb*xb*(q_sq+t)**2)/(2*q_sq*(q_sq*(2.-xb)+xb*t)) + t/(4.*_MASS_OF_PROTON_IN_GEV**2))*(cff_re_e*cff_re_et_star-cff_im_e*cff_im_et_star+cff_re_et*cff_re_e_star-cff_im_et*cff_im_e_star)

    return ((first_line+second_line+third_line+fourth_line)*q_sq*(q_sq+xb*t)/(tf.sqrt(1.+ep*ep)*((2.-xb)*q_sq+xb*t)**2))

def curly_c_imag_lp(
    q_sq: float, xb: float, t: float, ep: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    cff_re_h_star: float, cff_re_ht_star: float, cff_re_e_star: float, cff_re_et_star: float,
    cff_im_h_star: float, cff_im_ht_star: float, cff_im_e_star: float, cff_im_et_star: float):

    first_line=(4.*(1.-xb+(((3.-2.*xb)*q_sq + t))*ep*ep/(4.*(q_sq+xb*t)))*(cff_im_h*cff_re_ht_star+cff_re_ht*cff_im_h_star+cff_im_ht*cff_re_h_star+cff_re_h*cff_im_ht_star))
    second_line=-xb*xb*(q_sq-xb*t*(1.-2.*xb))*(cff_im_h*cff_re_et_star+cff_re_et*cff_im_h_star+cff_im_et*cff_re_h_star+cff_re_h*cff_im_et_star+cff_im_ht*cff_re_e_star+cff_re_e*cff_im_ht_star+cff_im_e*cff_re_ht_star+cff_re_ht*cff_im_e_star)/(q_sq+xb*t)
    third_line=-xb*((4.*(1.-xb)*(q_sq+xb*t)*t+ep*ep*(q_sq+t)**2)*(cff_im_h*cff_re_et_star+cff_re_et*cff_im_h_star+cff_im_et*cff_re_h_star+cff_re_h*cff_im_et_star)/(2.*q_sq*(q_sq+xb*t)))
    fourth_line=-xb*((q_sq*(2.-xb)+xb*t)/(q_sq+xb*t))*((xb*xb*(q_sq+t)**2)/(2*q_sq*(q_sq*(2.-xb)+xb*t)) + t/(4.*_MASS_OF_PROTON_IN_GEV**2))*(cff_im_e*cff_re_et_star+cff_re_e*cff_im_et_star+cff_im_et*cff_re_e_star+cff_re_et*cff_im_e_star)

    return ((first_line+second_line+third_line+fourth_line)*q_sq*(q_sq+xb*t)/(tf.sqrt(1.+ep*ep)*((2.-xb)*q_sq+xb*t)**2))

def dvcs_unp_c0(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,
    cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:
    
    first_term_prefactor = 2. * ( 2. - 2. * y + y**2 + (ep**2 * y**2 / 2.)) / (1. + ep**2)
    second_term_prefactor = 16. * k**2 / ((2. - xb)**2 * (1. + ep**2))
    first_term_curlyc = curly_c_real(
        q_sq, xb, t, ep,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        cff_im_h, cff_im_ht, cff_im_e, cff_im_et,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    second_term_curlyc = curly_c_real(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, -1.*cff_im_h, use_ww), f_eff(xi, -1.*cff_im_ht, use_ww), f_eff(xi, -1.*cff_im_e, use_ww), f_eff(xi, -1.*cff_im_et, use_ww))
    c0_dvcs_unpolarized_coefficient = first_term_prefactor * first_term_curlyc + second_term_prefactor * second_term_curlyc
    return c0_dvcs_unpolarized_coefficient

def dvcs_unp_c1(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float,
    cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = 8. * k * (2. - y) / ((2. - xb) * (1. + ep**2))
    curlyC_unp_DVCS = curly_c_real(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_unp_s1(
    lep_helicity: float,q_sq: float,xb: float,t: float,ep: float,y: float,xi: float, k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,
    cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:
    prefactor = -8. * k * lep_helicity * y * tf.sqrt(1. + ep**2) / ((2. - xb) * (1. + ep**2))
    curlyC_unp_DVCS = curly_c_imag(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_lp_c0(
    lep_helicity: float, target_polar: float,
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = 2.*lep_helicity*target_polar*y*(2.-y)/tf.sqrt(1.+ep*ep)
    first_term_curlyc = curly_c_real_lp(
        q_sq, xb, t, ep,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        cff_im_h, cff_im_ht, cff_im_e, cff_im_et,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * first_term_curlyc)

def dvcs_lp_c1(
    lep_helicity: float, target_polar: float,
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = 8.*target_polar*k*lep_helicity*y*tf.sqrt(1+ep*ep)/((2.-xb)*(1.+ep*ep))
    curlyC_unp_DVCS = curly_c_real_lp(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_lp_s1(
    lep_helicity: float, target_polar: float,
    q_sq: float,xb: float,t: float,ep: float,y: float,xi: float,k: float,
    cff_re_h: float,cff_re_ht: float,cff_re_e: float,cff_re_et: float,cff_im_h: float,cff_im_ht: float,cff_im_e: float,cff_im_et: float,
    use_ww: bool = True) -> float:

    prefactor = -8.*target_polar*k*(2.-y)/((2.-xb)*(1.+ep*ep))
    curlyC_unp_DVCS = curly_c_imag_lp(
        q_sq, xb, t, ep,
        f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww),
        f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww),
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et,
        -1.*cff_im_h, -1.*cff_im_ht, -1.*cff_im_e, -1.*cff_im_et)
    return (prefactor * curlyC_unp_DVCS)

def dvcs_squared(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, phi,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    dvcs_c0 = dvcs_unp_c0(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_c1 = dvcs_unp_c1(q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_s1 = dvcs_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    dvcs_c0_lp = dvcs_lp_c0(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_c1_lp = dvcs_lp_c1(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_s1_lp = dvcs_lp_s1(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    return (
        (
            (dvcs_c0 + dvcs_c0_lp) * tf.cos(0.* (tf.constant(np.pi) - phi)) +
            (dvcs_c1 + dvcs_c1_lp) * tf.cos(1.* (tf.constant(np.pi) - phi)) + 
            (dvcs_s1 + dvcs_s1_lp) * tf.sin(1.* (tf.constant(np.pi) - phi))) / (y * y * q_sq))

def i_c_unp_pp_0(
    q_sq: float,xb: float,t: float,ep: float,y: float,k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    two_minus_xb = 2. - xb
    two_minus_y = 2. - y
    first_term_in_brackets = k_tilde**2 * two_minus_y**2 / (q_sq * root_one_plus_epsilon_squared)
    second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (1. - y - (ep**2 * y**2 / 4.))
    second_term_in_brackets_second_part_numerator = 2. * xb * t_over_Q_squared * (two_minus_xb + 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep**2 / xb) + ep**2
    second_term_in_brackets_second_part =  1. + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
    prefactor = -4. * two_minus_y * one_plus_root_epsilon_stuff / tf.pow(root_one_plus_epsilon_squared, 4)
    c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)
    return c_0_plus_plus_unp

def i_c_unp_v_pp_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_term_in_brackets = (2. - y)**2 * k_tilde**2 / (root_one_plus_epsilon_squared * q_sq)
    second_term_first_multiplicative_term = 1. - y - (ep**2 * y**2 / 4.)
    second_term_second_multiplicative_term = one_plus_root_epsilon_stuff / 2.
    second_term_third_multiplicative_term = 1. + t_over_Q_squared
    second_term_fourth_multiplicative_term = 1. + (root_one_plus_epsilon_squared - 1. + (2. * xb)) * t_over_Q_squared / one_plus_root_epsilon_stuff
    second_term_in_brackets = second_term_first_multiplicative_term * second_term_second_multiplicative_term * second_term_third_multiplicative_term * second_term_fourth_multiplicative_term
    coefficient_prefactor = 8. * (2. - y) * xb * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_0_plus_plus_V_unp = coefficient_prefactor * (first_term_in_brackets + second_term_in_brackets)
    return c_0_plus_plus_V_unp

def i_c_unp_a_pp_0(
    q_sq: float,xb: float,t: float,ep: float,y: float,k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    two_minus_y = 2. - y
    ktilde_over_Q_squared = k_tilde**2 / q_sq
    curly_bracket_first_term = two_minus_y**2 * ktilde_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb) / (2. * root_one_plus_epsilon_squared)
    deepest_parentheses_term = (xb * (2. + one_plus_root_epsilon_stuff - 2. * xb) / one_plus_root_epsilon_stuff + (one_plus_root_epsilon_stuff - 2.)) * t_over_Q_squared
    square_bracket_term = one_plus_root_epsilon_stuff * (one_plus_root_epsilon_stuff - xb + deepest_parentheses_term) / 2. - (2. * ktilde_over_Q_squared)
    curly_bracket_second_term = (1. - y - ep**2 * y**2 / 4.) * square_bracket_term
    coefficient_prefactor = 8. * two_minus_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_0_plus_plus_A_unp = coefficient_prefactor * (curly_bracket_first_term + curly_bracket_second_term)
    return c_0_plus_plus_A_unp

def i_c_unp_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    bracket_quantity = ep**2 + t * (2. - 6.* xb - ep**2) / (3. * q_sq)
    prefactor = 12. * tf.sqrt(2.) * k * (2. - y) * tf.sqrt(1. - y - (ep**2 * y**2 / 4)) / tf.pow(1. + ep**2, 2.5)
    c_0_zero_plus_unp = prefactor * bracket_quantity
    return c_0_zero_plus_unp

def i_c_unp_v_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    t_over_Q_squared = t / q_sq
    main_part = xb * t_over_Q_squared * (1. - (1. - 2. * xb) * t_over_Q_squared)
    prefactor = 24. * tf.sqrt(2.) * k * (2. - y) * tf.sqrt(1. - y - (y**2 * ep**2 / 4.)) / (1. + ep**2)**2.5
    c_0_zero_plus_V_unp = prefactor * main_part
    return c_0_zero_plus_V_unp

def i_c_unp_a_0p_0(
    q_sq: float, xb: float, t: float,ep: float,y: float, k: float):
    t_over_Q_squared = t / q_sq
    fancy_xb_epsilon_term = 8. - 6. * xb + 5. * ep**2
    brackets_term = 1. - t_over_Q_squared * (2. - 12. * xb * (1. - xb) - ep**2) / fancy_xb_epsilon_term
    prefactor = 4. * tf.sqrt(2.) * k * (2. - y) * tf.sqrt(1. - y - (y**2 * ep**2 / 4.)) / tf.pow(1. + ep**2, 2.5)
    c_0_zero_plus_A_unp = prefactor * t_over_Q_squared * fancy_xb_epsilon_term * brackets_term
    return c_0_zero_plus_A_unp

def i_c_unp_pp_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_first_term = (1. + (1. - x_Bjorken) * (root_one_plus_epsilon_squared - 1.) / (2. * x_Bjorken) + epsilon**2 / (4. * x_Bjorken)) * x_Bjorken * t_over_Q_squared
    first_bracket_term = first_bracket_first_term - 3. * epsilon**2 / 4.
    second_bracket_term = 1. - (1. - 3. * x_Bjorken) * t_over_Q_squared + (1. - root_one_plus_epsilon_squared + 3. * epsilon**2) * x_Bjorken * t_over_Q_squared / (one_plus_root_epsilon_stuff - epsilon**2)
    fancy_y_coefficient = 2. - 2. * lepton_energy_fraction_y + lepton_energy_fraction_y**2 + epsilon**2 * lepton_energy_fraction_y**2 / 2.
    second_term = -4. * shorthand_k * fancy_y_coefficient * (one_plus_root_epsilon_stuff - epsilon**2) * second_bracket_term / root_one_plus_epsilon_squared**5
    first_term = -16. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * first_bracket_term / root_one_plus_epsilon_squared**5
    c_1_plus_plus_unp = first_term + second_term
    return c_1_plus_plus_unp

def i_c_unp_v_pp_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    first_bracket_term = (2. - lepton_energy_fraction_y)**2 * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)
    second_bracket_term_first_part = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    second_bracket_term_second_part = 0.5 * (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * t_prime / squared_Q_momentum_transfer
    coefficient_prefactor = 16. * shorthand_k * x_Bjorken * t_over_Q_squared / tf.pow(root_one_plus_epsilon_squared, 5)
    c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)
    return c_1_plus_plus_V_unp

def i_c_unp_a_pp_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    one_minus_2xb = 1. - 2. * x_Bjorken
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    first_bracket_term_second_part = 1. - one_minus_2xb * t_over_Q_squared + (4. * x_Bjorken * one_minus_xb + epsilon**2) * t_prime_over_Q_squared / (4. * root_one_plus_epsilon_squared)
    second_bracket_term = 1. - 0.5 * x_Bjorken + 0.25 * (one_minus_2xb + root_one_plus_epsilon_squared) * (1. - t_over_Q_squared) + (4. * x_Bjorken * one_minus_xb + epsilon**2) * t_prime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -16. * shorthand_k * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (2. - lepton_energy_fraction_y)**2 * second_bracket_term)
    return c_1_plus_plus_A_unp

def i_c_unp_0p_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    first_bracket_term = (2. - lepton_energy_fraction_y)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * x_Bjorken + (epsilon**2 / 4.)) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
    second_bracket_term = y_quantity * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared) * (epsilon**2 - 2. * (1. + (epsilon**2 / (2. * x_Bjorken))) * x_Bjorken * t_over_Q_squared) / root_one_plus_epsilon_squared
    prefactor = 8. * tf.sqrt(2. * y_quantity) / root_one_plus_epsilon_squared**4
    c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_1_zero_plus_unp

def i_c_unp_v_0p_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    k_tilde: float):
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    major_part = (2 - lepton_energy_fraction_y)**2 * k_tilde**2 / squared_Q_momentum_transfer + (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)**2 * y_quantity
    prefactor = 16. * tf.sqrt(2. * y_quantity) * x_Bjorken * t_over_Q_squared / (1. + epsilon**2)**2.5
    c_1_zero_plus_V_unp = prefactor * major_part
    return c_1_zero_plus_V_unp

def i_c_unp_a_0p_1(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    one_minus_2xb = 1. - 2. * x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    second_term_first_part = (1. - one_minus_2xb * t_over_Q_squared) * y_quantity
    second_term_second_part = 4. - 2. * x_Bjorken + 3. * epsilon**2 + t_over_Q_squared * (4. * x_Bjorken * (1. - x_Bjorken) + epsilon**2)
    first_term = k_tilde**2 * one_minus_2xb * (2. - lepton_energy_fraction_y)**2 / squared_Q_momentum_transfer
    prefactor = 8. * tf.sqrt(2. * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_1_zero_plus_unp_A = prefactor * (first_term + second_term_first_part * second_term_second_part)
    return c_1_zero_plus_unp_A

def i_c_unp_pp_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    first_bracket_term = 2. * epsilon**2 * k_tilde**2 / (root_one_plus_epsilon_squared * (1. + root_one_plus_epsilon_squared) * squared_Q_momentum_transfer)
    second_bracket_term = x_Bjorken * t_prime * t_over_Q_squared * (1. - x_Bjorken - 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * epsilon**2 / x_Bjorken) / squared_Q_momentum_transfer
    prefactor = 8. * (2. - lepton_energy_fraction_y) * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) / root_one_plus_epsilon_squared**4
    c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)
    return c_2_plus_plus_unp

def i_c_unp_v_pp_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    major_term = (4. * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)) + 0.5 * (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * (1. + t_over_Q_squared) * t_prime_over_Q_squared
    prefactor = 8. * (2. - lepton_energy_fraction_y) * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_V_unp = prefactor * major_term
    return c_2_plus_plus_V_unp

def i_c_unp_a_pp_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    first_bracket_term = 4. * (1. - 2. * x_Bjorken) * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)
    second_bracket_term = (3.  - root_one_plus_epsilon_squared - 2. * x_Bjorken + epsilon**2 / x_Bjorken ) * x_Bjorken * t_prime_over_Q_squared
    prefactor = 4. * (2. - lepton_energy_fraction_y) * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)
    return c_2_plus_plus_A_unp

def i_c_unp_0p_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    epsilon_squared_over_2 = epsilon**2 / 2.
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / x_Bjorken) / (1. + epsilon_squared_over_2)) * x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    prefactor = -8. * tf.sqrt(2. * y_quantity) * shorthand_k * (2. - lepton_energy_fraction_y) / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return c_2_zero_plus_unp

def i_c_unp_v_0p_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    y_quantity = tf.sqrt(1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * y_quantity * shorthand_k * (2. - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)
    return c_2_zero_plus_unp_V

def i_c_unp_a_0p_2(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    t_prime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    bracket_term = one_minus_xb + 0.5 * t_prime_over_Q_squared * (4. * x_Bjorken * one_minus_xb + epsilon**2) / root_one_plus_epsilon_squared
    prefactor = 8. * tf.sqrt(2. * y_quantity) * shorthand_k * (2. - lepton_energy_fraction_y) * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * bracket_term
    return c_2_zero_plus_unp_A

def i_c_unp_pp_3(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    major_term = (1. - x_Bjorken) * t_over_Q_squared + 0.5 * (root_one_plus_epsilon_squared - 1.) * (1. + t_over_Q_squared)
    intermediate_term = (root_one_plus_epsilon_squared - 1.) / root_one_plus_epsilon_squared**5
    prefactor = -8. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    c_3_plus_plus_unp = prefactor * intermediate_term * major_term
    return c_3_plus_plus_unp

def i_c_unp_v_pp_3(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    major_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * t_over_Q_squared
    prefactor = -8. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_3_plus_plus_V_unp = prefactor * major_term
    return c_3_plus_plus_V_unp

def i_c_unp_a_pp_3(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    main_term = squared_hadronic_momentum_transfer_t * t_prime * (x_Bjorken * (1. - x_Bjorken) + epsilon**2 / 4.) / squared_Q_momentum_transfer**2
    prefactor = 16. * shorthand_k * (1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.) / (1. + epsilon**2)**2.5
    c_3_plus_plus_A_unp = prefactor * main_term
    return c_3_plus_plus_A_unp

def i_s_unp_pp_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    bracket_term = 1. + ((1. - x_Bjorken + 0.5 * (root_one_plus_epsilon_squared - 1.)) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
    prefactor = 8. * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp = prefactor * bracket_term
    return s_1_plus_plus_unp

def i_s_unp_v_pp_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    bracket_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * t_over_Q_squared
    prefactor = -8. * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_1_plus_plus_unp_V = prefactor * bracket_term
    return s_1_plus_plus_unp_V

def i_s_unp_a_pp_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float,
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    one_minus_2xb = 1. - 2. * x_Bjorken
    bracket_term = 1. - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = 8. * lepton_helicity * shorthand_k * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * t_over_Q_squared / root_one_plus_epsilon_squared**2
    s_1_plus_plus_unp_A = prefactor * bracket_term
    return s_1_plus_plus_unp_A

def i_s_unp_0p_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    epsilon: float,
    lepton_energy_fraction_y: float,
    k_tilde: float):
    root_one_plus_epsilon_squared = (1. + epsilon**2)**2
    y_quantity = tf.sqrt(1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
    s_1_zero_plus_unp = 8. * tf.sqrt(2.) * lepton_helicity * (2. - lepton_energy_fraction_y) * lepton_energy_fraction_y * y_quantity * k_tilde**2 / (root_one_plus_epsilon_squared * squared_Q_momentum_transfer)
    return s_1_zero_plus_unp

def i_s_unp_v_0p_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float):
    one_plus_epsilon_squared_squared = (1. + epsilon**2)**2
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    bracket_term = 4. * (1. - 2. * x_Bjorken) * t_over_Q_squared * (1. + x_Bjorken * t_over_Q_squared) + epsilon**2 * (1. + t_over_Q_squared)**2
    prefactor = 4. * tf.sqrt(2. * fancy_y_stuff) * lepton_helicity * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * x_Bjorken * t_over_Q_squared / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_V = prefactor * bracket_term
    return s_1_zero_plus_unp_V

def i_s_unp_a_0p_1(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    one_plus_epsilon_squared_squared = (1. + epsilon**2)**2
    fancy_y_stuff = tf.sqrt(1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    prefactor = -8. * tf.sqrt(2.) * lepton_helicity * lepton_energy_fraction_y * (2. - lepton_energy_fraction_y) * (1. - 2. * x_Bjorken) / one_plus_epsilon_squared_squared
    s_1_zero_plus_unp_A = prefactor * fancy_y_stuff * squared_hadronic_momentum_transfer_t * shorthand_k**2 / squared_Q_momentum_transfer
    return s_1_zero_plus_unp_A

def i_s_unp_pp_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    epsilon: float,
    lepton_energy_fraction_y: float,
    t_prime: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    first_bracket_term = (epsilon**2 - x_Bjorken * (root_one_plus_epsilon_squared - 1.)) / (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken)
    second_bracket_term = (2. * x_Bjorken + epsilon**2) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
    prefactor = -4. * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * (1. + root_one_plus_epsilon_squared - 2. * x_Bjorken) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3
    s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)
    return s_2_plus_plus_unp

def i_s_unp_v_pp_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    one_minus_2xb = 1. - 2. * x_Bjorken
    bracket_term = root_one_plus_epsilon_squared - 1. + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared
    parentheses_term = 1. - one_minus_2xb * t_over_Q_squared
    prefactor = -4. * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_V = prefactor * parentheses_term * bracket_term
    return s_2_plus_plus_unp_V

def i_s_unp_a_pp_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    t_prime: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    tPrime_over_Q_squared = t_prime / squared_Q_momentum_transfer
    fancy_y_stuff = 1. - lepton_energy_fraction_y - epsilon**2 * lepton_energy_fraction_y**2 / 4.
    last_term = 1. + (4. * (1. - x_Bjorken) * x_Bjorken + epsilon**2) * t_over_Q_squared / (4. - 2. * x_Bjorken + 3. * epsilon**2)
    middle_term = 1. + root_one_plus_epsilon_squared - 2. * x_Bjorken
    prefactor = -8. * lepton_helicity * fancy_y_stuff * lepton_energy_fraction_y * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_plus_plus_unp_A = prefactor * middle_term * last_term
    return s_2_plus_plus_unp_A

def i_s_unp_0p_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    epsilon_squared_over_2 = epsilon**2 / 2.
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    bracket_term = 1. + ((1. + epsilon_squared_over_2 / x_Bjorken) / (1. + epsilon_squared_over_2)) * x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    prefactor = 8. * lepton_helicity * tf.sqrt(2. * y_quantity) * shorthand_k * lepton_energy_fraction_y / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term
    return s_2_zero_plus_unp

def i_s_unp_v_0p_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    y_quantity = tf.sqrt(1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.))
    prefactor = -8. * tf.sqrt(2.) * lepton_helicity * y_quantity * shorthand_k * lepton_energy_fraction_y * x_Bjorken * t_over_Q_squared / root_one_plus_epsilon_squared**4
    s_2_zero_plus_unp_V = prefactor * (1. - (1. - 2. * x_Bjorken) * t_over_Q_squared)
    return s_2_zero_plus_unp_V

def i_s_unp_a_0p_2(
    lepton_helicity: float,
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float, 
    squared_hadronic_momentum_transfer_t: float,
    epsilon: float,
    lepton_energy_fraction_y: float, 
    shorthand_k: float):
    root_one_plus_epsilon_squared = tf.sqrt(1. + epsilon**2)
    t_over_Q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer
    one_minus_xb = 1. - x_Bjorken
    y_quantity = 1. - lepton_energy_fraction_y - (epsilon**2 * lepton_energy_fraction_y**2 / 4.)
    main_term = 4. * one_minus_xb + 2. * epsilon**2 + 4. * t_over_Q_squared * (4. * x_Bjorken * one_minus_xb + epsilon**2)
    prefactor = -2. * tf.sqrt(2. * y_quantity) * lepton_helicity * shorthand_k * lepton_energy_fraction_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
    c_2_zero_plus_unp_A = prefactor * main_term
    return c_2_zero_plus_unp_A

def i_curly_c_unp(
    squared_Q_momentum_transfer: float,
    x_Bjorken: float,
    squared_hadronic_momentum_transfer_t: float,
    Dirac_form_factor_F1: float,
    Pauli_form_factor_F2: float,
    cff_h: float,
    cff_h_tilde: float,
    cff_e: float) -> float:
    weighted_cffs = (Dirac_form_factor_F1 * cff_h) - (squared_hadronic_momentum_transfer_t * Pauli_form_factor_F2 * cff_e / (4. * _MASS_OF_PROTON_IN_GEV**2))
    second_term = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) * cff_h_tilde / (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
    curly_C_unpolarized_interference = weighted_cffs + second_term
    return curly_C_unpolarized_interference

def i_curly_c_v_unp(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float,
    squared_hadronic_momentum_transfer_t: float,
    Dirac_form_factor_F1: float,
    Pauli_form_factor_F2: float,
    cff_h: float,
    cff_e: float) -> float:
    cff_term = cff_h + cff_e
    second_term = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) / (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
    curly_C_unpolarized_interference_V = cff_term * second_term
    return curly_C_unpolarized_interference_V

def i_curly_c_a_unp(
    squared_Q_momentum_transfer: float, 
    x_Bjorken: float,
    squared_hadronic_momentum_transfer_t: float,
    Dirac_form_factor_F1: float,
    Pauli_form_factor_F2: float,
    cff_ht: float) -> float:
    xb_modulation = x_Bjorken * (Dirac_form_factor_F1 + Pauli_form_factor_F2) / (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))
    curly_C_unpolarized_interference_A = cff_ht * xb_modulation
    return curly_C_unpolarized_interference_A

def i_c_lp_pp_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq 
    first_bracket_term = (2. - y)**2 * k_tilde**2 / q_sq
    second_bracket_term_first_part = 1. - y + (ep**2 * y**2 / 4.)
    second_bracket_term_second_part = xb * t_over_Q_squared - (ep**2 * (1. - t_over_Q_squared) / 2.)
    second_bracket_term_third_part = 1. + t_over_Q_squared * ((root_one_plus_epsilon_squared - 1. + 2. * xb) / (1. + root_one_plus_epsilon_squared))
    second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part
    prefactor = -4. * lep_helicity * target_polar * y * (1. + root_one_plus_epsilon_squared) / root_one_plus_epsilon_squared**5
    c_0_plus_plus_LP = prefactor * (first_bracket_term + second_bracket_term)
    return c_0_plus_plus_LP

def i_c_lp_v_pp_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_term = (2. - y)**2 * (one_plus_root_epsilon_stuff - 2. * xb) * k_tilde**2 / (q_sq * one_plus_root_epsilon_stuff)
    second_bracket_term_first_part = 1. - y - (ep**2 * y**2 / 4.)
    second_bracket_term_second_part = 2. - xb + 3. * ep**2 / 2
    second_bracket_term_third_part = 1. + (t_over_Q_squared * (4. * (1. - xb) * xb + ep**2) / (4. - 2. * xb + 3. * ep**2))
    second_bracket_term_fourth_part = 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. + 2. * xb) / one_plus_root_epsilon_stuff)
    second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part * second_bracket_term_fourth_part
    prefactor = 4. * lep_helicity * target_polar * y * one_plus_root_epsilon_stuff * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_0_plus_plus_V_LP = prefactor * (first_bracket_term + second_bracket_term)
    return c_0_plus_plus_V_LP

def i_c_lp_a_pp_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_bracket_term = 2. * (2. - y)**2 * k_tilde**2 / q_sq
    second_bracket_term_first_part = 1. - y - (ep**2 * y**2 / 4.)
    second_bracket_term_second_part = 1. - (1. - 2. * xb) * t_over_Q_squared
    second_bracket_term_third_part = 1. + (t_over_Q_squared * (root_one_plus_epsilon_squared - 1. + 2. * xb) / one_plus_root_epsilon_stuff)
    second_bracket_term = second_bracket_term_first_part * one_plus_root_epsilon_stuff * second_bracket_term_second_part * second_bracket_term_third_part
    prefactor = 4. * lep_helicity * target_polar * y * xb * t_over_Q_squared / root_one_plus_epsilon_squared**5
    c_0_plus_plus_A_LP = prefactor * (first_bracket_term + second_bracket_term)
    return c_0_plus_plus_A_LP

def i_c_lp_0p_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * lep_helicity * target_polar * shorthand_k * (1. - xb) * y / (1. + ep**2)**2
    c_0_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * t / q_sq
    return c_0_zero_plus_LP

def i_c_lp_v_0p_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    modulating_factor = (xb - (t * (1. - 2. * xb) / q_sq)) / (1. - xb)
    c_0_zero_plus_LP = i_c_lp_0p_0(
        lep_helicity,
        target_polar,
        q_sq, 
        xb, 
        t,
        ep,
        y, 
        shorthand_k)
    c_0_zero_plus_V_LP = c_0_zero_plus_LP * modulating_factor
    return c_0_zero_plus_V_LP

def i_c_lp_a_0p_0(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * tf.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    t_over_Q_squared = t / q_sq
    c_0_zero_plus_A_LP = prefactor * root_combination_of_y_and_epsilon * xb * t_over_Q_squared * (1. + t_over_Q_squared)
    return c_0_zero_plus_A_LP

def i_c_lp_0p_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    ep: float,
    y: float, 
    k_tilde: float,
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * tf.sqrt(2.) * lep_helicity * target_polar * shorthand_k * (1. - y) * y / (1. + ep**2)**2
    c_1_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * k_tilde**2 / q_sq
    return c_1_zero_plus_LP

def i_c_lp_v_0p_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * lep_helicity * target_polar  * (2. - y) * y / (1. + ep**2)**2
    c_1_zero_plus_V_LP = prefactor * root_combination_of_y_and_epsilon * t * k_tilde**2 / q_sq**2
    return c_1_zero_plus_V_LP

def i_c_lp_pp_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    one_plus_root_epsilon_minus_epsilon_squared = one_plus_root_epsilon_stuff - ep**2
    major_factor = 1. - ((t / q_sq) * (1. - 2. * xb * (one_plus_root_epsilon_stuff + 1.) / one_plus_root_epsilon_minus_epsilon_squared))
    prefactor = -4. * lep_helicity * target_polar * y * shorthand_k * (2. - y) / root_one_plus_epsilon_squared**5
    c_1_plus_plus_LP = prefactor * one_plus_root_epsilon_minus_epsilon_squared * major_factor
    return c_1_plus_plus_LP

def i_c_lp_v_pp_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    one_minus_xb = 1. - xb
    root_epsilon_and_xb_quantity = root_one_plus_epsilon_squared + 2. * one_minus_xb
    bracket_factor_numerator = 1. + ((1. - ep**2) / root_one_plus_epsilon_squared) - (2. * xb * (1. + (4. * one_minus_xb / root_one_plus_epsilon_squared)))
    bracket_factor_denominator = 2. * root_epsilon_and_xb_quantity
    bracket_factor = 1. - (t_prime * bracket_factor_numerator / (q_sq * bracket_factor_denominator))
    prefactor = 8. * lep_helicity * target_polar * shorthand_k * y * (2. - y) / root_one_plus_epsilon_squared**4
    c_1_plus_plus_V_LP = prefactor * root_epsilon_and_xb_quantity * t * bracket_factor / q_sq
    return c_1_plus_plus_V_LP
    
def i_c_lp_a_pp_1(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    t_over_Q_squared = t / q_sq
    major_factor = xb * t_over_Q_squared * (1. - (1. - 2. * xb) * t_over_Q_squared)
    prefactor = 16. * lep_helicity * target_polar * shorthand_k * y * (2. - y) / tf.sqrt(1. + ep**2)**5
    c_1_plus_plus_A_LP = prefactor * major_factor
    return c_1_plus_plus_A_LP

def i_c_lp_pp_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_multiplicative_factor = (-1. * one_plus_root_epsilon_stuff + 2.) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb)
    second_multiplicative_factor = xb * t_over_Q_squared - (ep**2 * (1. - t_over_Q_squared) / 2.)
    prefactor = -4. * lep_helicity * target_polar * y * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
    c_2_plus_plus_LP = prefactor * first_multiplicative_factor * second_multiplicative_factor
    return c_2_plus_plus_LP

def i_c_lp_v_pp_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_multiplicative_factor = (one_plus_root_epsilon_stuff - 2.) + t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb)
    second_multiplicative_factor = 1. + (t_over_Q_squared * (4. * (1. - xb) * xb + ep**2 ) / (4. - 2. * xb + 3. * ep**2))
    third_multiplicative_factor = t_over_Q_squared * (4. - 2. * xb + 3. * ep**2)
    prefactor = -2.*lep_helicity*target_polar*y*(1.-y-(y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
    c_2_plus_plus_V_LP = prefactor * first_multiplicative_factor * second_multiplicative_factor * third_multiplicative_factor
    return c_2_plus_plus_V_LP

def i_c_lp_a_pp_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    first_multiplicative_factor = (1. - root_one_plus_epsilon_squared) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb)
    second_multiplicative_factor = xb * t_over_Q_squared * (1. - t_over_Q_squared * (1. - 2. * xb))
    prefactor = 4. * lep_helicity * target_polar * y * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**5
    c_2_plus_plus_A_LP = prefactor * first_multiplicative_factor * second_multiplicative_factor
    return c_2_plus_plus_A_LP

def i_c_lp_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * tf.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    c_2_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * (1. + (xb * t / q_sq))
    return c_2_zero_plus_LP

def i_c_lp_v_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    c_2_zero_plus_V_LP = prefactor * root_combination_of_y_and_epsilon * (1. - xb ) * t / q_sq
    return c_2_zero_plus_V_LP

def i_c_lp_a_0p_2(
    lep_helicity: float,
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * lep_helicity * target_polar * shorthand_k * y / (1. + ep**2)**2
    t_over_Q_squared = t / q_sq
    c_2_zero_plus_A_LP = prefactor * root_combination_of_y_and_epsilon * xb * t_over_Q_squared * (1. + t / q_sq)
    return c_2_zero_plus_A_LP

def i_s_lp_pp_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    t_over_Q_squared = t / q_sq
    epsilon_y_over_2_squared = (ep * y / 2.) ** 2
    first_bracket_term = 2. * root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb) / one_plus_root_epsilon_stuff)
    second_bracket_term = (3. * ep**2 / 2.) + (t_over_Q_squared * (1. - root_one_plus_epsilon_squared - ep**2 / 2. - xb * (3.  - root_one_plus_epsilon_squared)))
    almost_prefactor = 4. * target_polar * shorthand_k / root_one_plus_epsilon_squared**6
    prefactor_one = almost_prefactor * (2. - 2. * y + y**2 + 2. * epsilon_y_over_2_squared) * one_plus_root_epsilon_stuff
    prefactor_two = 2. * almost_prefactor * (1. - y - epsilon_y_over_2_squared)
    s_1_plus_plus_LP = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term
    return s_1_plus_plus_LP

def i_s_lp_v_pp_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    shorthand_k: float) -> float:
    ep_squared = ep**2
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep_squared)
    t_over_Q_squared = t / q_sq
    t_prime_over_Q_squared = t_prime / q_sq
    epsilon_y_over_2_squared = ep_squared * y**2 / 4.
    first_bracket_term = 1. - (t_prime_over_Q_squared * ((1. - 2. * xb) * (1. - 2. * xb + root_one_plus_epsilon_squared)) / (2. * root_one_plus_epsilon_squared**2))
    second_term_parentheses_term = t_over_Q_squared * (1. - (xb * ((3. + root_one_plus_epsilon_squared) / 4.)) + (5. * ep_squared / 8.))
    second_bracket_term_numerator = 1. - root_one_plus_epsilon_squared + (ep_squared / 2.) - (2. * xb * (3. * (1. - xb) - root_one_plus_epsilon_squared))
    second_bracket_term_denominator = 4. - (xb * (root_one_plus_epsilon_squared + 3.)) + (5. * ep_squared / 2.)
    second_bracket_term = 1. - (t_over_Q_squared * second_bracket_term_numerator / second_bracket_term_denominator)
    almost_prefactor = 8. * target_polar * shorthand_k / root_one_plus_epsilon_squared**4
    prefactor_one = almost_prefactor * (2. - 2. * y + y**2 + 2. * epsilon_y_over_2_squared) * t_over_Q_squared
    prefactor_two = 4. * almost_prefactor * (1. - y - epsilon_y_over_2_squared) / root_one_plus_epsilon_squared**2
    s_1_plus_plus_V_LP = prefactor_one * first_bracket_term + prefactor_two * second_term_parentheses_term * second_bracket_term
    return s_1_plus_plus_V_LP

def i_s_lp_a_pp_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    t_over_Q_squared = t / q_sq
    xB_t_over_Q_squared = xb * t_over_Q_squared
    three_plus_root_epsilon_stuff = 3 + root_one_plus_epsilon_squared
    epsilon_y_over_2_squared = (ep * y / 2.) ** 2
    almost_prefactor = 8. * target_polar * shorthand_k / root_one_plus_epsilon_squared**6
    first_bracket_term = root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (1. + root_one_plus_epsilon_squared - 2. * xb))
    second_bracket_term = 1. - (t_over_Q_squared * (3.  - root_one_plus_epsilon_squared - 6. * xb) / three_plus_root_epsilon_stuff)
    prefactor_one = -1. * almost_prefactor * (2. - 2. * y + y**2 + 2. * epsilon_y_over_2_squared) * xB_t_over_Q_squared
    prefactor_two = almost_prefactor * (1. - y - epsilon_y_over_2_squared) * three_plus_root_epsilon_stuff * xB_t_over_Q_squared
    s_1_plus_plus_A_LP = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term
    return s_1_plus_plus_A_LP

def i_s_lp_0p_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    combination_of_y_and_epsilon = 1. - y - (y**2 * ep**2 / 4.)
    t_over_Q_squared = t / q_sq
    first_bracket_term = k_tilde**2 * (2. - y)**2 / q_sq
    second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * (2. * xb * t_over_Q_squared - (ep**2 * (1. - t_over_Q_squared)))
    prefactor = 8. * tf.sqrt(2.) * target_polar  * tf.sqrt(combination_of_y_and_epsilon) / tf.sqrt((1. + ep**2)**5)
    s_1_zero_plus_LP = prefactor * (first_bracket_term + second_bracket_term)
    return s_1_zero_plus_LP

def i_s_lp_v_0p_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k_tilde: float) -> float:
    combination_of_y_and_epsilon = 1. - y - (y**2 * ep**2 / 4.)
    t_over_Q_squared = t / q_sq
    first_bracket_term = k_tilde**2 * (2. - y)**2 / q_sq
    second_bracket_term_long = 4. - 2. * xb + 3. * ep**2 + t_over_Q_squared * (4. * xb * (1. - xb) + ep**2)
    second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * second_bracket_term_long
    prefactor = -8. * tf.sqrt(2.) * target_polar  * tf.sqrt(combination_of_y_and_epsilon) * t_over_Q_squared / tf.sqrt((1. + ep**2)**5)
    s_1_zero_plus_V_LP = prefactor * (first_bracket_term + second_bracket_term)
    return s_1_zero_plus_V_LP

def i_s_lp_a_0p_1(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float) -> float:
    combination_of_y_and_epsilon_to_3_halves = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))**3
    t_over_Q_squared = t / q_sq
    prefactor = -16. * tf.sqrt(2.) * target_polar * xb * t_over_Q_squared * (1. + t_over_Q_squared) / tf.sqrt((1. + ep**2)**5)
    s_1_zero_plus_A_LP = prefactor * combination_of_y_and_epsilon_to_3_halves * (1. - (1. - 2. * xb) * t_over_Q_squared)
    return s_1_zero_plus_A_LP

def i_s_lp_pp_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    bracket_term = 4. * k_tilde**2 * (one_plus_root_epsilon_stuff - 2. * xb) * (one_plus_root_epsilon_stuff + xb * t / q_sq) * t_prime / (root_one_plus_epsilon_squared * q_sq**2)
    prefactor = -4. * target_polar * (2. - y) * (1. - y - (ep**2 * y**2 / 4.)) / root_one_plus_epsilon_squared**5
    s_2_plus_plus_LP = prefactor * bracket_term
    return s_2_plus_plus_LP

def i_s_lp_v_pp_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    t_prime: float,
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    bracket_term_second_term = (3.  - root_one_plus_epsilon_squared - (2. * xb) + (ep**2 / xb)) * xb * t_prime / q_sq
    bracket_term_first_term = 4. * k_tilde**2 * (1. - 2. * xb) / (root_one_plus_epsilon_squared * q_sq)
    bracket_term = t * (bracket_term_first_term - bracket_term_second_term) / q_sq
    prefactor = 4. * target_polar * (2. - y) * (1. - y - ep**2 * y**2 / 4.) / root_one_plus_epsilon_squared**5
    s_2_plus_plus_V_LP = prefactor * bracket_term
    return s_2_plus_plus_V_LP

def i_s_lp_a_pp_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    k_tilde: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    bracket_term_first_term = (1. + root_one_plus_epsilon_squared - 2. * xb) * (1. - ((1. - 2. * xb) * t / q_sq)) * t_prime / q_sq
    bracket_term_second_term = 4. * k_tilde**2 / q_sq
    bracket_term = xb * t * (bracket_term_second_term - bracket_term_first_term) / q_sq
    prefactor = 4. * target_polar * (2. - y) * (1. - y - ep**2 * y**2 / 4.) / root_one_plus_epsilon_squared**5
    s_2_plus_plus_A_LP = prefactor * bracket_term
    return s_2_plus_plus_A_LP

def i_s_lp_0p_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = 8. * tf.sqrt(2.) * target_polar * k * (2. - y )/ tf.sqrt((1. + ep**2)**5)
    s_2_zero_plus_LP = prefactor * root_combination_of_y_and_epsilon * (1. + (xb * t / q_sq))
    return s_2_zero_plus_LP

def i_s_lp_v_0p_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    prefactor = -8. * tf.sqrt(2.) * target_polar * shorthand_k * (2. - y) * t / (tf.sqrt((1. + ep**2)**5) * q_sq)
    s_2_zero_plus_V_LP = prefactor * (1. - xb) * root_combination_of_y_and_epsilon
    return s_2_zero_plus_V_LP

def i_s_lp_a_0p_2(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float, 
    shorthand_k: float) -> float:
    root_combination_of_y_and_epsilon = tf.sqrt(1. - y - (y**2 * ep**2 / 4.))
    t_over_Q_squared = t / q_sq
    prefactor = -8. * tf.sqrt(2.) * target_polar  * shorthand_k * (2. - y) * xb * t_over_Q_squared / tf.sqrt((1. + ep**2)**5)
    s_2_zero_plus_A_LP = prefactor * root_combination_of_y_and_epsilon * (1. + t_over_Q_squared)
    return s_2_zero_plus_A_LP

def i_s_lp_pp_3(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
    prefactor = -4. * target_polar * shorthand_k * (1. - y - y**2 * ep**2 / 4.) / root_one_plus_epsilon_squared**6
    s_3_plus_plus_LP = prefactor * (one_plus_root_epsilon_stuff - 2. * xb) * ep**2 * t_prime / (q_sq * one_plus_root_epsilon_stuff)
    return s_3_plus_plus_LP

def i_s_lp_v_pp_3(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    multiplicative_contribution = t * t_prime * (4. * (1. - xb) * xb + ep**2) / q_sq**2
    prefactor = 4. * target_polar * shorthand_k * (1. - y - y**2 * ep**2 / 4.) / root_one_plus_epsilon_squared**6
    s_3_plus_plus_V_LP = prefactor * multiplicative_contribution
    return s_3_plus_plus_V_LP
    
def i_s_lp_a_pp_3(
    target_polar: float,
    q_sq: float, 
    xb: float, 
    t: float,
    ep: float,
    y: float,
    t_prime: float,
    shorthand_k: float) -> float:
    root_one_plus_epsilon_squared = tf.sqrt(1. + ep**2)
    multiplicative_contribution = xb * t * t_prime * (1. + root_one_plus_epsilon_squared - 2. * xb) / q_sq**2
    prefactor = -8. * target_polar * shorthand_k * (1. - y - (y**2 * ep**2 / 4.)) / root_one_plus_epsilon_squared**6
    s_3_plus_plus_A_LP = prefactor * multiplicative_contribution
    return s_3_plus_plus_A_LP

def i_curly_c_lp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float,
    cff_ht: float,
    cff_e: float,
    cff_et: float) -> float:
    t_over_Q_squared = t / q_sq
    ratio_of_xb_to_more_xb = xb / (2. - xb + xb * t_over_Q_squared)
    x_Bjorken_correction = xb * (1. - t_over_Q_squared) / 2.
    first_cff_contribution = ratio_of_xb_to_more_xb * (f1 + f2) * (cff_h + x_Bjorken_correction * cff_e)
    second_cff_contribution = (1. + (_MASS_OF_PROTON_IN_GEV**2 * xb * ratio_of_xb_to_more_xb * (3. + t_over_Q_squared) / q_sq)) * f1 * cff_ht
    third_cff_contribution = t_over_Q_squared * 2. * (1. - 2. * xb) * ratio_of_xb_to_more_xb * f2 * cff_ht
    fourth_cff_contribution = ratio_of_xb_to_more_xb * (x_Bjorken_correction * f1 + t * f2 / (4. * _MASS_OF_PROTON_IN_GEV**2)) * cff_et
    curly_C_longitudinally_polarized_interference = first_cff_contribution + second_cff_contribution - third_cff_contribution - fourth_cff_contribution
    return curly_C_longitudinally_polarized_interference

def i_curly_c_v_lp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_h: float,
    cff_e: float) -> float:
    t_over_Q_squared = t / q_sq
    ratio_of_xb_to_more_xb = xb / (2. - xb + xb * t_over_Q_squared)
    sum_of_form_factors = f1 + f2
    curly_C_V_longitudinally_polarized_interference = ratio_of_xb_to_more_xb * sum_of_form_factors * (cff_h + (xb * (1. - t_over_Q_squared) * cff_e / 2.))
    return curly_C_V_longitudinally_polarized_interference

def i_curly_c_a_lp(
    q_sq: float, 
    xb: float,
    t: float,
    f1: float,
    f2: float,
    cff_ht: float,
    cff_et: float) -> float:
    t_over_Q_squared = t / q_sq
    ratio_of_xb_to_more_xb = xb / (2. - xb + xb * t_over_Q_squared)
    sum_of_form_factors = f1 + f2
    cff_appearance = cff_ht * (1. + (2. * xb * _MASS_OF_PROTON_IN_GEV**2 / q_sq)) + (xb * cff_et / 2.)
    curly_C_A_longitudinally_polarized_interference = ratio_of_xb_to_more_xb * sum_of_form_factors * cff_appearance
    return curly_C_A_longitudinally_polarized_interference

def i_unp_c0(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_0 = i_c_unp_pp_0(q_sq, xb, t, ep, y, ktilde)
    i_c_pp_v_0 = i_c_unp_v_pp_0(q_sq, xb, t, ep, y, ktilde)
    i_c_pp_a_0 = i_c_unp_a_pp_0(q_sq, xb, t, ep, y, ktilde)

    i_c_0p_0 = i_c_unp_0p_0(q_sq, xb, t, ep, y, k)
    i_c_0p_v_0 = i_c_unp_v_0p_0(q_sq, xb, t, ep, y, k)
    i_c_0p_a_0 = i_c_unp_a_0p_0(q_sq, xb, t, ep, y, k)
    
    return (i_c_pp_0*i_curly_c + i_c_pp_v_0*i_curly_c_v + i_c_pp_a_0*i_curly_c_a + i_c_0p_0*i_curly_c_eff + i_c_0p_v_0*i_curly_c_eff_v + i_c_0p_a_0*i_curly_c_eff_a)

def i_unp_c1(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_1 = i_c_unp_pp_1(q_sq, xb, t, ep, y, k)
    i_c_pp_v_1 = i_c_unp_v_pp_1(q_sq, xb, t, ep, y, tprime, k)
    i_c_pp_a_1 = i_c_unp_a_pp_1(q_sq, xb, t, ep, y, tprime, k)

    i_c_0p_1 = i_c_unp_0p_1(q_sq, xb, t, ep, y, tprime)
    i_c_0p_v_1 = i_c_unp_v_0p_1(q_sq, xb, t, ep, y, ktilde)
    i_c_0p_a_1 = i_c_unp_a_0p_1(q_sq, xb, t, ep, y, ktilde)

    return (i_c_pp_1*i_curly_c + i_c_pp_v_1*i_curly_c_v + i_c_pp_a_1*i_curly_c_a + i_c_0p_1*i_curly_c_eff + i_c_0p_v_1*i_curly_c_eff_v + i_c_0p_a_1*i_curly_c_eff_a)

def i_unp_c2(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_2 = i_c_unp_pp_2(q_sq, xb, t, ep, y, tprime, ktilde)
    i_c_pp_v_2 = i_c_unp_v_pp_2(q_sq, xb, t, ep, y, tprime, ktilde)
    i_c_pp_a_2 = i_c_unp_a_pp_2(q_sq, xb, t, ep, y, tprime, ktilde)

    i_c_0p_2 = i_c_unp_0p_2(q_sq, xb, t, ep, y, k)
    i_c_0p_v_2 = i_c_unp_v_0p_2(q_sq, xb, t, ep, y, k)
    i_c_0p_a_2 = i_c_unp_a_0p_2(q_sq, xb, t, ep, y, tprime, k)

    return (i_c_pp_2*i_curly_c + i_c_pp_v_2*i_curly_c_v + i_c_pp_a_2*i_curly_c_a + i_c_0p_2*i_curly_c_eff + i_c_0p_v_2*i_curly_c_eff_v + i_c_0p_a_2*i_curly_c_eff_a)

def i_unp_c3(
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_re_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_3 = i_c_unp_pp_3(q_sq, xb, t, ep, y, k)
    i_c_pp_v_3 = i_c_unp_v_pp_3(q_sq, xb, t, ep, y, k)
    i_c_pp_a_3 = i_c_unp_a_pp_3(q_sq, xb, t, ep, y, tprime, k)

    i_c_0p_3 = 0.
    i_c_0p_v_3 = 0.
    i_c_0p_a_3 = 0.

    return (i_c_pp_3*i_curly_c + i_c_pp_v_3*i_curly_c_v + i_c_pp_a_3*i_curly_c_a + i_c_0p_3*i_curly_c_eff + i_c_0p_v_3*i_curly_c_eff_v + i_c_0p_a_3*i_curly_c_eff_a)

def i_lp_c0(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e, cff_re_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_re_ht, cff_re_et)

    i_curly_c_eff = ktilde*tf.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*tf.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*tf.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_0 = i_c_lp_pp_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, ktilde)
    i_c_pp_v_0 = i_c_lp_v_pp_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, ktilde)
    i_c_pp_a_0 = i_c_lp_a_pp_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, ktilde)

    i_c_0p_0 = i_c_lp_0p_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_v_0 = i_c_lp_v_0p_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_a_0 = i_c_lp_a_0p_0(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_c_pp_0*i_curly_c + i_c_pp_v_0*i_curly_c_v + i_c_pp_a_0*i_curly_c_a + i_c_0p_0*i_curly_c_eff + i_c_0p_v_0*i_curly_c_eff_v + i_c_0p_a_0*i_curly_c_eff_a)

def i_lp_c1(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e, cff_re_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_re_ht, cff_re_et)

    i_curly_c_eff = ktilde*tf.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*tf.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*tf.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_1 = i_c_lp_pp_1(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_pp_v_1 = i_c_lp_v_pp_1(lep_helicity, target_polar, q_sq, xb, t, ep, y, tprime, k)
    i_c_pp_a_1 = i_c_lp_a_pp_1(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)

    i_c_0p_1 = i_c_lp_0p_1(lep_helicity, target_polar, q_sq, ep, y, ktilde, k)
    i_c_0p_v_1 = i_c_lp_v_0p_1(lep_helicity, target_polar, q_sq, t, ep, y, ktilde)
    i_c_0p_a_1 = 0.0
    
    return (i_c_pp_1*i_curly_c + i_c_pp_v_1*i_curly_c_v + i_c_pp_a_1*i_curly_c_a + i_c_0p_1*i_curly_c_eff + i_c_0p_v_1*i_curly_c_eff_v + i_c_0p_a_1*i_curly_c_eff_a)

def i_lp_c2(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float,
    f1: float, f2: float, ktilde: float, cff_re_h: float, cff_re_ht: float, cff_re_e: float, cff_re_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_ht, cff_re_e, cff_re_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_re_h, cff_re_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_re_ht, cff_re_et)

    i_curly_c_eff = ktilde*tf.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_e, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*tf.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_h, use_ww), f_eff(xi, cff_re_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*tf.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_re_ht, use_ww), f_eff(xi, cff_re_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_c_pp_2 = i_c_lp_pp_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)
    i_c_pp_v_2 = i_c_lp_v_pp_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)
    i_c_pp_a_2 = i_c_lp_a_pp_2(lep_helicity, target_polar, q_sq, xb, t, ep, y)

    i_c_0p_2 = i_c_lp_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_v_2 = i_c_lp_v_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    i_c_0p_a_2 = i_c_lp_a_0p_2(lep_helicity, target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_c_pp_2*i_curly_c + i_c_pp_v_2*i_curly_c_v + i_c_pp_a_2*i_curly_c_a + i_c_0p_2*i_curly_c_eff + i_c_0p_v_2*i_curly_c_eff_v + i_c_0p_a_2*i_curly_c_eff_a)

def i_unp_s1(
    lep_helicity: float, q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_im_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_1 = i_s_unp_pp_1(lep_helicity, q_sq, xb, ep, y, tprime, k)
    i_s_pp_v_1 = i_s_unp_v_pp_1(lep_helicity, q_sq, xb, t, ep, y, k)
    i_s_pp_a_1 = i_s_unp_a_pp_1(lep_helicity, q_sq, xb, t, ep, y, tprime, k)

    i_s_0p_1 = i_s_unp_0p_1(lep_helicity, q_sq, ep, y, ktilde)
    i_s_0p_v_1 = i_s_unp_v_0p_1(lep_helicity, q_sq, xb, t, ep, y)
    i_s_0p_a_1 = i_s_unp_a_0p_1(lep_helicity, q_sq, xb, t, ep, y, k)

    return (i_s_pp_1*i_curly_c + i_s_pp_v_1*i_curly_c_v + i_s_pp_a_1*i_curly_c_a + i_s_0p_1*i_curly_c_eff + i_s_0p_v_1*i_curly_c_eff_v + i_s_0p_a_1*i_curly_c_eff_a)

def i_unp_s2(
    lep_helicity: float, q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, k: float, tprime: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, use_ww: bool = True):

    i_curly_c = i_curly_c_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e)
    i_curly_c_v = i_curly_c_v_unp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_unp(q_sq, xb, t, f1, f2, cff_im_ht)

    i_curly_c_eff = ktilde * tf.sqrt(2.) * i_curly_c_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde * tf.sqrt(2.) * i_curly_c_v_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde * tf.sqrt(2.) * i_curly_c_a_unp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_2 = i_s_unp_pp_2(lep_helicity, q_sq, xb, ep, y, tprime)
    i_s_pp_v_2 = i_s_unp_v_pp_2(lep_helicity, q_sq, xb, t, ep, y)
    i_s_pp_a_2 = i_s_unp_a_pp_2(lep_helicity, q_sq, xb, t, ep, y, tprime)

    i_s_0p_2 = i_s_unp_0p_2(lep_helicity, q_sq, xb, t, ep, y, k)
    i_s_0p_v_2 = i_s_unp_v_0p_2(lep_helicity, q_sq, xb, t, ep, y, k)
    i_s_0p_a_2 = i_s_unp_a_0p_2(lep_helicity, q_sq, xb, t, ep, y, k)

    return (i_s_pp_2*i_curly_c + i_s_pp_v_2*i_curly_c_v + i_s_pp_a_2*i_curly_c_a + i_s_0p_2*i_curly_c_eff + i_s_0p_v_2*i_curly_c_eff_v + i_s_0p_a_2*i_curly_c_eff_a)

def i_lp_s1(
    target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e, cff_im_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_im_ht, cff_im_et)

    i_curly_c_eff = ktilde*tf.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*tf.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*tf.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_1 = i_s_lp_pp_1(target_polar, q_sq, xb, t, ep, y, k)
    i_s_pp_v_1 = i_s_lp_v_pp_1(target_polar, q_sq, xb, t, ep, y, tprime, k)
    i_s_pp_a_1 = i_s_lp_a_pp_1(target_polar, q_sq, xb, t, ep, y, k)

    i_s_0p_1 = i_s_lp_0p_1(target_polar, q_sq, xb, t, ep, y, ktilde)
    i_s_0p_v_1 = i_s_lp_v_0p_1(target_polar, q_sq, xb, t, ep, y, ktilde)
    i_s_0p_a_1 = i_s_lp_a_0p_1(target_polar, q_sq, xb, t, ep, y)
    
    return (i_s_pp_1*i_curly_c + i_s_pp_v_1*i_curly_c_v + i_s_pp_a_1*i_curly_c_a + i_s_0p_1*i_curly_c_eff + i_s_0p_v_1*i_curly_c_eff_v + i_s_0p_a_1*i_curly_c_eff_a)

def i_lp_s2(
    lep_helicity: float, target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e, cff_im_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_im_ht, cff_im_et)

    i_curly_c_eff = ktilde*tf.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*tf.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*tf.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_2 = i_s_lp_pp_2(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)
    i_s_pp_v_2 = i_s_lp_v_pp_2(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)
    i_s_pp_a_2 = i_s_lp_a_pp_2(target_polar, q_sq, xb, t, ep, y, tprime, ktilde)

    i_s_0p_2 = i_s_lp_0p_2(target_polar, q_sq, xb, t, ep, y, k)
    i_s_0p_v_2 = i_s_lp_v_0p_2(target_polar, q_sq, xb, t, ep, y, k)
    i_s_0p_a_2 = i_s_lp_a_0p_2(target_polar, q_sq, xb, t, ep, y, k)
    
    return (i_s_pp_2*i_curly_c + i_s_pp_v_2*i_curly_c_v + i_s_pp_a_2*i_curly_c_a + i_s_0p_2*i_curly_c_eff + i_s_0p_v_2*i_curly_c_eff_v + i_s_0p_a_2*i_curly_c_eff_a)

def i_lp_s3(
    target_polar: float,
    q_sq: float, xb: float, t: float, ep: float, y: float, xi: float, tprime: float, k: float,
    f1: float, f2: float, ktilde: float, cff_im_h: float, cff_im_ht: float, cff_im_e: float, cff_im_et: float, use_ww: bool = True):

    i_curly_c = i_curly_c_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_ht, cff_im_e, cff_im_et)
    i_curly_c_v = i_curly_c_v_lp(q_sq, xb, t, f1, f2, cff_im_h, cff_im_e)
    i_curly_c_a = i_curly_c_a_lp(q_sq, xb, t, f1, f2, cff_im_ht, cff_im_et)

    i_curly_c_eff = ktilde*tf.sqrt(2.)*i_curly_c_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_e, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_v = ktilde*tf.sqrt(2.)*i_curly_c_v_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_h, use_ww), f_eff(xi, cff_im_e, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))
    i_curly_c_eff_a = ktilde*tf.sqrt(2.)*i_curly_c_a_lp(q_sq, xb, t, f1, f2, f_eff(xi, cff_im_ht, use_ww), f_eff(xi, cff_im_et, use_ww)) / ((2. - xb) * tf.sqrt(q_sq))

    i_s_pp_3 = i_s_lp_pp_3(target_polar, q_sq, xb, ep, y, tprime, k)
    i_s_pp_v_3 = i_s_lp_v_pp_3(target_polar, q_sq, xb, t, ep, y, tprime, k)
    i_s_pp_a_3 = i_s_lp_a_pp_3(target_polar, q_sq, xb, t, ep, y, tprime, k)

    i_s_0p_3 = 0.0
    i_s_0p_v_3 = 0.0
    i_s_0p_a_3 = 0.0
    
    return (i_s_pp_3*i_curly_c + i_s_pp_v_3*i_curly_c_v + i_s_pp_a_3*i_curly_c_a + i_s_0p_3*i_curly_c_eff + i_s_0p_v_3*i_curly_c_eff_v + i_s_0p_a_3*i_curly_c_eff_a)

def interference_amplitude(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    i_c0_unp = i_unp_c0(q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_c1_unp = i_unp_c1(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_c2_unp = i_unp_c2(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_c3_unp = i_unp_c3(q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, use_ww)
    i_s1_unp = i_unp_s1(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    i_s2_unp = i_unp_s2(lep_helicity, q_sq, xb, t, ep, y, xi, k, tprime, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, use_ww)
    i_s3_unp = 0.0

    i_c0_lp = i_lp_c0(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
    i_c1_lp = i_lp_c1(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
    i_c2_lp = i_lp_c2(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, cff_re_h, cff_re_ht, cff_re_e, cff_re_et, use_ww)
    i_c3_lp = 0.5 * 0.0
    i_s1_lp = i_lp_s1(target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    i_s2_lp = i_lp_s2(lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    i_s3_lp = i_lp_s3(target_polar, q_sq, xb, t, ep, y, xi, tprime, k, f1, f2, ktilde, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    return (
        (
            (i_c0_unp + i_c0_lp) * tf.cos(0. * (tf.constant(np.pi) - phi)) +
            (i_c1_unp + i_c1_lp) * tf.cos(1. * (tf.constant(np.pi) - phi)) + 
            (i_c2_unp + i_c2_lp) * tf.cos(2. * (tf.constant(np.pi) - phi)) + 
            (i_c3_unp + i_c3_lp) * tf.cos(3. * (tf.constant(np.pi) - phi)) + 
            (i_s1_unp + i_s1_lp) * tf.sin(1. * (tf.constant(np.pi) - phi)) + 
            (i_s2_unp + i_s2_lp) * tf.sin(2. * (tf.constant(np.pi) - phi)) + 
            (i_s3_unp + i_s3_lp) * tf.sin(3. * (tf.constant(np.pi) - phi))
        )/(xb * y * y * y * t * p1 * p2))

def bkm10_cross_section(
    lep_helicity, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):
    
    bh_plus_beam_plus_target = bh_squared(+1.0, +0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_plus_beam_minus_target = bh_squared(+1.0, -0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_minus_beam_plus_target = bh_squared(-1.0, +0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)
    bh_minus_beam_minus_target = bh_squared(-1.0, -0.5, q_sq, xb, t, ep, y, k, f1, f2, phi, p1, p2)

    dvcs_plus_beam_plus_target = dvcs_squared(
        +1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_plus_beam_minus_target = dvcs_squared(
        +1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_minus_beam_plus_target = dvcs_squared(
        -1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    dvcs_minus_beam_minus_target = dvcs_squared(
        -1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, phi,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    interference_plus_beam_plus_target = interference_amplitude(
        +1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_plus_beam_minus_target = interference_amplitude(
        +1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_minus_beam_plus_target = interference_amplitude(
        -1.0, +0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    interference_minus_beam_minus_target = interference_amplitude(
        -1.0, -0.5,
        q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2, 
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    tf_cross_section = 0.0
    
    if lep_helicity == 0.0:
        if target_polar == 0.0:
            tf_cross_section = 0.25 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + bh_plus_beam_minus_target + bh_minus_beam_plus_target + bh_minus_beam_minus_target +
                    dvcs_plus_beam_plus_target + dvcs_plus_beam_minus_target + dvcs_minus_beam_plus_target + dvcs_minus_beam_minus_target +
                    interference_plus_beam_plus_target + interference_plus_beam_minus_target + interference_minus_beam_plus_target + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
        
        elif target_polar == +0.5:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + 0.0 + bh_minus_beam_plus_target + 0.0 +
                    dvcs_plus_beam_plus_target + 0.0 + dvcs_minus_beam_plus_target + 0.0 +
                    interference_plus_beam_plus_target + 0.0 + interference_minus_beam_plus_target + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
        
        elif target_polar == -0.5:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + bh_plus_beam_minus_target + 0.0 + bh_minus_beam_minus_target +
                    0.0 + dvcs_plus_beam_minus_target + 0.0 + dvcs_minus_beam_minus_target +
                    0.0 + interference_plus_beam_minus_target + 0.0 + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))

    elif lep_helicity == +1.0:
        if target_polar == 0.0:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + bh_plus_beam_minus_target + 0.0 + 0.0 +
                    dvcs_plus_beam_plus_target + dvcs_plus_beam_minus_target + 0.0 + 0.0 +
                    interference_plus_beam_plus_target + interference_plus_beam_minus_target + 0.0 + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
        
        elif target_polar == +0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    bh_plus_beam_plus_target + 0.0 + 0.0 + 0.0 +
                    dvcs_plus_beam_plus_target + 0.0 + 0.0 + 0.0 +
                    interference_plus_beam_plus_target + 0.0 + 0.0 + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
        
        elif target_polar == -0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + bh_plus_beam_minus_target + 0.0 + 0.0 +
                    0.0 + dvcs_plus_beam_minus_target + 0.0 + 0.0 +
                    0.0 + interference_plus_beam_minus_target + 0.0 + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
    
    elif lep_helicity == -1.0:
        if target_polar == 0.0:
            tf_cross_section = 0.5 * (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + 0.0 + bh_minus_beam_plus_target + bh_minus_beam_minus_target +
                    0.0 + 0.0 + dvcs_minus_beam_plus_target + dvcs_minus_beam_minus_target +
                    0.0 + 0.0 + interference_minus_beam_plus_target + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
        elif target_polar == +0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + 0.0 + bh_minus_beam_plus_target + 0.0 +
                    0.0 + 0.0 + dvcs_minus_beam_plus_target + 0.0 +
                    0.0 + 0.0 + interference_minus_beam_plus_target + 0.0) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
        elif target_polar == -0.5:
            tf_cross_section = (
                _CONVERSION_GEV6_GEV4NB*_QED_FINE_STRUCTURE**3*xb*y*y*(
                    0.0 + 0.0 + 0.0 + bh_minus_beam_minus_target +
                    0.0 + 0.0 + 0.0 + dvcs_minus_beam_minus_target +
                    0.0 + 0.0 + 0.0 + interference_minus_beam_minus_target) / (8.*tf.constant(np.pi)*q_sq*q_sq*tf.sqrt(1. + ep**2)))
            
    return tf_cross_section

def bkm10_bsa(
    target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    plus_beam_cross_section = bkm10_cross_section(
        +1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_cross_section = bkm10_cross_section(
        -1.0, target_polar, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
        cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)
    
    tf_bsa = (
        (plus_beam_cross_section - minus_beam_cross_section) / 
        (plus_beam_cross_section + minus_beam_cross_section))
        
    return tf_bsa

def bkm10_tsa(
    lep_helicity, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    plus_beam_plus_lp_target = bkm10_cross_section(
    +1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_plus_lp_target = bkm10_cross_section(
    -1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    plus_beam_minus_lp_target = bkm10_cross_section(
    +1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_minus_lp_target = bkm10_cross_section(
    -1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    if lep_helicity == 0.0:

        sigma_plus = 0.5 * (plus_beam_plus_lp_target + minus_beam_plus_lp_target)
        sigma_minus = 0.5 * (plus_beam_minus_lp_target + minus_beam_minus_lp_target)

    if lep_helicity == 1.0:

        sigma_plus = plus_beam_plus_lp_target
        sigma_minus = plus_beam_minus_lp_target

    if lep_helicity == -1.0:

        sigma_plus = minus_beam_plus_lp_target
        sigma_minus = minus_beam_minus_lp_target

    numerator = sigma_plus - sigma_minus
    denominator = sigma_plus + sigma_minus
    tsa = numerator / denominator
    return tsa

def bkm10_dsa(
    q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww: bool = True):

    plus_beam_plus_lp_target = bkm10_cross_section(
    +1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_plus_lp_target = bkm10_cross_section(
    -1.0, +0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    plus_beam_minus_lp_target = bkm10_cross_section(
    +1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    minus_beam_minus_lp_target = bkm10_cross_section(
    -1.0, -0.5, q_sq, xb, t, ep, y, xi, k, f1, f2, ktilde, tprime, phi, p1, p2,
    cff_re_h, cff_re_ht, cff_re_e, cff_re_et, cff_im_h, cff_im_ht, cff_im_e, cff_im_et, use_ww)

    numerator = ((plus_beam_plus_lp_target-plus_beam_minus_lp_target)-(minus_beam_plus_lp_target-minus_beam_minus_lp_target))
    denominator = plus_beam_plus_lp_target+plus_beam_minus_lp_target+minus_beam_plus_lp_target+minus_beam_minus_lp_target
    dsa = numerator / denominator
    return dsa
