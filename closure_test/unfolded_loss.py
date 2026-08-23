"""
An alternative simultaneous fit loss that is suspected to be
more optimized than the current one.
Created: 20260817
Last changed: 20260822
Notes:
    1.  2026/08/22:
        This only contains the unpolarized coefficients.
    2.  2026/08/23:
        All of the functions are here.
"""

import numpy as np
import tensorflow as tf

class UnfoldedSimultaneousFitLoss(tf.keras.losses.Loss):
    def __init__(self, name = "simultaneous_loss"):
        super().__init__(name = name)

        # debugging parameter:
        self.debugging = False

        self.use_ww = True

        self.gev6_to_gev4_per_nb = tf.constant(.389379 * 1000000.)
        self.mp_sq = tf.constant(0.93827208816)
        self.qed_alpha = tf.constant(1./137.035999177)
        self.fe_constant = tf.constant(0.710649)
        self.mu_proton = tf.constant(2.79284734463)

        self._OBSERVABLE_WEIGHT_1 = 0.5 * 1.0
        self._OBSERVABLE_WEIGHT_2 = 0.5 * 1.0
        self._OBSERVABLE_WEIGHT_3 = 0.0 * 0.5
        self._OBSERVABLE_WEIGHT_4 = 0.0 * 0.5
    
    def call(self, true_values, predicted_values):
        
        # the CFFs:
        cff_h_real_tf = predicted_values[:, 0]
        cff_h_imag_tf = predicted_values[:, 1]
        cff_ht_real_tf = predicted_values[:, 2]
        cff_ht_imag_tf = predicted_values[:, 3]
        cff_e_real_tf = predicted_values[:, 4]
        cff_e_imag_tf = predicted_values[:, 5]
        cff_et_real_tf = predicted_values[:, 6]
        cff_et_imag_tf = predicted_values[:, 7]

        # kinematics:
        t_tf = predicted_values[:, 8]
        xb_tf = predicted_values[:, 9]
        q_sq_tf = predicted_values[:, 10]
        phi_tf = predicted_values[:, 11]

        # derived quantities -> form factors:
        fe_tf = predicted_values[:, 12]
        fg_tf = predicted_values[:, 13]
        f1_tf = predicted_values[:, 14]
        f2_tf = predicted_values[:, 15]

        # derived quantities -> kinematics
        ep_tf = predicted_values[:, 16]
        y_lep_tf = predicted_values[:, 17]
        xi_tf = predicted_values[:, 18]
        tprime_tf = predicted_values[:, 19]
        ktilde_tf = predicted_values[:, 20]
        k_tf = predicted_values[:, 21]

        # derived quantities -> phi-depdendent stuff:
        kdd_tf = predicted_values[:, 22]
        p1_tf = predicted_values[:, 23]
        p2_tf = predicted_values[:, 24]

        # polarizations:
        lep_lambda = predicted_values[:, 25]
        tgt_lambda = predicted_values[:, 26]

        # the effective CFFs (using WW!):
        cff_h_real_eff_tf = 2. * cff_h_real_tf / (1. + xi_tf)
        cff_h_imag_eff_tf = 2. * cff_h_imag_tf / (1. + xi_tf)
        cff_ht_real_eff_tf = 2. * cff_ht_real_tf / (1. + xi_tf)
        cff_ht_imag_eff_tf = 2. * cff_ht_imag_tf / (1. + xi_tf)
        cff_e_real_eff_tf = 2. * cff_e_real_tf / (1. + xi_tf)
        cff_e_imag_eff_tf = 2. * cff_e_imag_tf / (1. + xi_tf)
        cff_et_real_eff_tf = 2. * cff_et_real_tf / (1. + xi_tf)
        cff_et_imag_eff_tf = 2. * cff_et_imag_tf / (1. + xi_tf)

        # observables:
        true_cross_section = true_values[:, 0]
        true_bsa = true_values[:, 1]

        # BH: c0
        first_line = 8. * k_tf**2 * (((2. + 3. * ep_tf**2) * (f1_tf**2 - (t_tf * f2_tf**2 / (4. * self.mp_sq**2))) / (t_tf / q_sq_tf)) + (2. * xb_tf**2 * (f1_tf + f2_tf)**2))
        second_line_first_part = (2. + ep_tf**2) * ((4. * xb_tf**2 * self.mp_sq**2 / t_tf) * (1. + (t_tf / q_sq_tf))**2 + 4. * (1 - xb_tf) * (1. + (xb_tf * (t_tf / q_sq_tf)))) * (f1_tf**2 - (t_tf * f2_tf**2 / (4. * self.mp_sq**2)))
        second_line_second_part = 4. * xb_tf**2 * (xb_tf + (1. - xb_tf + (ep_tf**2 / 2.)) * (1 - (t_tf / q_sq_tf))**2 - xb_tf * (1. - 2. * xb_tf) * (t_tf / q_sq_tf)**2) * (f1_tf + f2_tf)**2
        second_line = (2. - y_lep_tf)**2 * (second_line_first_part + second_line_second_part)
        third_line = 8. * (1. + ep_tf**2) * (1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)) * (2. * ep_tf**2 * (1 - (t_tf / (4. * self.mp_sq**2))) * (f1_tf**2 - (t_tf * f2_tf**2 / (4. * self.mp_sq**2))) - xb_tf**2 * (1 - (t_tf / q_sq_tf))**2 * (f1_tf + f2_tf)**2)
        bh_c0 = first_line + second_line + third_line

        if self.debugging: tf.print(f"[DEBUG]: BH c0 unp: {bh_c0}")

        # BH: c1
        addition_of_form_factors_squared = (f1_tf + f2_tf)**2
        weighted_combination_of_form_factors = f1_tf**2 - ((t_tf / (4. * self.mp_sq**2)) * f2_tf**2)
        first_line_first_part = ((4. * xb_tf**2 * self.mp_sq**2 / t_tf) - 2. * xb_tf - ep_tf**2) * weighted_combination_of_form_factors
        first_line_second_part = 2. * xb_tf**2 * (1. - (1. - 2. * xb_tf) * (t_tf / q_sq_tf)) * addition_of_form_factors_squared
        bh_c1 = 8. * k_tf * (2. - y_lep_tf) * (first_line_first_part + first_line_second_part)

        if self.debugging: tf.print(f"[DEBUG]: BH c1 unp: {bh_c1}")

        # BH: c2
        addition_of_form_factors_squared = (f1_tf + f2_tf)**2
        weighted_combination_of_form_factors = f1_tf**2 - ((t_tf/ (4. * self.mp_sq**2)) * f2_tf**2)
        first_part_of_contribution = (4. * self.mp_sq**2 / t_tf) * weighted_combination_of_form_factors
        bh_c2 = 8. * xb_tf**2 * k_tf**2 * (first_part_of_contribution + 2. * addition_of_form_factors_squared)

        # BH LP: c0
        sum_of_form_factors = (f1_tf + f2_tf)
        t_over_four_mp_squared = t_tf / (4. * self.mp_sq**2)
        weighted_sum_of_form_factors = f1_tf + t_over_four_mp_squared * f2_tf
        one_minus_xb = 1. - xb_tf
        t_over_Q_squared = t_tf / q_sq_tf
        one_minus_t_over_Q_squared = 1. - t_over_Q_squared
        first_term_first_bracket = 0.5 * xb_tf * (one_minus_t_over_Q_squared) - t_over_four_mp_squared
        first_term_second_bracket = 2. - xb_tf - (2. * (one_minus_xb)**2 * t_over_Q_squared) + (ep_tf**2 * one_minus_t_over_Q_squared) - (xb_tf * (1. - 2. * xb_tf) * t_over_Q_squared**2)
        first_term = 0.5 * sum_of_form_factors * first_term_first_bracket * first_term_second_bracket
        second_term_first_bracket = xb_tf**2 * (1. + t_over_Q_squared)**2 / (4. * t_over_four_mp_squared) + ((1. - xb_tf) * (1. + xb_tf * t_over_Q_squared))
        second_term = (1. - (1. - xb_tf) * t_over_Q_squared) * weighted_sum_of_form_factors * second_term_first_bracket
        prefactor = 8. * lep_lambda * tgt_lambda * xb_tf * (2. - y_lep_tf) * y_lep_tf * tf.sqrt(1. + ep_tf**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
        bh_lp_c0 = prefactor * (first_term + second_term)

        # BH LP: c1
        sum_of_form_factors = (f1_tf + f2_tf)
        t_over_four_mp_squared = t_tf / (4. * self.mp_sq**2)
        weighted_sum_of_form_factors = f1_tf + t_over_four_mp_squared * f2_tf
        t_over_Q_squared = t_tf / q_sq_tf
        first_term = ((2. * t_over_four_mp_squared) - (xb_tf * (1. - t_over_Q_squared))) * ((1. - xb_tf + (xb_tf * t_over_Q_squared))) * sum_of_form_factors
        second_term_bracket_term = 1. + xb_tf - ((3. - 2. * xb_tf) * (1. + xb_tf * t_over_Q_squared)) - (xb_tf**2 * (1. + t_over_Q_squared**2) / t_over_four_mp_squared)
        second_term = weighted_sum_of_form_factors * second_term_bracket_term
        prefactor = -8. * lep_lambda * tgt_lambda * xb_tf * y_lep_tf * k_tf * tf.sqrt(1. + ep_tf**2) * sum_of_form_factors / (1. - t_over_four_mp_squared)
        bh_lp_c1 = prefactor * (first_term + second_term)

        # BH LP: c2
        bh_lp_c2 = 0.0

        # DVCS Re[CurlyC](F| F*):
        first_line = (4.*(1.-xb_tf)*(cff_h_real_tf*cff_h_real_tf - cff_h_imag_tf*(-cff_h_imag_tf))) + (4.*(1.-xb_tf + 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_real_tf*cff_ht_real_tf - cff_ht_imag_tf*(-cff_ht_imag_tf)))
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_real_tf*cff_e_real_tf - cff_e_imag_tf*(-cff_h_imag_tf) + cff_e_real_tf*cff_h_real_tf - cff_h_imag_tf*(-cff_e_imag_tf))/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) - (xb_tf**2*q_sq_tf*(cff_ht_real_tf*cff_et_real_tf - cff_et_imag_tf*(-cff_ht_imag_tf) + cff_et_real_tf*cff_ht_real_tf - cff_ht_imag_tf*(-cff_et_imag_tf))/(q_sq_tf+xb_tf*t_tf))
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) + 0.25*((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp_sq**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_real_tf*cff_e_real_tf - cff_e_imag_tf*(-cff_e_imag_tf)) -0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_real_tf*cff_et_real_tf - cff_et_imag_tf*(-cff_et_imag_tf))/((q_sq_tf+xb_tf*t_tf)*self.mp_sq**2)
        dvcs_real_curlyc = ((first_line + next_line + final_line)*q_sq_tf*(q_sq_tf+xb_tf*t_tf)/((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2)

        # DVCS Re[CurlyC](Feff| Feff*):
        first_line = (4.*(1.-xb_tf)*(cff_h_real_eff_tf*cff_h_real_eff_tf - cff_h_imag_eff_tf*(-cff_h_imag_eff_tf))) + (4.*(1.-xb_tf + 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_real_eff_tf*cff_ht_real_eff_tf - cff_ht_imag_eff_tf*(-cff_ht_imag_eff_tf)))
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_real_eff_tf*cff_e_real_eff_tf - cff_e_imag_eff_tf*(-cff_h_imag_eff_tf) + cff_e_real_eff_tf*cff_h_real_eff_tf - cff_h_imag_eff_tf*(-cff_e_imag_eff_tf))/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) - (xb_tf**2*q_sq_tf*(cff_ht_real_eff_tf*cff_et_real_eff_tf - cff_et_imag_eff_tf*(-cff_ht_imag_eff_tf) + cff_et_real_eff_tf*cff_ht_real_eff_tf - cff_ht_imag_eff_tf*(-cff_et_imag_eff_tf))/(q_sq_tf+xb_tf*t_tf))
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf)) + 0.25*((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp_sq**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_real_eff_tf*cff_e_real_eff_tf - cff_e_imag_eff_tf*(-cff_e_imag_eff_tf)) -0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_real_eff_tf*cff_ht_real_eff_tf - cff_et_imag_eff_tf*(-cff_et_imag_eff_tf))/((q_sq_tf+xb_tf*t_tf)*self.mp_sq**2)
        dvcs_real_curlyc_feff = ((first_line + next_line + final_line)*q_sq_tf*(q_sq_tf+xb_tf*t_tf)/((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2)

        # DVCS Re[CurlyC](Feff | F*):
        first_line = 4.*(1.-xb_tf)*(cff_h_real_eff_tf*cff_h_real_tf - cff_h_imag_eff_tf*(-cff_h_imag_tf))+4.*(1.-xb_tf+ 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_ht_imag_tf))
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_real_eff_tf*cff_e_real_tf- cff_e_imag_eff_tf*(-cff_h_imag_tf)+ cff_e_real_eff_tf*cff_h_real_tf - cff_h_imag_eff_tf*(-cff_e_imag_tf))/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))-xb_tf**2*q_sq_tf*(cff_ht_real_eff_tf*cff_et_real_tf - cff_et_imag_eff_tf*(-cff_ht_imag_tf)+ cff_et_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_et_imag_tf))/(q_sq_tf+xb_tf*t_tf)
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))+0.25*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp_sq**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_real_eff_tf*cff_e_real_tf- cff_e_imag_eff_tf*(-cff_e_imag_tf))-0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_real_eff_tf*cff_et_real_tf- cff_et_imag_eff_tf*(-cff_et_imag_tf))/((q_sq_tf+xb_tf*t_tf)*self.mp_sq**2)
        dvcs_real_curlyc_f_eff = ((first_line + next_line + final_line)* q_sq_tf*(q_sq_tf+xb_tf*t_tf)/ ((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2)

        # DVCS Im[CurlyC](Feff| F*):
        first_line = 4.*(1.-xb_tf)*(cff_h_imag_eff_tf*cff_h_real_tf- cff_h_real_eff_tf*cff_h_imag_tf)+4.*(1.-xb_tf+ 0.25*((2.*q_sq_tf + t_tf)*ep_tf**2)/(q_sq_tf + xb_tf*t_tf))*(cff_ht_imag_eff_tf*cff_ht_real_tf - cff_ht_real_eff_tf*cff_ht_imag_tf)
        next_line = -xb_tf**2*(q_sq_tf+t_tf)**2*(cff_h_imag_eff_tf*cff_e_real_tf - cff_e_real_eff_tf*cff_h_imag_tf + cff_e_imag_eff_tf*cff_h_real_tf - cff_h_real_eff_tf*cff_e_imag_tf)/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))-xb_tf**2*q_sq_tf*(cff_ht_imag_eff_tf*cff_et_real_tf - cff_et_real_eff_tf*cff_ht_imag_tf + cff_et_imag_eff_tf*cff_ht_real_tf- cff_ht_real_eff_tf*cff_et_imag_tf)/(q_sq_tf+xb_tf*t_tf)
        final_line = -1.*(xb_tf**2*(q_sq_tf+t_tf)**2/(q_sq_tf*(q_sq_tf+xb_tf*t_tf))+0.25*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2*t_tf/(q_sq_tf*self.mp_sq**2*(q_sq_tf+xb_tf*t_tf)))*(cff_e_imag_eff_tf*cff_e_real_tf- cff_e_real_eff_tf*cff_e_imag_tf)-0.25*xb_tf**2*q_sq_tf*t_tf*(cff_et_imag_eff_tf*cff_et_real_tf- cff_et_real_eff_tf*cff_et_imag_tf)/((q_sq_tf+xb_tf*t_tf)*self.mp_sq**2)
        dvcs_imag_curlyc_f_feff = ((first_line + next_line + final_line)*q_sq_tf*(q_sq_tf+xb_tf*t_tf)/((2.-xb_tf)*q_sq_tf+xb_tf*t_tf)**2)

        # DVCS Re[CurlyC_LP](F | F*)
        # [NOTE]: this one is for c0 LP:
        first_line = 4.*(1.-xb_tf+ ((3.-2.*xb_tf)*q_sq_tf + t_tf)*ep_tf**2/(4.*(q_sq_tf + xb_tf*t_tf)))*(cff_h_real_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_h_imag_tf)+ cff_ht_real_tf*cff_h_real_tf- cff_h_imag_tf*(-cff_ht_imag_tf))
        second_line = (-xb_tf**2*(q_sq_tf - xb_tf*t_tf*(1.-2.*xb_tf))*(cff_h_real_tf*cff_et_real_tf- cff_et_imag_tf*(-cff_h_imag_tf) + cff_et_real_tf*cff_h_real_tf- cff_h_imag_tf*(-cff_et_imag_tf)+ cff_ht_real_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_e_imag_tf))/(q_sq_tf + xb_tf*t_tf))
        third_line = (-xb_tf*(4.*(1.-xb_tf)*(q_sq_tf + xb_tf*t_tf)*t_tf+ ep_tf**2*(q_sq_tf+t_tf)**2)*(cff_ht_real_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_e_imag_tf))/(2.*q_sq_tf*(q_sq_tf + xb_tf*t_tf)))
        fourth_line = (-xb_tf*((q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)/(q_sq_tf + xb_tf*t_tf))*(xb_tf**2*(q_sq_tf+t_tf)**2/(2.*q_sq_tf*(q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)) + t_tf/(4.*self.mp_sq**2))*(cff_e_real_tf*cff_et_real_tf- cff_e_imag_tf*(-cff_et_imag_tf) + cff_et_real_tf*cff_e_real_tf- cff_et_imag_tf*(-cff_e_imag_tf)))
        dvcs_real_curlyc_lp = ((first_line + second_line + third_line + fourth_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/(tf.sqrt(1.+ep_tf**2)*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2))

        # DVCS LP Re[CurlyC_LP](F_eff | F*)
        # [NOTE]: Need this for c1 LP:
        first_line = (4.*(1.-xb_tf+ ((3.-2.*xb_tf)*q_sq_tf + t_tf)*ep_tf**2/(4.*(q_sq_tf + xb_tf*t_tf)))*(cff_h_real_eff_tf*cff_ht_real_tf- cff_ht_imag_tf*(-cff_h_imag_tf)+ cff_ht_real_eff_tf*cff_h_real_tf- cff_h_imag_eff_tf*(-cff_ht_imag_tf)))
        second_line = (-xb_tf**2*(q_sq_tf - xb_tf*t_tf*(1.-2.*xb_tf))*(cff_h_real_eff_tf*cff_et_real_tf- cff_et_imag_tf*(-cff_h_imag_tf)+ cff_et_real_eff_tf*cff_h_real_tf- cff_h_imag_eff_tf*(-cff_et_imag_tf)+ cff_ht_real_eff_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_e_imag_tf))/(q_sq_tf + xb_tf*t_tf))
        third_line = (-xb_tf*(4.*(1.-xb_tf)*(q_sq_tf + xb_tf*t_tf)*t_tf+ ep_tf**2*(q_sq_tf+t_tf)**2)*(cff_ht_real_eff_tf*cff_e_real_tf- cff_e_imag_tf*(-cff_ht_imag_tf)+ cff_e_real_eff_tf*cff_ht_real_tf- cff_ht_imag_eff_tf*(-cff_e_imag_tf))/(2.*q_sq_tf*(q_sq_tf + xb_tf*t_tf)))
        fourth_line = (-xb_tf*((q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)/(q_sq_tf + xb_tf*t_tf))*(xb_tf**2*(q_sq_tf+t_tf)**2/(2.*q_sq_tf*(q_sq_tf*(2.-xb_tf) + xb_tf*t_tf))+ t_tf/(4.*self.mp_sq**2))*(cff_e_real_eff_tf*cff_et_real_tf- cff_et_imag_tf*(-cff_e_imag_tf)+ cff_et_real_eff_tf*cff_e_real_tf- cff_e_imag_eff_tf*(-cff_et_imag_tf)))
        dvcs_real_curlyc_lp_feff_f = ((first_line + second_line + third_line + fourth_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/(tf.sqrt(1.+ep_tf**2)*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2))

        # DVCS LP Im[CurlyC_LP](F_eff | F*)
        # [NOTE]: Need this for s1 LP:
        first_line = (4.*(1.-xb_tf+ ((3.-2.*xb_tf)*q_sq_tf + t_tf)*ep_tf**2/(4.*(q_sq_tf + xb_tf*t_tf)))*(cff_h_imag_eff_tf*cff_ht_real_tf - cff_ht_real_tf*cff_h_imag_tf + cff_ht_imag_eff_tf*cff_h_real_tf- cff_h_real_tf*cff_ht_imag_tf))
        second_line = (-xb_tf**2*(q_sq_tf - xb_tf*t_tf*(1.-2.*xb_tf))*(cff_h_imag_eff_tf*cff_et_real_tf- cff_et_real_tf*cff_h_imag_tf+ cff_et_imag_eff_tf*cff_h_real_tf- cff_h_real_tf*cff_et_imag_tf+ cff_ht_imag_eff_tf*cff_e_real_tf- cff_e_real_tf*cff_ht_imag_tf+ cff_e_imag_eff_tf*cff_ht_real_tf- cff_ht_real_tf*cff_e_imag_tf)/(q_sq_tf + xb_tf*t_tf))
        third_line = (-xb_tf*(4.*(1.-xb_tf)*(q_sq_tf + xb_tf*t_tf)*t_tf+ ep_tf**2*(q_sq_tf+t_tf)**2)*(cff_h_imag_eff_tf*cff_et_real_tf- cff_et_real_tf*cff_h_imag_tf+ cff_et_imag_eff_tf*cff_h_real_tf- cff_h_real_tf*cff_et_imag_tf)/(2.*q_sq_tf*(q_sq_tf + xb_tf*t_tf)))
        fourth_line = (-xb_tf*((q_sq_tf*(2.-xb_tf) + xb_tf*t_tf)/(q_sq_tf + xb_tf*t_tf))*(xb_tf**2*(q_sq_tf+t_tf)**2/(2.*q_sq_tf*(q_sq_tf*(2.-xb_tf) + xb_tf*t_tf))+ t_tf/(4.*self.mp_sq**2))*(cff_e_imag_eff_tf*cff_et_real_tf- cff_et_real_tf*cff_e_imag_tf+ cff_et_imag_eff_tf*cff_e_real_tf- cff_e_real_tf*cff_et_imag_tf))
        dvcs_imag_curlyc_lp_feff_f = ((first_line + second_line + third_line + fourth_line)* q_sq_tf*(q_sq_tf + xb_tf*t_tf)/(tf.sqrt(1.+ep_tf**2)*((2.-xb_tf)*q_sq_tf + xb_tf*t_tf)**2))

        # DVCS: c0(F | F*):
        first_term_prefactor = 2. * ( 2. - 2. * y_lep_tf + y_lep_tf**2 + (ep_tf**2 * y_lep_tf**2 / 2.)) / (1. + ep_tf**2)
        second_term_prefactor = 16. * k_tf**2 / ((2. - xb_tf)**2 * (1. + ep_tf**2))
        dvcs_unp_c0 = first_term_prefactor * dvcs_real_curlyc + second_term_prefactor * dvcs_real_curlyc_feff

        # DVCS: c1(Feff | F*):
        prefactor = 8. * k_tf * (2. - y_lep_tf) / ((2. - xb_tf) * (1. + ep_tf**2))
        dvcs_unp_c1 = prefactor * dvcs_real_curlyc_f_eff

        # DVCS: s1(Feff | F*):
        prefactor = -8. * k_tf * lep_lambda * y_lep_tf * tf.sqrt(1. + ep_tf**2) / ((2. - xb_tf) * (1. + ep_tf**2))
        dvcs_unp_s1 = prefactor * dvcs_imag_curlyc_f_feff

        # DVCS LP: c0:
        prefactor = 2.*lep_lambda*tgt_lambda*y_lep_tf*(2.-y_lep_tf)/tf.sqrt(1.+ep_tf*ep_tf)
        dvcs_lp_c0 = prefactor * dvcs_real_curlyc_lp

        # DVCS LP: c1:
        prefactor = 8.*tgt_lambda*k_tf*lep_lambda*y_lep_tf*tf.sqrt(1+ep_tf*ep_tf)/((2.-xb_tf)*(1.+ep_tf*ep_tf))
        dvcs_lp_c1 = prefactor * dvcs_real_curlyc_lp_feff_f
    
        # DVCS LP: s1:
        prefactor = -8.*tgt_lambda*k_tf*(2.-y_lep_tf)/((2.-xb_tf)*(1.+ep_tf*ep_tf))
        dvcs_lp_s1 = prefactor * dvcs_imag_curlyc_lp_feff_f

        # Interference: C(n = 0)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        two_minus_xb = 2. - xb_tf
        two_minus_y = 2. - y_lep_tf
        first_term_in_brackets = ktilde_tf**2 * two_minus_y**2 / (q_sq_tf * root_one_plus_epsilon_squared)
        second_term_in_brackets_first_part = t_over_Q_squared * two_minus_xb * (1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        second_term_in_brackets_second_part_numerator = 2. * xb_tf * t_over_Q_squared * (two_minus_xb + 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep_tf**2 / xb_tf) + ep_tf**2
        second_term_in_brackets_second_part =  1. + second_term_in_brackets_second_part_numerator / (two_minus_xb * one_plus_root_epsilon_stuff)
        prefactor = -4. * two_minus_y * one_plus_root_epsilon_stuff / tf.pow(root_one_plus_epsilon_squared, 4)
        c_0_plus_plus_unp = prefactor * (first_term_in_brackets + second_term_in_brackets_first_part * second_term_in_brackets_second_part)

        # Interference: CV(n = 0)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_term_in_brackets = (2. - y_lep_tf)**2 * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)
        second_term_first_multiplicative_term = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_term_second_multiplicative_term = one_plus_root_epsilon_stuff / 2.
        second_term_third_multiplicative_term = 1. + t_over_Q_squared
        second_term_fourth_multiplicative_term = 1. + (root_one_plus_epsilon_squared - 1. + (2. * xb_tf)) * t_over_Q_squared / one_plus_root_epsilon_stuff
        second_term_in_brackets = second_term_first_multiplicative_term * second_term_second_multiplicative_term * second_term_third_multiplicative_term * second_term_fourth_multiplicative_term
        coefficient_prefactor = 8. * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_0_plus_plus_V_unp = coefficient_prefactor * (first_term_in_brackets + second_term_in_brackets)

        # Interference: CA(n = 0):
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        two_minus_y = 2. - y_lep_tf
        ktilde_over_Q_squared = ktilde_tf**2 / q_sq_tf
        curly_bracket_first_term = two_minus_y**2 * ktilde_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf) / (2. * root_one_plus_epsilon_squared)
        deepest_parentheses_term = (xb_tf * (2. + one_plus_root_epsilon_stuff - 2. * xb_tf) / one_plus_root_epsilon_stuff + (one_plus_root_epsilon_stuff - 2.)) * t_over_Q_squared
        square_bracket_term = one_plus_root_epsilon_stuff * (one_plus_root_epsilon_stuff - xb_tf + deepest_parentheses_term) / 2. - (2. * ktilde_over_Q_squared)
        curly_bracket_second_term = (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * square_bracket_term
        coefficient_prefactor = 8. * two_minus_y * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_0_plus_plus_A_unp = coefficient_prefactor * (curly_bracket_first_term + curly_bracket_second_term)

        # Interference: C(n = 0)0+:
        bracket_quantity = ep_tf**2 + t_tf * (2. - 6.* xb_tf - ep_tf**2) / (3. * q_sq_tf)
        prefactor = 12. * tf.sqrt(2.) * k_tf * (2. - y_lep_tf) * tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4)) / tf.pow(1. + ep_tf**2, 2.5)
        c_0_zero_plus_unp = prefactor * bracket_quantity

        # Interference: CV(n = 0)0+:
        t_over_Q_squared = t_tf / q_sq_tf
        main_part = xb_tf * t_over_Q_squared * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)
        prefactor = 24. * tf.sqrt(2.) * k_tf * (2. - y_lep_tf) * tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / (1. + ep_tf**2)**2.5
        c_0_zero_plus_V_unp = prefactor * main_part

        # Interference: CA(n = 0)0+:
        t_over_Q_squared = t_tf / q_sq_tf
        fancy_xb_epsilon_term = 8. - 6. * xb_tf + 5. * ep_tf**2
        brackets_term = 1. - t_over_Q_squared * (2. - 12. * xb_tf * (1. - xb_tf) - ep_tf**2) / fancy_xb_epsilon_term
        prefactor = 4. * tf.sqrt(2.) * k_tf * (2. - y_lep_tf) * tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / tf.pow(1. + ep_tf**2, 2.5)
        c_0_zero_plus_A_unp = prefactor * t_over_Q_squared * fancy_xb_epsilon_term * brackets_term

        # Interference: C(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_bracket_first_term = (1. + (1. - xb_tf) * (root_one_plus_epsilon_squared - 1.) / (2. * xb_tf) + ep_tf**2 / (4. * xb_tf)) * xb_tf * t_over_Q_squared
        first_bracket_term = first_bracket_first_term - 3. * ep_tf**2 / 4.
        second_bracket_term = 1. - (1. - 3. * xb_tf) * t_over_Q_squared + (1. - root_one_plus_epsilon_squared + 3. * ep_tf**2) * xb_tf * t_over_Q_squared / (one_plus_root_epsilon_stuff - ep_tf**2)
        fancy_y_coefficient = 2. - 2. * y_lep_tf + y_lep_tf**2 + ep_tf**2 * y_lep_tf**2 / 2.
        second_term = -4. * k_tf * fancy_y_coefficient * (one_plus_root_epsilon_stuff - ep_tf**2) * second_bracket_term / root_one_plus_epsilon_squared**5
        first_term = -16. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * first_bracket_term / root_one_plus_epsilon_squared**5
        c_1_plus_plus_unp = first_term + second_term

        # Interference: CV(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = (2. - y_lep_tf)**2 * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)
        second_bracket_term_first_part = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        second_bracket_term_second_part = 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * tprime_tf / q_sq_tf
        coefficient_prefactor = 16. * k_tf * xb_tf * t_over_Q_squared / tf.pow(root_one_plus_epsilon_squared, 5)
        c_1_plus_plus_V_unp = coefficient_prefactor * (first_bracket_term + second_bracket_term_first_part * second_bracket_term_second_part)

        # Interference: CA(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        one_minus_2xb = 1. - 2. * xb_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        first_bracket_term_second_part = 1. - one_minus_2xb * t_over_Q_squared + (4. * xb_tf * one_minus_xb + ep_tf**2) * t_prime_over_Q_squared / (4. * root_one_plus_epsilon_squared)
        second_bracket_term = 1. - 0.5 * xb_tf + 0.25 * (one_minus_2xb + root_one_plus_epsilon_squared) * (1. - t_over_Q_squared) + (4. * xb_tf * one_minus_xb + ep_tf**2) * t_prime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
        prefactor = -16. * k_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_1_plus_plus_A_unp = prefactor * (fancy_y_stuff * first_bracket_term_second_part - (2. - y_lep_tf)**2 * second_bracket_term)

        # Interference: C(n = 1)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        first_bracket_term = (2. - y_lep_tf)**2 * t_prime_over_Q_squared * (one_minus_xb + (one_minus_xb * xb_tf + (ep_tf**2 / 4.)) * t_prime_over_Q_squared / root_one_plus_epsilon_squared)
        second_bracket_term = y_quantity * (1. - (1. - 2. * xb_tf) * t_over_Q_squared) * (ep_tf**2 - 2. * (1. + (ep_tf**2 / (2. * xb_tf))) * xb_tf * t_over_Q_squared) / root_one_plus_epsilon_squared
        prefactor = 8. * tf.sqrt(2. * y_quantity) / root_one_plus_epsilon_squared**4
        c_1_zero_plus_unp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: CV(n = 1)0+:
        t_over_Q_squared = t_tf / q_sq_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        major_part = (2 - y_lep_tf)**2 * ktilde_tf**2 / q_sq_tf + (1. - (1. - 2. * xb_tf) * t_over_Q_squared)**2 * y_quantity
        prefactor = 16. * tf.sqrt(2. * y_quantity) * xb_tf * t_over_Q_squared / (1. + ep_tf**2)**2.5
        c_1_zero_plus_V_unp = prefactor * major_part

        # Interference: CA(n = 1)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_minus_2xb = 1. - 2. * xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_term_first_part = (1. - one_minus_2xb * t_over_Q_squared) * y_quantity
        second_term_second_part = 4. - 2. * xb_tf + 3. * ep_tf**2 + t_over_Q_squared * (4. * xb_tf * (1. - xb_tf) + ep_tf**2)
        first_term = ktilde_tf**2 * one_minus_2xb * (2. - y_lep_tf)**2 / q_sq_tf
        prefactor = 8. * tf.sqrt(2. * y_quantity) * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_1_zero_plus_A_unp = prefactor * (first_term + second_term_first_part * second_term_second_part)

        # Interference: C(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = 2. * ep_tf**2 * ktilde_tf**2 / (root_one_plus_epsilon_squared * (1. + root_one_plus_epsilon_squared) * q_sq_tf)
        second_bracket_term = xb_tf * tprime_tf * t_over_Q_squared * (1. - xb_tf - 0.5 * (root_one_plus_epsilon_squared - 1.) + 0.5 * ep_tf**2 / xb_tf) / q_sq_tf
        prefactor = 8. * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / root_one_plus_epsilon_squared
        c_2_plus_plus_unp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: CV(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        major_term = (4. * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)) + 0.5 * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * (1. + t_over_Q_squared) * t_prime_over_Q_squared
        prefactor = 8. * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_2_plus_plus_V_unp = prefactor * major_term

        # Interference: CA(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        first_bracket_term = 4. * (1. - 2. * xb_tf) * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)
        second_bracket_term = (3.  - root_one_plus_epsilon_squared - 2. * xb_tf + ep_tf**2 / xb_tf ) * xb_tf * t_prime_over_Q_squared
        prefactor = 4. * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_2_plus_plus_A_unp = prefactor * (first_bracket_term - second_bracket_term)

        # Interference: C(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        epsilon_squared_over_2 = ep_tf**2 / 2.
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb_tf) / (1. + epsilon_squared_over_2)) * xb_tf * t_tf / q_sq_tf
        prefactor = -8. * tf.sqrt(2. * y_quantity) * k_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**5
        c_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term

        # Interference: CV(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        y_quantity = tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * y_quantity * k_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_2_zero_plus_V_unp = prefactor * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)

        # Interference: CA(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        bracket_term = one_minus_xb + 0.5 * t_prime_over_Q_squared * (4. * xb_tf * one_minus_xb + ep_tf**2) / root_one_plus_epsilon_squared
        prefactor = 8. * tf.sqrt(2. * y_quantity) * k_tf * (2. - y_lep_tf) * t_over_Q_squared / root_one_plus_epsilon_squared**4
        c_2_zero_plus_A_unp = prefactor * bracket_term

        # Interference: C(n = 3)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        major_term = (1. - xb_tf) * t_over_Q_squared + 0.5 * (root_one_plus_epsilon_squared - 1.) * (1. + t_over_Q_squared)
        intermediate_term = (root_one_plus_epsilon_squared - 1.) / root_one_plus_epsilon_squared**5
        prefactor = -8. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.)
        c_3_plus_plus_unp = prefactor * intermediate_term * major_term

        # Interference: CV(n = 3)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        major_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * t_over_Q_squared
        prefactor = -8. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_3_plus_plus_V_unp = prefactor * major_term

        # Interference: CA(n = 3)++:
        main_term = t_tf * tprime_tf * (xb_tf * (1. - xb_tf) + ep_tf**2 / 4.) / q_sq_tf**2
        prefactor = 16. * k_tf * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / (1. + ep_tf**2)**2.5
        c_3_plus_plus_A_unp = prefactor * main_term

        # Interference: C(n = 3)0+:
        c_3_zero_plus_unp = 0.0

        # Interference: CV(n = 3)0+:
        c_3_zero_plus_V_unp = 0.0

        # Interference: CA(n = 3)0+:
        c_3_zero_plus_A_unp = 0.0

        # Interference: S(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        bracket_term = 1. + ((1. - xb_tf + 0.5 * (root_one_plus_epsilon_squared - 1.)) / root_one_plus_epsilon_squared**2) * tPrime_over_Q_squared
        prefactor = 8. * lep_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**2
        s_1_plus_plus_unp = prefactor * bracket_term

        # Interference: SV(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        bracket_term = root_one_plus_epsilon_squared - 1. + (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * t_over_Q_squared
        prefactor = -8. * lep_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_1_plus_plus_V_unp = prefactor * bracket_term

        # Interference: SA(n = 1)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        one_minus_2xb = 1. - 2. * xb_tf
        bracket_term = 1. - one_minus_2xb * (one_minus_2xb + root_one_plus_epsilon_squared) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
        prefactor = 8. * lep_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) * t_over_Q_squared / root_one_plus_epsilon_squared**2
        s_1_plus_plus_A_unp = prefactor * bracket_term

        # Interference: S(n = 1)0+:
        root_one_plus_epsilon_squared = (1. + ep_tf**2)**2
        y_quantity = tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        s_1_zero_plus_unp = 8. * tf.sqrt(2.) * lep_lambda * (2. - y_lep_tf) * y_lep_tf * y_quantity * ktilde_tf**2 / (root_one_plus_epsilon_squared * q_sq_tf)

        # Interference: SV(n = 1)0+:
        one_plus_epsilon_squared_squared = (1. + ep_tf**2)**2
        t_over_Q_squared = t_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        bracket_term = 4. * (1. - 2. * xb_tf) * t_over_Q_squared * (1. + xb_tf * t_over_Q_squared) + ep_tf**2 * (1. + t_over_Q_squared)**2
        prefactor = 4. * tf.sqrt(2. * fancy_y_stuff) * lep_lambda * y_lep_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / one_plus_epsilon_squared_squared
        s_1_zero_plus_V_unp = prefactor * bracket_term

        # Interference: SA(n = 1)0+:
        one_plus_epsilon_squared_squared = (1. + ep_tf**2)**2
        fancy_y_stuff = tf.sqrt(1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.)
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * y_lep_tf * (2. - y_lep_tf) * (1. - 2. * xb_tf) / one_plus_epsilon_squared_squared
        s_1_zero_plus_A_unp = prefactor * fancy_y_stuff * t_tf * k_tf**2 / q_sq_tf

        # Interference: S(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        first_bracket_term = (ep_tf**2 - xb_tf * (root_one_plus_epsilon_squared - 1.)) / (1. + root_one_plus_epsilon_squared - 2. * xb_tf)
        second_bracket_term = (2. * xb_tf + ep_tf**2) * tPrime_over_Q_squared / (2. * root_one_plus_epsilon_squared)
        prefactor = -4. * lep_lambda * fancy_y_stuff * y_lep_tf * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * tPrime_over_Q_squared / root_one_plus_epsilon_squared**3
        s_2_plus_plus_unp = prefactor * (first_bracket_term - second_bracket_term)

        # Interference: SV(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        one_minus_2xb = 1. - 2. * xb_tf
        bracket_term = root_one_plus_epsilon_squared - 1. + (one_minus_2xb + root_one_plus_epsilon_squared) * t_over_Q_squared
        parentheses_term = 1. - one_minus_2xb * t_over_Q_squared
        prefactor = -4. * lep_lambda * fancy_y_stuff * y_lep_tf * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_plus_plus_V_unp = prefactor * parentheses_term * bracket_term

        # Interference: SA(n = 2)++:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        tPrime_over_Q_squared = tprime_tf / q_sq_tf
        fancy_y_stuff = 1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.
        last_term = 1. + (4. * (1. - xb_tf) * xb_tf + ep_tf**2) * t_over_Q_squared / (4. - 2. * xb_tf + 3. * ep_tf**2)
        middle_term = 1. + root_one_plus_epsilon_squared - 2. * xb_tf
        prefactor = -8. * lep_lambda * fancy_y_stuff * y_lep_tf * t_over_Q_squared * tPrime_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_plus_plus_A_unp = prefactor * middle_term * last_term

        # Interference: S(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        epsilon_squared_over_2 = ep_tf**2 / 2.
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        bracket_term = 1. + ((1. + epsilon_squared_over_2 / xb_tf) / (1. + epsilon_squared_over_2)) * xb_tf * t_tf / q_sq_tf
        prefactor = 8. * lep_lambda * tf.sqrt(2. * y_quantity) * k_tf * y_lep_tf / root_one_plus_epsilon_squared**4
        s_2_zero_plus_unp = prefactor * (1. + epsilon_squared_over_2) * bracket_term

        # Interference: SV(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        y_quantity = tf.sqrt(1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * y_quantity * k_tf * y_lep_tf * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_zero_plus_V_unp = prefactor * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)

        # Interference: SA(n = 2)0+:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_minus_xb = 1. - xb_tf
        y_quantity = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        main_term = 4. * one_minus_xb + 2. * ep_tf**2 + 4. * t_over_Q_squared * (4. * xb_tf * one_minus_xb + ep_tf**2)
        prefactor = -2. * tf.sqrt(2. * y_quantity) * lep_lambda * k_tf * y_lep_tf * t_over_Q_squared / root_one_plus_epsilon_squared**4
        s_2_zero_plus_A_unp = prefactor * main_term

        # Interference: S(n = 3)++:
        s_3_plus_plus_unp = 0.0

        # Interference: SV(n = 3)++:
        s_3_plus_plus_V_unp = 0.0

        # Interference: SA(n = 3)++:
        s_3_plus_plus_A_unp = 0.0

        # Interference: S(n = 3)++:
        s_3_zero_plus_unp = 0.0

        # Interference: SV(n = 3)0+:
        s_3_zero_plus_V_unp = 0.0

        # Interference: SA(n = 3)0+:
        s_3_zero_plus_A_unp = 0.0

        # Interference: C(n = 0)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf 
        first_bracket_term = (2. - y_lep_tf)**2 * ktilde_tf**2 / q_sq_tf
        second_bracket_term_first_part = 1. - y_lep_tf + (ep_tf**2 * y_lep_tf**2 / 4.)
        second_bracket_term_second_part = xb_tf * t_over_Q_squared - (ep_tf**2 * (1. - t_over_Q_squared) / 2.)
        second_bracket_term_third_part = 1. + t_over_Q_squared * ((root_one_plus_epsilon_squared - 1. + 2. * xb_tf) / (1. + root_one_plus_epsilon_squared))
        second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part
        prefactor = -4. * lep_lambda * tgt_lambda * y_lep_tf * (1. + root_one_plus_epsilon_squared) / root_one_plus_epsilon_squared**5
        c_0_plus_plus_lp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: CV(n = 0)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_bracket_term = (2. - y_lep_tf)**2 * (one_plus_root_epsilon_stuff - 2. * xb_tf) * ktilde_tf**2 / (q_sq_tf * one_plus_root_epsilon_stuff)
        second_bracket_term_first_part = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_bracket_term_second_part = 2. - xb_tf + 3. * ep_tf**2 / 2
        second_bracket_term_third_part = 1. + (t_over_Q_squared * (4. * (1. - xb_tf) * xb_tf + ep_tf**2) / (4. - 2. * xb_tf + 3. * ep_tf**2))
        second_bracket_term_fourth_part = 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. + 2. * xb_tf) / one_plus_root_epsilon_stuff)
        second_bracket_term = second_bracket_term_first_part * second_bracket_term_second_part * second_bracket_term_third_part * second_bracket_term_fourth_part
        prefactor = 4. * lep_lambda * tgt_lambda * y_lep_tf * one_plus_root_epsilon_stuff * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_0_plus_plus_V_lp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: CA(n = 0)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_bracket_term = 2. * (2. - y_lep_tf)**2 * ktilde_tf**2 / q_sq_tf
        second_bracket_term_first_part = 1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)
        second_bracket_term_second_part = 1. - (1. - 2. * xb_tf) * t_over_Q_squared
        second_bracket_term_third_part = 1. + (t_over_Q_squared * (root_one_plus_epsilon_squared - 1. + 2. * xb_tf) / one_plus_root_epsilon_stuff)
        second_bracket_term = second_bracket_term_first_part * one_plus_root_epsilon_stuff * second_bracket_term_second_part * second_bracket_term_third_part
        prefactor = 4. * lep_lambda * tgt_lambda * y_lep_tf * xb_tf * t_over_Q_squared / root_one_plus_epsilon_squared**5
        c_0_plus_plus_A_lp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: C(n = 0)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * (1. - xb_tf) * y_lep_tf / (1. + ep_tf**2)**2
        c_0_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * t_tf / q_sq_tf

        # Interference: CV(n = 0)0+ LP:
        modulating_factor = (xb_tf - (t_tf * (1. - 2. * xb_tf) / q_sq_tf)) / (1. - xb_tf)
        c_0_zero_plus_V_lp = c_0_zero_plus_lp * modulating_factor

        # Interference: CA(n = 0)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        t_over_Q_squared = t_tf / q_sq_tf
        c_0_zero_plus_A_lp = prefactor * root_combination_of_y_and_epsilon * xb_tf * t_over_Q_squared * (1. + t_over_Q_squared)

        # Interference: C(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        one_plus_root_epsilon_minus_epsilon_squared = one_plus_root_epsilon_stuff - ep_tf**2
        major_factor = 1. - ((t_tf / q_sq_tf) * (1. - 2. * xb_tf * (one_plus_root_epsilon_stuff + 1.) / one_plus_root_epsilon_minus_epsilon_squared))
        prefactor = -4. * lep_lambda * tgt_lambda * y_lep_tf * k_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**5
        c_1_plus_plus_lp = prefactor * one_plus_root_epsilon_minus_epsilon_squared * major_factor

        # Interference: CV(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_minus_xb = 1. - xb_tf
        root_epsilon_and_xb_quantity = root_one_plus_epsilon_squared + 2. * one_minus_xb
        bracket_factor_numerator = 1. + ((1. - ep_tf**2) / root_one_plus_epsilon_squared) - (2. * xb_tf * (1. + (4. * one_minus_xb / root_one_plus_epsilon_squared)))
        bracket_factor_denominator = 2. * root_epsilon_and_xb_quantity
        bracket_factor = 1. - (tprime_tf * bracket_factor_numerator / (q_sq_tf * bracket_factor_denominator))
        prefactor = 8. * lep_lambda * tgt_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) / root_one_plus_epsilon_squared**4
        c_1_plus_plus_V_lp = prefactor * root_epsilon_and_xb_quantity * t_tf * bracket_factor / q_sq_tf

        # Interference: CA(n = 1)++ LP:
        t_over_Q_squared = t_tf / q_sq_tf
        major_factor = xb_tf * t_over_Q_squared * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)
        prefactor = 16. * lep_lambda * tgt_lambda * k_tf * y_lep_tf * (2. - y_lep_tf) / tf.sqrt(1. + ep_tf**2)**5
        c_1_plus_plus_A_lp = prefactor * major_factor

        # Interference: C(n = 1)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * (1. - y_lep_tf) * y_lep_tf / (1. + ep_tf**2)**2
        c_1_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * ktilde_tf**2 / q_sq_tf

        # Interference: CV(n = 1)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda  * (2. - y_lep_tf) * y_lep_tf / (1. + ep_tf**2)**2
        c_1_zero_plus_V_lp = prefactor * root_combination_of_y_and_epsilon * t_tf * ktilde_tf**2 / q_sq_tf**2

        # Interference: CA(n = 1)0+ LP:
        c_1_zero_plus_A_lp = 0.0

        # Interference: C(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_multiplicative_factor = (-1. * one_plus_root_epsilon_stuff + 2.) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf)
        second_multiplicative_factor = xb_tf * t_over_Q_squared - (ep_tf**2 * (1. - t_over_Q_squared) / 2.)
        prefactor = -4. * lep_lambda * tgt_lambda * y_lep_tf * (1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        c_2_plus_plus_lp = prefactor * first_multiplicative_factor * second_multiplicative_factor

        # Interference: CV(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_multiplicative_factor = (one_plus_root_epsilon_stuff - 2.) + t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf)
        second_multiplicative_factor = 1. + (t_over_Q_squared * (4. * (1. - xb_tf) * xb_tf + ep_tf**2 ) / (4. - 2. * xb_tf + 3. * ep_tf**2))
        third_multiplicative_factor = t_over_Q_squared * (4. - 2. * xb_tf + 3. * ep_tf**2)
        prefactor = -2.*lep_lambda*tgt_lambda*y_lep_tf*(1.-y_lep_tf-(y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        c_2_plus_plus_V_lp = prefactor * first_multiplicative_factor * second_multiplicative_factor * third_multiplicative_factor

        # Interference: CA(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        first_multiplicative_factor = (1. - root_one_plus_epsilon_squared) - t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf)
        second_multiplicative_factor = xb_tf * t_over_Q_squared * (1. - t_over_Q_squared * (1. - 2. * xb_tf))
        prefactor = 4. * lep_lambda * tgt_lambda * y_lep_tf * (1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        c_2_plus_plus_A_lp = prefactor * first_multiplicative_factor * second_multiplicative_factor

        # Interference: C(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        c_2_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * (1. + (xb_tf * t_tf / q_sq_tf))
    
        # Interference: CV(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        c_2_zero_plus_V_lp = prefactor * root_combination_of_y_and_epsilon * (1. - xb_tf ) * t_tf / q_sq_tf

        # Interference: CA(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * lep_lambda * tgt_lambda * k_tf * y_lep_tf / (1. + ep_tf**2)**2
        t_over_Q_squared = t_tf / q_sq_tf
        c_2_zero_plus_A_lp = prefactor * root_combination_of_y_and_epsilon * xb_tf * t_over_Q_squared * (1. + t_tf / q_sq_tf)

        # Interference: CA(n = 3)++ LP:
        c_3_plus_plus_lp = 0.0

        # Interference: CA(n = 3)++ LP:
        c_3_plus_plus_V_lp = 0.0

        # Interference: CA(n = 3)++ LP:
        c_3_plus_plus_A_lp = 0.0

        # Interference: CA(n = 3)0+ LP:
        c_3_zero_plus_lp = 0.0

        # Interference: CA(n = 3)0+ LP:
        c_3_zero_plus_V_lp = 0.0

        # Interference: CA(n = 3)0+ LP:
        c_3_zero_plus_A_lp = 0.0

        # Interference: S(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        t_over_Q_squared = t_tf / q_sq_tf
        epsilon_y_over_2_squared = (ep_tf * y_lep_tf / 2.) ** 2
        first_bracket_term = 2. * root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (one_plus_root_epsilon_stuff - 2. * xb_tf) / one_plus_root_epsilon_stuff)
        second_bracket_term = (3. * ep_tf**2 / 2.) + (t_over_Q_squared * (1. - root_one_plus_epsilon_squared - ep_tf**2 / 2. - xb_tf * (3.  - root_one_plus_epsilon_squared)))
        almost_prefactor = 4. * tgt_lambda * k_tf / root_one_plus_epsilon_squared**6
        prefactor_one = almost_prefactor * (2. - 2. * y_lep_tf + y_lep_tf**2 + 2. * epsilon_y_over_2_squared) * one_plus_root_epsilon_stuff
        prefactor_two = 2. * almost_prefactor * (1. - y_lep_tf - epsilon_y_over_2_squared)
        s_1_plus_plus_lp = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term

        # Interference: SV(n = 1)++ LP:
        ep_squared = ep_tf**2
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_squared)
        t_over_Q_squared = t_tf / q_sq_tf
        t_prime_over_Q_squared = tprime_tf / q_sq_tf
        epsilon_y_over_2_squared = ep_squared * y_lep_tf**2 / 4.
        first_bracket_term = 1. - (t_prime_over_Q_squared * ((1. - 2. * xb_tf) * (1. - 2. * xb_tf + root_one_plus_epsilon_squared)) / (2. * root_one_plus_epsilon_squared**2))
        second_term_parentheses_term = t_over_Q_squared * (1. - (xb_tf * ((3. + root_one_plus_epsilon_squared) / 4.)) + (5. * ep_squared / 8.))
        second_bracket_term_numerator = 1. - root_one_plus_epsilon_squared + (ep_squared / 2.) - (2. * xb_tf * (3. * (1. - xb_tf) - root_one_plus_epsilon_squared))
        second_bracket_term_denominator = 4. - (xb_tf * (root_one_plus_epsilon_squared + 3.)) + (5. * ep_squared / 2.)
        second_bracket_term = 1. - (t_over_Q_squared * second_bracket_term_numerator / second_bracket_term_denominator)
        almost_prefactor = 8. * tgt_lambda * k_tf / root_one_plus_epsilon_squared**4
        prefactor_one = almost_prefactor * (2. - 2. * y_lep_tf + y_lep_tf**2 + 2. * epsilon_y_over_2_squared) * t_over_Q_squared
        prefactor_two = 4. * almost_prefactor * (1. - y_lep_tf - epsilon_y_over_2_squared) / root_one_plus_epsilon_squared**2
        s_1_plus_plus_V_lp = prefactor_one * first_bracket_term + prefactor_two * second_term_parentheses_term * second_bracket_term

        # Interference: SA(n = 1)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        t_over_Q_squared = t_tf / q_sq_tf
        xB_t_over_Q_squared = xb_tf * t_over_Q_squared
        three_plus_root_epsilon_stuff = 3 + root_one_plus_epsilon_squared
        epsilon_y_over_2_squared = (ep_tf * y_lep_tf / 2.) ** 2
        almost_prefactor = 8. * tgt_lambda * k_tf / root_one_plus_epsilon_squared**6
        first_bracket_term = root_one_plus_epsilon_squared - 1. + (t_over_Q_squared * (1. + root_one_plus_epsilon_squared - 2. * xb_tf))
        second_bracket_term = 1. - (t_over_Q_squared * (3.  - root_one_plus_epsilon_squared - 6. * xb_tf) / three_plus_root_epsilon_stuff)
        prefactor_one = -1. * almost_prefactor * (2. - 2. * y_lep_tf + y_lep_tf**2 + 2. * epsilon_y_over_2_squared) * xB_t_over_Q_squared
        prefactor_two = almost_prefactor * (1. - y_lep_tf - epsilon_y_over_2_squared) * three_plus_root_epsilon_stuff * xB_t_over_Q_squared
        s_1_plus_plus_A_lp = prefactor_one * first_bracket_term + prefactor_two * second_bracket_term

        # Interference: S(n = 1)0+ LP:
        combination_of_y_and_epsilon = 1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = ktilde_tf**2 * (2. - y_lep_tf)**2 / q_sq_tf
        second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * (2. * xb_tf * t_over_Q_squared - (ep_tf**2 * (1. - t_over_Q_squared)))
        prefactor = 8. * tf.sqrt(2.) * tgt_lambda  * tf.sqrt(combination_of_y_and_epsilon) / tf.sqrt((1. + ep_tf**2)**5)
        s_1_zero_plus_lp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: SV(n = 1)0+ LP:
        combination_of_y_and_epsilon = 1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)
        t_over_Q_squared = t_tf / q_sq_tf
        first_bracket_term = ktilde_tf**2 * (2. - y_lep_tf)**2 / q_sq_tf
        second_bracket_term_long = 4. - 2. * xb_tf + 3. * ep_tf**2 + t_over_Q_squared * (4. * xb_tf * (1. - xb_tf) + ep_tf**2)
        second_bracket_term = (1. + t_over_Q_squared) * combination_of_y_and_epsilon * second_bracket_term_long
        prefactor = -8. * tf.sqrt(2.) * tgt_lambda  * tf.sqrt(combination_of_y_and_epsilon) * t_over_Q_squared / tf.sqrt((1. + ep_tf**2)**5)
        s_1_zero_plus_V_lp = prefactor * (first_bracket_term + second_bracket_term)

        # Interference: SA(n = 1)0+ LP:
        combination_of_y_and_epsilon_to_3_halves = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))**3
        t_over_Q_squared = t_tf / q_sq_tf
        prefactor = -16. * tf.sqrt(2.) * tgt_lambda * xb_tf * t_over_Q_squared * (1. + t_over_Q_squared) / tf.sqrt((1. + ep_tf**2)**5)
        s_1_zero_plus_A_lp = prefactor * combination_of_y_and_epsilon_to_3_halves * (1. - (1. - 2. * xb_tf) * t_over_Q_squared)

        # Interference: S(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        bracket_term = 4. * ktilde_tf**2 * (one_plus_root_epsilon_stuff - 2. * xb_tf) * (one_plus_root_epsilon_stuff + xb_tf * t_tf / q_sq_tf) * tprime_tf / (root_one_plus_epsilon_squared * q_sq_tf**2)
        prefactor = -4. * tgt_lambda * (2. - y_lep_tf) * (1. - y_lep_tf - (ep_tf**2 * y_lep_tf**2 / 4.)) / root_one_plus_epsilon_squared**5
        s_2_plus_plus_lp = prefactor * bracket_term

        # Interference: SV(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        bracket_term_second_term = (3.  - root_one_plus_epsilon_squared - (2. * xb_tf) + (ep_tf**2 / xb_tf)) * xb_tf * tprime_tf / q_sq_tf
        bracket_term_first_term = 4. * ktilde_tf**2 * (1. - 2. * xb_tf) / (root_one_plus_epsilon_squared * q_sq_tf)
        bracket_term = t_tf * (bracket_term_first_term - bracket_term_second_term) / q_sq_tf
        prefactor = 4. * tgt_lambda * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / root_one_plus_epsilon_squared**5
        s_2_plus_plus_V_lp = prefactor * bracket_term

        # Interference: SA(n = 2)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        bracket_term_first_term = (1. + root_one_plus_epsilon_squared - 2. * xb_tf) * (1. - ((1. - 2. * xb_tf) * t_tf / q_sq_tf)) * tprime_tf / q_sq_tf
        bracket_term_second_term = 4. * ktilde_tf**2 / q_sq_tf
        bracket_term = xb_tf * t_tf * (bracket_term_second_term - bracket_term_first_term) / q_sq_tf
        prefactor = 4. * tgt_lambda * (2. - y_lep_tf) * (1. - y_lep_tf - ep_tf**2 * y_lep_tf**2 / 4.) / root_one_plus_epsilon_squared**5
        s_2_plus_plus_A_lp = prefactor * bracket_term

        # Interference: S(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = 8. * tf.sqrt(2.) * tgt_lambda * k * (2. - y_lep_tf )/ tf.sqrt((1. + ep_tf**2)**5)
        s_2_zero_plus_lp = prefactor * root_combination_of_y_and_epsilon * (1. + (xb_tf * t_tf / q_sq_tf))

        # Interference: SV(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        prefactor = -8. * tf.sqrt(2.) * tgt_lambda * k_tf * (2. - y_lep_tf) * t_tf / (tf.sqrt((1. + ep_tf**2)**5) * q_sq_tf)
        s_2_zero_plus_V_lp = prefactor * (1. - xb_tf) * root_combination_of_y_and_epsilon

        # Interference: SA(n = 2)0+ LP:
        root_combination_of_y_and_epsilon = tf.sqrt(1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.))
        t_over_Q_squared = t_tf / q_sq_tf
        prefactor = -8. * tf.sqrt(2.) * tgt_lambda  * k_tf * (2. - y_lep_tf) * xb_tf * t_over_Q_squared / tf.sqrt((1. + ep_tf**2)**5)
        s_2_zero_plus_A_lp = prefactor * root_combination_of_y_and_epsilon * (1. + t_over_Q_squared)

        # Interference: S(n = 3)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        one_plus_root_epsilon_stuff = 1. + root_one_plus_epsilon_squared
        prefactor = -4. * tgt_lambda * k_tf * (1. - y_lep_tf - y_lep_tf**2 * ep_tf**2 / 4.) / root_one_plus_epsilon_squared**6
        s_3_plus_plus_lp = prefactor * (one_plus_root_epsilon_stuff - 2. * xb_tf) * ep_tf**2 * tprime_tf / (q_sq_tf * one_plus_root_epsilon_stuff)

        # Interference: SV(n = 3)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        multiplicative_contribution = t_tf * tprime_tf * (4. * (1. - xb_tf) * xb_tf + ep_tf**2) / q_sq_tf**2
        prefactor = 4. * tgt_lambda * k_tf * (1. - y_lep_tf - y_lep_tf**2 * ep_tf**2 / 4.) / root_one_plus_epsilon_squared**6
        s_3_plus_plus_V_lp = prefactor * multiplicative_contribution

        # Interference: SA(n = 3)++ LP:
        root_one_plus_epsilon_squared = tf.sqrt(1. + ep_tf**2)
        multiplicative_contribution = xb_tf * t_tf * tprime_tf * (1. + root_one_plus_epsilon_squared - 2. * xb_tf) / q_sq_tf**2
        prefactor = -8. * tgt_lambda * k_tf * (1. - y_lep_tf - (y_lep_tf**2 * ep_tf**2 / 4.)) / root_one_plus_epsilon_squared**6
        s_3_plus_plus_A_lp = prefactor * multiplicative_contribution

        # Interference: S(n = 3)0+ LP:
        s_3_zero_plus_lp = 0.0

        # Interference: SV(n = 3)0+ LP:
        s_3_zero_plus_V_lp = 0.0

        # Interference: SA(n = 3)0+ LP:
        s_3_zero_plus_A_lp = 0.0

        # Interference: Re[CurlyC(F)]
        i_curly_c_unp_real = (
            (f1_tf*cff_h_real_tf) - t_tf * f2_tf * cff_e_real_tf / (4.*self.mp_sq**2) +
            xb_tf * (f1_tf+f2_tf)*cff_ht_real_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
            )

        # Interference: Re[CurlyC(F_eff)]
        i_curly_c_unp_feff = (
            (f1_tf*cff_h_real_eff_tf)- t_tf * f2_tf * cff_e_real_eff_tf / (4.*self.mp_sq**2) +
            xb_tf * (f1_tf + f2_tf)*cff_ht_real_eff_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyC(F)]
        i_curly_c_unp_imag = (
            (f1_tf * cff_h_imag_tf) - t_tf * f2_tf * cff_e_imag_tf / (4.*self.mp_sq**2) +
            xb_tf * (f1_tf + f2_tf) * cff_ht_imag_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyC(F_eff)]
        i_curly_c_unp_imag_feff = (
            (f1_tf * cff_h_imag_eff_tf) - t_tf * f2_tf * cff_e_imag_eff_tf / (4.*self.mp_sq**2) +
            xb_tf * (f1_tf + f2_tf) * cff_ht_imag_eff_tf / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCV(F)]
        i_curly_c_v_unp_real = (
            (cff_h_real_tf + cff_e_real_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCV(F_eff)]
        i_curly_c_v_unp_real_feff = (
            (cff_h_real_eff_tf + cff_e_real_eff_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCV(F)]
        i_curly_c_v_unp_imag = (
            (cff_h_imag_tf + cff_e_imag_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCV(F_eff)]
        i_curly_c_v_unp_imag_feff = (
            (cff_h_imag_eff_tf + cff_e_imag_eff_tf) * xb_tf * (f1_tf + f2_tf) / (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCA(F)]
        i_curly_c_a_unp_real = (
            cff_ht_real_tf* xb_tf * (f1_tf + f2_tf)/ (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyCA(F_eff)]
        i_curly_c_a_unp_real_feff = (
            cff_ht_real_eff_tf* xb_tf * (f1_tf + f2_tf)/ (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCA(F)]
        i_curly_c_a_unp_imag = (
            cff_ht_real_tf* xb_tf * (f1_tf + f2_tf)/ (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Im[CurlyCA(F_eff)]
        i_curly_c_a_unp_imag_feff = (
            cff_ht_imag_eff_tf* xb_tf * (f1_tf + f2_tf)/ (2. - xb_tf + xb_tf*t_tf/q_sq_tf)
        )

        # Interference: Re[CurlyC(F)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_real = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_tf+xb_correction_tf*cff_e_real_tf)+
            (
                1.+(self.mp_sq**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_real_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_real_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp_sq**2))*cff_et_real_tf
        )

        # Interference: Re[CurlyC(F_eff)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_real_feff = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_eff_tf+xb_correction_tf*cff_e_real_eff_tf)+
            (
                1.+(self.mp_sq**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_real_eff_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_real_eff_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp_sq**2))*cff_et_real_eff_tf
        )

        # Interference: Im[CurlyC(F)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_imag = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_tf+xb_correction_tf*cff_e_imag_tf)+
            (
                1.+(self.mp_sq**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_imag_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_imag_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp_sq**2))*cff_et_imag_tf
        )

        # Interference: Im[CurlyC(F_eff)] LP:
        t_over_q_sq_tf = t_tf / q_sq_tf
        ratio_xb_tf = xb_tf / (2. - xb_tf + xb_tf*t_over_q_sq_tf)
        xb_correction_tf = (xb_tf * (1. - t_over_q_sq_tf) / 2.)
        i_curly_c_lp_imag_feff = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_eff_tf+xb_correction_tf*cff_e_imag_eff_tf)+
            (
                1.+(self.mp_sq**2*xb_tf*ratio_xb_tf*(3. + t_over_q_sq_tf)/q_sq_tf)
            )*f1_tf*cff_ht_imag_eff_tf-t_over_q_sq_tf*2.*(1.-2.*xb_tf)*ratio_xb_tf*f2_tf*cff_ht_imag_eff_tf
            - ratio_xb_tf*(xb_correction_tf*f1_tf+ t_tf*f2_tf/(4.*self.mp_sq**2))*cff_et_imag_eff_tf
        )

        # Interference: Re[CurlyCV(F)] LP
        i_curly_c_v_lp_real = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_tf+ xb_correction_tf*cff_e_real_tf)
        )

        # Interference: Re[CurlyCV(F_eff)] LP
        i_curly_c_v_lp_real_feff = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_real_eff_tf+ xb_correction_tf*cff_e_real_eff_tf)
        )

        # Interference: Im[CurlyCV(F)] LP
        i_curly_c_v_lp_imag = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_tf+ xb_correction_tf*cff_e_imag_tf)
        )

        # Interference: Im[CurlyCV(F_eff)] LP
        i_curly_c_v_lp_imag_feff = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_h_imag_eff_tf+ xb_correction_tf*cff_e_imag_eff_tf)
        )

        # Interference Re[CurlyCA(F)] LP
        i_curly_c_a_lp_real = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_real_tf*(1. + 2.*xb_tf*self.mp_sq**2/q_sq_tf)+xb_tf*cff_et_real_tf/2.)
        )

        # Interference Re[CurlyCA(F_eff)] LP
        i_curly_c_a_lp_real_feff = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_real_eff_tf*(1. + 2.*xb_tf*self.mp_sq**2/q_sq_tf)+xb_tf*cff_et_real_eff_tf/2.)
        )

        # Interference Im[CurlyCA(F)] LP
        i_curly_c_a_lp_imag = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_imag_tf*(1. + 2.*xb_tf*self.mp_sq**2/q_sq_tf)+xb_tf*cff_et_imag_tf/2.)
        )

        # Interference Im[CurlyCA(F_eff)] LP
        i_curly_c_a_lp_imag_feff = (
            ratio_xb_tf*(f1_tf + f2_tf)*(cff_ht_imag_eff_tf*(1. + 2.*xb_tf*self.mp_sq**2/q_sq_tf)+xb_tf*cff_et_imag_eff_tf/2.)
        )

        # Interference: c_{0}:
        c0_unp = (
            c_0_plus_plus_unp * i_curly_c_unp_real +
            c_0_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_0_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_0_zero_plus_unp * i_curly_c_unp_feff +
            c_0_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_0_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        c0_lp = (
            c_0_plus_plus_lp * i_curly_c_lp_real +
            c_0_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_0_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_0_zero_plus_lp * i_curly_c_lp_real_feff +
            c_0_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_0_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )
        
        # Interference: c_{1}:
        c1_unp = (
            c_1_plus_plus_unp * i_curly_c_unp_real +
            c_1_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_1_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_1_zero_plus_unp * i_curly_c_unp_feff +
            c_1_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_1_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        c1_lp = (
            c_1_plus_plus_lp * i_curly_c_lp_real +
            c_1_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_1_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_1_zero_plus_lp * i_curly_c_lp_real_feff +
            c_1_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_1_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )

        # Interference: c_{2}:
        c2_unp = (
            c_2_plus_plus_unp * i_curly_c_unp_real +
            c_2_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_2_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_2_zero_plus_unp * i_curly_c_unp_feff +
            c_2_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_2_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        c2_lp = (
            c_2_plus_plus_lp * i_curly_c_lp_real +
            c_2_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_2_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_2_zero_plus_lp * i_curly_c_lp_real_feff +
            c_2_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_2_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )

        # Interference: c_{3}:
        c3_unp = (
            c_3_plus_plus_unp * i_curly_c_unp_real +
            c_3_plus_plus_V_unp * i_curly_c_v_unp_real +
            c_3_plus_plus_A_unp * i_curly_c_a_unp_real +
            c_3_zero_plus_unp * i_curly_c_unp_feff +
            c_3_zero_plus_V_unp * i_curly_c_v_unp_real_feff +
            c_3_zero_plus_A_unp * i_curly_c_a_unp_real_feff
            )

        c3_lp = (
            c_3_plus_plus_lp * i_curly_c_lp_real +
            c_3_plus_plus_V_lp * i_curly_c_v_lp_real +
            c_3_plus_plus_A_lp * i_curly_c_a_lp_real +
            c_3_zero_plus_lp * i_curly_c_lp_real_feff +
            c_3_zero_plus_V_lp * i_curly_c_v_lp_real_feff +
            c_3_zero_plus_A_lp * i_curly_c_a_lp_real_feff
            )

        # Interference: s_{1}:
        s1_unp = (
            s_1_plus_plus_unp * i_curly_c_unp_imag +
            s_1_plus_plus_V_unp * i_curly_c_v_unp_imag +
            s_1_plus_plus_A_unp * i_curly_c_a_unp_imag+
            s_1_zero_plus_unp * i_curly_c_unp_imag_feff +
            s_1_zero_plus_V_unp * i_curly_c_v_unp_imag_feff +
            s_1_zero_plus_A_unp * i_curly_c_a_unp_imag_feff
            )

        s1_lp = (
            s_1_plus_plus_lp * i_curly_c_lp_imag +
            s_1_plus_plus_V_lp * i_curly_c_v_lp_imag +
            s_1_plus_plus_A_lp * i_curly_c_a_lp_imag+
            s_1_zero_plus_lp * i_curly_c_lp_imag_feff +
            s_1_zero_plus_V_lp * i_curly_c_v_lp_imag_feff +
            s_1_zero_plus_A_lp * i_curly_c_a_lp_imag_feff
            )

        # Interference: s_{2}:
        s2_unp = (
            s_2_plus_plus_unp * i_curly_c_unp_imag +
            s_2_plus_plus_V_unp * i_curly_c_v_unp_imag +
            s_2_plus_plus_A_unp * i_curly_c_a_unp_imag +
            s_2_zero_plus_unp * i_curly_c_unp_imag_feff +
            s_2_zero_plus_V_unp * i_curly_c_v_unp_imag_feff +
            s_2_zero_plus_A_unp * i_curly_c_a_unp_imag_feff
            )

        s2_lp = (
            s_2_plus_plus_lp * i_curly_c_lp_imag +
            s_2_plus_plus_V_lp * i_curly_c_v_lp_imag +
            s_2_plus_plus_A_lp * i_curly_c_a_lp_imag+
            s_2_zero_plus_lp * i_curly_c_lp_imag_feff +
            s_2_zero_plus_V_lp * i_curly_c_v_lp_imag_feff +
            s_2_zero_plus_A_lp * i_curly_c_a_lp_imag_feff
            )

        # Interference: s_{3}:
        s3_unp = (
            s_3_plus_plus_unp * i_curly_c_unp_imag +
            s_3_plus_plus_V_unp * i_curly_c_v_unp_imag +
            s_3_plus_plus_A_unp * i_curly_c_a_unp_imag +
            s_3_zero_plus_unp * i_curly_c_unp_imag_feff +
            s_3_zero_plus_V_unp * i_curly_c_v_unp_imag_feff +
            s_3_zero_plus_A_unp * i_curly_c_a_unp_imag_feff
            )

        s3_lp = (
            s_3_plus_plus_lp * i_curly_c_lp_imag +
            s_3_plus_plus_V_lp * i_curly_c_v_lp_imag +
            s_3_plus_plus_A_lp * i_curly_c_a_lp_imag+
            s_3_zero_plus_lp * i_curly_c_lp_imag_feff +
            s_3_zero_plus_V_lp * i_curly_c_v_lp_imag_feff +
            s_3_zero_plus_A_lp * i_curly_c_a_lp_imag_feff
            )

        # Trigomometry:
        cosine_0 = tf.cos(0. * (tf.constant(np.pi) - phi_tf))
        cosine_1 = tf.cos(1. * (tf.constant(np.pi) - phi_tf))
        cosine_2 = tf.cos(2. * (tf.constant(np.pi) - phi_tf))
        cosine_3 = tf.cos(3. * (tf.constant(np.pi) - phi_tf))
        sine_1 = tf.sin(1. * (tf.constant(np.pi) - phi_tf))
        sine_2 = tf.sin(2. * (tf.constant(np.pi) - phi_tf))
        sine_3 = tf.sin(2. * (tf.constant(np.pi) - phi_tf))

        bh_squared = (
            (
                (bh_c0 + bh_lp_c0) * cosine_0 +
                (bh_c1 + bh_lp_c1) * cosine_1 +
                (bh_c2 + bh_lp_c2) * cosine_2
            ) / 
            (xb_tf * xb_tf * y_lep_tf * y_lep_tf * (1. + ep_tf**2)**2 * t_tf * p1_tf * p2_tf)
            )

        dvcs_squared = (
            (
                (dvcs_unp_c0 + dvcs_lp_c0) * cosine_0 +
                (dvcs_unp_c1 + dvcs_lp_c1) * cosine_1 +
                (dvcs_unp_s1 + dvcs_lp_s1) * sine_1
            ) / (y_lep_tf * y_lep_tf * q_sq_tf)
        )

        interference = (
            (
                (c0_unp + c0_lp) * cosine_0 +
                (c1_unp + c1_lp) * cosine_1 +
                (c2_unp + c2_lp) * cosine_2 +
                (c3_unp + c3_lp) * cosine_3 +
                (s1_unp + s1_lp) * sine_1 +
                (s2_unp + s2_lp) * sine_2 +
                (s3_unp + s3_lp) * sine_3
            ) /(xb_tf * y_lep_tf * y_lep_tf * y_lep_tf * t_tf * p1_tf * p2_tf)
        )

        cross_section = bh_squared + dvcs_squared + interference

        predicted_bsa = 0.0

        # compute cross-section residuals:
        residuals_cross_section = true_cross_section - cross_section
        # compute BSA residuals:
        residuals_bsa = true_bsa - predicted_bsa

        # compute the MSE:
        mean_squared_error = (
            self._OBSERVABLE_WEIGHT_1 * tf.reduce_mean(tf.square(residuals_cross_section))+
            self._OBSERVABLE_WEIGHT_2 * tf.reduce_mean(tf.square(residuals_bsa)))

        return mean_squared_error
