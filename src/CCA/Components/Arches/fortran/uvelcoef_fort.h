
#ifndef fspec_uvelcoef
#define fspec_uvelcoef

#ifdef __cplusplus

extern "C" void uvelcoef_(int* uu_low_x, int* uu_low_y, int* uu_low_z, int* uu_high_x, int* uu_high_y, int* uu_high_z, double* uu_ptr,
                          int* cesav_low_x, int* cesav_low_y, int* cesav_low_z, int* cesav_high_x, int* cesav_high_y, int* cesav_high_z, double* cesav_ptr,
                          int* cwsav_low_x, int* cwsav_low_y, int* cwsav_low_z, int* cwsav_high_x, int* cwsav_high_y, int* cwsav_high_z, double* cwsav_ptr,
                          int* cnsav_low_x, int* cnsav_low_y, int* cnsav_low_z, int* cnsav_high_x, int* cnsav_high_y, int* cnsav_high_z, double* cnsav_ptr,
                          int* cssav_low_x, int* cssav_low_y, int* cssav_low_z, int* cssav_high_x, int* cssav_high_y, int* cssav_high_z, double* cssav_ptr,
                          int* ctsav_low_x, int* ctsav_low_y, int* ctsav_low_z, int* ctsav_high_x, int* ctsav_high_y, int* ctsav_high_z, double* ctsav_ptr,
                          int* cbsav_low_x, int* cbsav_low_y, int* cbsav_low_z, int* cbsav_high_x, int* cbsav_high_y, int* cbsav_high_z, double* cbsav_ptr,
                          int* ap_low_x, int* ap_low_y, int* ap_low_z, int* ap_high_x, int* ap_high_y, int* ap_high_z, double* ap_ptr,
                          int* ae_low_x, int* ae_low_y, int* ae_low_z, int* ae_high_x, int* ae_high_y, int* ae_high_z, double* ae_ptr,
                          int* aw_low_x, int* aw_low_y, int* aw_low_z, int* aw_high_x, int* aw_high_y, int* aw_high_z, double* aw_ptr,
                          int* an_low_x, int* an_low_y, int* an_low_z, int* an_high_x, int* an_high_y, int* an_high_z, double* an_ptr,
                          int* as_low_x, int* as_low_y, int* as_low_z, int* as_high_x, int* as_high_y, int* as_high_z, double* as_ptr,
                          int* at_low_x, int* at_low_y, int* at_low_z, int* at_high_x, int* at_high_y, int* at_high_z, double* at_ptr,
                          int* ab_low_x, int* ab_low_y, int* ab_low_z, int* ab_high_x, int* ab_high_y, int* ab_high_z, double* ab_ptr,
                          int* vv_low_x, int* vv_low_y, int* vv_low_z, int* vv_high_x, int* vv_high_y, int* vv_high_z, double* vv_ptr,
                          int* ww_low_x, int* ww_low_y, int* ww_low_z, int* ww_high_x, int* ww_high_y, int* ww_high_z, double* ww_ptr,
                          int* den_low_x, int* den_low_y, int* den_low_z, int* den_high_x, int* den_high_y, int* den_high_z, double* den_ptr,
                          int* vis_low_x, int* vis_low_y, int* vis_low_z, int* vis_high_x, int* vis_high_y, int* vis_high_z, double* vis_ptr,
                          int* den_ref_low_x, int* den_ref_low_y, int* den_ref_low_z, int* den_ref_high_x, int* den_ref_high_y, int* den_ref_high_z, double* den_ref_ptr,
                          int* SU_low_x, int* SU_low_y, int* SU_low_z, int* SU_high_x, int* SU_high_y, int* SU_high_z, double* SU_ptr,
                          int* old_den_low_x, int* old_den_low_y, int* old_den_low_z, int* old_den_high_x, int* old_den_high_y, int* old_den_high_z, double* old_den_ptr,
                          int* old_UU_low_x, int* old_UU_low_y, int* old_UU_low_z, int* old_UU_high_x, int* old_UU_high_y, int* old_UU_high_z, double* old_UU_ptr,
                          int* eps_low_x, int* eps_low_y, int* eps_low_z, int* eps_high_x, int* eps_high_y, int* eps_high_z, double* eps_ptr,
                          double* deltat,
                          double* grav,
                          bool* lcend,
                          int* ceeu_low, int* ceeu_high, double* ceeu_ptr,
                          int* cweu_low, int* cweu_high, double* cweu_ptr,
                          int* cwwu_low, int* cwwu_high, double* cwwu_ptr,
                          int* cnn_low, int* cnn_high, double* cnn_ptr,
                          int* csn_low, int* csn_high, double* csn_ptr,
                          int* css_low, int* css_high, double* css_ptr,
                          int* ctt_low, int* ctt_high, double* ctt_ptr,
                          int* cbt_low, int* cbt_high, double* cbt_ptr,
                          int* cbb_low, int* cbb_high, double* cbb_ptr,
                          int* sewu_low, int* sewu_high, double* sewu_ptr,
                          int* sew_low, int* sew_high, double* sew_ptr,
                          int* sns_low, int* sns_high, double* sns_ptr,
                          int* stb_low, int* stb_high, double* stb_ptr,
                          int* dxepu_low, int* dxepu_high, double* dxepu_ptr,
                          int* dxpwu_low, int* dxpwu_high, double* dxpwu_ptr,
                          int* dxpw_low, int* dxpw_high, double* dxpw_ptr,
                          int* dynp_low, int* dynp_high, double* dynp_ptr,
                          int* dyps_low, int* dyps_high, double* dyps_ptr,
                          int* dztp_low, int* dztp_high, double* dztp_ptr,
                          int* dzpb_low, int* dzpb_high, double* dzpb_ptr,
                          int* fac1u_low, int* fac1u_high, double* fac1u_ptr,
                          int* fac2u_low, int* fac2u_high, double* fac2u_ptr,
                          int* fac3u_low, int* fac3u_high, double* fac3u_ptr,
                          int* fac4u_low, int* fac4u_high, double* fac4u_ptr,
                          int* iesdu_low, int* iesdu_high, int* iesdu_ptr,
                          int* iwsdu_low, int* iwsdu_high, int* iwsdu_ptr,
                          int* nfac_low, int* nfac_high, double* nfac_ptr,
                          int* sfac_low, int* sfac_high, double* sfac_ptr,
                          int* tfac_low, int* tfac_high, double* tfac_ptr,
                          int* bfac_low, int* bfac_high, double* bfac_ptr,
                          int* fac1ns_low, int* fac1ns_high, double* fac1ns_ptr,
                          int* fac2ns_low, int* fac2ns_high, double* fac2ns_ptr,
                          int* fac3ns_low, int* fac3ns_high, double* fac3ns_ptr,
                          int* fac4ns_low, int* fac4ns_high, double* fac4ns_ptr,
                          int* n_shift_low, int* n_shift_high, int* n_shift_ptr,
                          int* s_shift_low, int* s_shift_high, int* s_shift_ptr,
                          int* fac1tb_low, int* fac1tb_high, double* fac1tb_ptr,
                          int* fac2tb_low, int* fac2tb_high, double* fac2tb_ptr,
                          int* fac3tb_low, int* fac3tb_high, double* fac3tb_ptr,
                          int* fac4tb_low, int* fac4tb_high, double* fac4tb_ptr,
                          int* t_shift_low, int* t_shift_high, int* t_shift_ptr,
                          int* b_shift_low, int* b_shift_high, int* b_shift_ptr,
                          int* idxLoU,
                          int* idxHiU);

static void fort_uvelcoef( Uintah::constSFCXVariable<double> & uu,
                           Uintah::SFCXVariable<double> & cesav,
                           Uintah::SFCXVariable<double> & cwsav,
                           Uintah::SFCXVariable<double> & cnsav,
                           Uintah::SFCXVariable<double> & cssav,
                           Uintah::SFCXVariable<double> & ctsav,
                           Uintah::SFCXVariable<double> & cbsav,
                           Uintah::SFCXVariable<double> & ap,
                           Uintah::SFCXVariable<double> & ae,
                           Uintah::SFCXVariable<double> & aw,
                           Uintah::SFCXVariable<double> & an,
                           Uintah::SFCXVariable<double> & as,
                           Uintah::SFCXVariable<double> & at,
                           Uintah::SFCXVariable<double> & ab,
                           Uintah::constSFCYVariable<double> & vv,
                           Uintah::constSFCZVariable<double> & ww,
                           Uintah::constCCVariable<double> & den,
                           Uintah::constCCVariable<double> & vis,
                           Uintah::constCCVariable<double> & den_ref,
                           Uintah::SFCXVariable<double> & SU,
                           Uintah::constCCVariable<double> & old_den,
                           Uintah::constSFCXVariable<double> & old_UU,
                           Uintah::constCCVariable<double> & eps,
                           double & deltat,
                           double & grav,
                           bool & lcend,
                           Uintah::OffsetArray1<double> & ceeu,
                           Uintah::OffsetArray1<double> & cweu,
                           Uintah::OffsetArray1<double> & cwwu,
                           Uintah::OffsetArray1<double> & cnn,
                           Uintah::OffsetArray1<double> & csn,
                           Uintah::OffsetArray1<double> & css,
                           Uintah::OffsetArray1<double> & ctt,
                           Uintah::OffsetArray1<double> & cbt,
                           Uintah::OffsetArray1<double> & cbb,
                           Uintah::OffsetArray1<double> & sewu,
                           Uintah::OffsetArray1<double> & sew,
                           Uintah::OffsetArray1<double> & sns,
                           Uintah::OffsetArray1<double> & stb,
                           Uintah::OffsetArray1<double> & dxepu,
                           Uintah::OffsetArray1<double> & dxpwu,
                           Uintah::OffsetArray1<double> & dxpw,
                           Uintah::OffsetArray1<double> & dynp,
                           Uintah::OffsetArray1<double> & dyps,
                           Uintah::OffsetArray1<double> & dztp,
                           Uintah::OffsetArray1<double> & dzpb,
                           Uintah::OffsetArray1<double> & fac1u,
                           Uintah::OffsetArray1<double> & fac2u,
                           Uintah::OffsetArray1<double> & fac3u,
                           Uintah::OffsetArray1<double> & fac4u,
                           Uintah::OffsetArray1<int> & iesdu,
                           Uintah::OffsetArray1<int> & iwsdu,
                           Uintah::OffsetArray1<double> & nfac,
                           Uintah::OffsetArray1<double> & sfac,
                           Uintah::OffsetArray1<double> & tfac,
                           Uintah::OffsetArray1<double> & bfac,
                           Uintah::OffsetArray1<double> & fac1ns,
                           Uintah::OffsetArray1<double> & fac2ns,
                           Uintah::OffsetArray1<double> & fac3ns,
                           Uintah::OffsetArray1<double> & fac4ns,
                           Uintah::OffsetArray1<int> & n_shift,
                           Uintah::OffsetArray1<int> & s_shift,
                           Uintah::OffsetArray1<double> & fac1tb,
                           Uintah::OffsetArray1<double> & fac2tb,
                           Uintah::OffsetArray1<double> & fac3tb,
                           Uintah::OffsetArray1<double> & fac4tb,
                           Uintah::OffsetArray1<int> & t_shift,
                           Uintah::OffsetArray1<int> & b_shift,
                           Uintah::IntVector & idxLoU,
                           Uintah::IntVector & idxHiU )
{
  Uintah::IntVector uu_low = uu.offset();
  Uintah::IntVector uu_high = uu.size() + uu_low - Uintah::IntVector(1, 1, 1);
  int uu_low_x = uu_low.x();
  int uu_high_x = uu_high.x();
  int uu_low_y = uu_low.y();
  int uu_high_y = uu_high.y();
  int uu_low_z = uu_low.z();
  int uu_high_z = uu_high.z();
  Uintah::IntVector cesav_low = cesav.offset();
  Uintah::IntVector cesav_high = cesav.size() + cesav_low - Uintah::IntVector(1, 1, 1);
  int cesav_low_x = cesav_low.x();
  int cesav_high_x = cesav_high.x();
  int cesav_low_y = cesav_low.y();
  int cesav_high_y = cesav_high.y();
  int cesav_low_z = cesav_low.z();
  int cesav_high_z = cesav_high.z();
  Uintah::IntVector cwsav_low = cwsav.offset();
  Uintah::IntVector cwsav_high = cwsav.size() + cwsav_low - Uintah::IntVector(1, 1, 1);
  int cwsav_low_x = cwsav_low.x();
  int cwsav_high_x = cwsav_high.x();
  int cwsav_low_y = cwsav_low.y();
  int cwsav_high_y = cwsav_high.y();
  int cwsav_low_z = cwsav_low.z();
  int cwsav_high_z = cwsav_high.z();
  Uintah::IntVector cnsav_low = cnsav.offset();
  Uintah::IntVector cnsav_high = cnsav.size() + cnsav_low - Uintah::IntVector(1, 1, 1);
  int cnsav_low_x = cnsav_low.x();
  int cnsav_high_x = cnsav_high.x();
  int cnsav_low_y = cnsav_low.y();
  int cnsav_high_y = cnsav_high.y();
  int cnsav_low_z = cnsav_low.z();
  int cnsav_high_z = cnsav_high.z();
  Uintah::IntVector cssav_low = cssav.offset();
  Uintah::IntVector cssav_high = cssav.size() + cssav_low - Uintah::IntVector(1, 1, 1);
  int cssav_low_x = cssav_low.x();
  int cssav_high_x = cssav_high.x();
  int cssav_low_y = cssav_low.y();
  int cssav_high_y = cssav_high.y();
  int cssav_low_z = cssav_low.z();
  int cssav_high_z = cssav_high.z();
  Uintah::IntVector ctsav_low = ctsav.offset();
  Uintah::IntVector ctsav_high = ctsav.size() + ctsav_low - Uintah::IntVector(1, 1, 1);
  int ctsav_low_x = ctsav_low.x();
  int ctsav_high_x = ctsav_high.x();
  int ctsav_low_y = ctsav_low.y();
  int ctsav_high_y = ctsav_high.y();
  int ctsav_low_z = ctsav_low.z();
  int ctsav_high_z = ctsav_high.z();
  Uintah::IntVector cbsav_low = cbsav.offset();
  Uintah::IntVector cbsav_high = cbsav.size() + cbsav_low - Uintah::IntVector(1, 1, 1);
  int cbsav_low_x = cbsav_low.x();
  int cbsav_high_x = cbsav_high.x();
  int cbsav_low_y = cbsav_low.y();
  int cbsav_high_y = cbsav_high.y();
  int cbsav_low_z = cbsav_low.z();
  int cbsav_high_z = cbsav_high.z();
  Uintah::IntVector ap_low = ap.offset();
  Uintah::IntVector ap_high = ap.size() + ap_low - Uintah::IntVector(1, 1, 1);
  int ap_low_x = ap_low.x();
  int ap_high_x = ap_high.x();
  int ap_low_y = ap_low.y();
  int ap_high_y = ap_high.y();
  int ap_low_z = ap_low.z();
  int ap_high_z = ap_high.z();
  Uintah::IntVector ae_low = ae.offset();
  Uintah::IntVector ae_high = ae.size() + ae_low - Uintah::IntVector(1, 1, 1);
  int ae_low_x = ae_low.x();
  int ae_high_x = ae_high.x();
  int ae_low_y = ae_low.y();
  int ae_high_y = ae_high.y();
  int ae_low_z = ae_low.z();
  int ae_high_z = ae_high.z();
  Uintah::IntVector aw_low = aw.offset();
  Uintah::IntVector aw_high = aw.size() + aw_low - Uintah::IntVector(1, 1, 1);
  int aw_low_x = aw_low.x();
  int aw_high_x = aw_high.x();
  int aw_low_y = aw_low.y();
  int aw_high_y = aw_high.y();
  int aw_low_z = aw_low.z();
  int aw_high_z = aw_high.z();
  Uintah::IntVector an_low = an.offset();
  Uintah::IntVector an_high = an.size() + an_low - Uintah::IntVector(1, 1, 1);
  int an_low_x = an_low.x();
  int an_high_x = an_high.x();
  int an_low_y = an_low.y();
  int an_high_y = an_high.y();
  int an_low_z = an_low.z();
  int an_high_z = an_high.z();
  Uintah::IntVector as_low = as.offset();
  Uintah::IntVector as_high = as.size() + as_low - Uintah::IntVector(1, 1, 1);
  int as_low_x = as_low.x();
  int as_high_x = as_high.x();
  int as_low_y = as_low.y();
  int as_high_y = as_high.y();
  int as_low_z = as_low.z();
  int as_high_z = as_high.z();
  Uintah::IntVector at_low = at.offset();
  Uintah::IntVector at_high = at.size() + at_low - Uintah::IntVector(1, 1, 1);
  int at_low_x = at_low.x();
  int at_high_x = at_high.x();
  int at_low_y = at_low.y();
  int at_high_y = at_high.y();
  int at_low_z = at_low.z();
  int at_high_z = at_high.z();
  Uintah::IntVector ab_low = ab.offset();
  Uintah::IntVector ab_high = ab.size() + ab_low - Uintah::IntVector(1, 1, 1);
  int ab_low_x = ab_low.x();
  int ab_high_x = ab_high.x();
  int ab_low_y = ab_low.y();
  int ab_high_y = ab_high.y();
  int ab_low_z = ab_low.z();
  int ab_high_z = ab_high.z();
  Uintah::IntVector vv_low = vv.offset();
  Uintah::IntVector vv_high = vv.size() + vv_low - Uintah::IntVector(1, 1, 1);
  int vv_low_x = vv_low.x();
  int vv_high_x = vv_high.x();
  int vv_low_y = vv_low.y();
  int vv_high_y = vv_high.y();
  int vv_low_z = vv_low.z();
  int vv_high_z = vv_high.z();
  Uintah::IntVector ww_low = ww.offset();
  Uintah::IntVector ww_high = ww.size() + ww_low - Uintah::IntVector(1, 1, 1);
  int ww_low_x = ww_low.x();
  int ww_high_x = ww_high.x();
  int ww_low_y = ww_low.y();
  int ww_high_y = ww_high.y();
  int ww_low_z = ww_low.z();
  int ww_high_z = ww_high.z();
  Uintah::IntVector den_low = den.offset();
  Uintah::IntVector den_high = den.size() + den_low - Uintah::IntVector(1, 1, 1);
  int den_low_x = den_low.x();
  int den_high_x = den_high.x();
  int den_low_y = den_low.y();
  int den_high_y = den_high.y();
  int den_low_z = den_low.z();
  int den_high_z = den_high.z();
  Uintah::IntVector vis_low = vis.offset();
  Uintah::IntVector vis_high = vis.size() + vis_low - Uintah::IntVector(1, 1, 1);
  int vis_low_x = vis_low.x();
  int vis_high_x = vis_high.x();
  int vis_low_y = vis_low.y();
  int vis_high_y = vis_high.y();
  int vis_low_z = vis_low.z();
  int vis_high_z = vis_high.z();
  Uintah::IntVector den_ref_low = den_ref.offset();
  Uintah::IntVector den_ref_high = den_ref.size() + den_ref_low - Uintah::IntVector(1, 1, 1);
  int den_ref_low_x = den_ref_low.x();
  int den_ref_high_x = den_ref_high.x();
  int den_ref_low_y = den_ref_low.y();
  int den_ref_high_y = den_ref_high.y();
  int den_ref_low_z = den_ref_low.z();
  int den_ref_high_z = den_ref_high.z();
  Uintah::IntVector SU_low = SU.offset();
  Uintah::IntVector SU_high = SU.size() + SU_low - Uintah::IntVector(1, 1, 1);
  int SU_low_x = SU_low.x();
  int SU_high_x = SU_high.x();
  int SU_low_y = SU_low.y();
  int SU_high_y = SU_high.y();
  int SU_low_z = SU_low.z();
  int SU_high_z = SU_high.z();
  Uintah::IntVector old_den_low = old_den.offset();
  Uintah::IntVector old_den_high = old_den.size() + old_den_low - Uintah::IntVector(1, 1, 1);
  int old_den_low_x = old_den_low.x();
  int old_den_high_x = old_den_high.x();
  int old_den_low_y = old_den_low.y();
  int old_den_high_y = old_den_high.y();
  int old_den_low_z = old_den_low.z();
  int old_den_high_z = old_den_high.z();
  Uintah::IntVector old_UU_low = old_UU.offset();
  Uintah::IntVector old_UU_high = old_UU.size() + old_UU_low - Uintah::IntVector(1, 1, 1);
  int old_UU_low_x = old_UU_low.x();
  int old_UU_high_x = old_UU_high.x();
  int old_UU_low_y = old_UU_low.y();
  int old_UU_high_y = old_UU_high.y();
  int old_UU_low_z = old_UU_low.z();
  int old_UU_high_z = old_UU_high.z();
  Uintah::IntVector eps_low = eps.offset();
  Uintah::IntVector eps_high = eps.size() + eps_low - Uintah::IntVector(1, 1, 1);
  int eps_low_x = eps_low.x();
  int eps_high_x = eps_high.x();
  int eps_low_y = eps_low.y();
  int eps_high_y = eps_high.y();
  int eps_low_z = eps_low.z();
  int eps_high_z = eps_high.z();
  int ceeu_low = ceeu.low();
  int ceeu_high = ceeu.high();
  int cweu_low = cweu.low();
  int cweu_high = cweu.high();
  int cwwu_low = cwwu.low();
  int cwwu_high = cwwu.high();
  int cnn_low = cnn.low();
  int cnn_high = cnn.high();
  int csn_low = csn.low();
  int csn_high = csn.high();
  int css_low = css.low();
  int css_high = css.high();
  int ctt_low = ctt.low();
  int ctt_high = ctt.high();
  int cbt_low = cbt.low();
  int cbt_high = cbt.high();
  int cbb_low = cbb.low();
  int cbb_high = cbb.high();
  int sewu_low = sewu.low();
  int sewu_high = sewu.high();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  int dxepu_low = dxepu.low();
  int dxepu_high = dxepu.high();
  int dxpwu_low = dxpwu.low();
  int dxpwu_high = dxpwu.high();
  int dxpw_low = dxpw.low();
  int dxpw_high = dxpw.high();
  int dynp_low = dynp.low();
  int dynp_high = dynp.high();
  int dyps_low = dyps.low();
  int dyps_high = dyps.high();
  int dztp_low = dztp.low();
  int dztp_high = dztp.high();
  int dzpb_low = dzpb.low();
  int dzpb_high = dzpb.high();
  int fac1u_low = fac1u.low();
  int fac1u_high = fac1u.high();
  int fac2u_low = fac2u.low();
  int fac2u_high = fac2u.high();
  int fac3u_low = fac3u.low();
  int fac3u_high = fac3u.high();
  int fac4u_low = fac4u.low();
  int fac4u_high = fac4u.high();
  int iesdu_low = iesdu.low();
  int iesdu_high = iesdu.high();
  int iwsdu_low = iwsdu.low();
  int iwsdu_high = iwsdu.high();
  int nfac_low = nfac.low();
  int nfac_high = nfac.high();
  int sfac_low = sfac.low();
  int sfac_high = sfac.high();
  int tfac_low = tfac.low();
  int tfac_high = tfac.high();
  int bfac_low = bfac.low();
  int bfac_high = bfac.high();
  int fac1ns_low = fac1ns.low();
  int fac1ns_high = fac1ns.high();
  int fac2ns_low = fac2ns.low();
  int fac2ns_high = fac2ns.high();
  int fac3ns_low = fac3ns.low();
  int fac3ns_high = fac3ns.high();
  int fac4ns_low = fac4ns.low();
  int fac4ns_high = fac4ns.high();
  int n_shift_low = n_shift.low();
  int n_shift_high = n_shift.high();
  int s_shift_low = s_shift.low();
  int s_shift_high = s_shift.high();
  int fac1tb_low = fac1tb.low();
  int fac1tb_high = fac1tb.high();
  int fac2tb_low = fac2tb.low();
  int fac2tb_high = fac2tb.high();
  int fac3tb_low = fac3tb.low();
  int fac3tb_high = fac3tb.high();
  int fac4tb_low = fac4tb.low();
  int fac4tb_high = fac4tb.high();
  int t_shift_low = t_shift.low();
  int t_shift_high = t_shift.high();
  int b_shift_low = b_shift.low();
  int b_shift_high = b_shift.high();
  uvelcoef_( &uu_low_x, &uu_low_y, &uu_low_z, &uu_high_x, &uu_high_y, &uu_high_z, const_cast<double*>(uu.getPointer()),
             &cesav_low_x, &cesav_low_y, &cesav_low_z, &cesav_high_x, &cesav_high_y, &cesav_high_z, cesav.getPointer(),
             &cwsav_low_x, &cwsav_low_y, &cwsav_low_z, &cwsav_high_x, &cwsav_high_y, &cwsav_high_z, cwsav.getPointer(),
             &cnsav_low_x, &cnsav_low_y, &cnsav_low_z, &cnsav_high_x, &cnsav_high_y, &cnsav_high_z, cnsav.getPointer(),
             &cssav_low_x, &cssav_low_y, &cssav_low_z, &cssav_high_x, &cssav_high_y, &cssav_high_z, cssav.getPointer(),
             &ctsav_low_x, &ctsav_low_y, &ctsav_low_z, &ctsav_high_x, &ctsav_high_y, &ctsav_high_z, ctsav.getPointer(),
             &cbsav_low_x, &cbsav_low_y, &cbsav_low_z, &cbsav_high_x, &cbsav_high_y, &cbsav_high_z, cbsav.getPointer(),
             &ap_low_x, &ap_low_y, &ap_low_z, &ap_high_x, &ap_high_y, &ap_high_z, ap.getPointer(),
             &ae_low_x, &ae_low_y, &ae_low_z, &ae_high_x, &ae_high_y, &ae_high_z, ae.getPointer(),
             &aw_low_x, &aw_low_y, &aw_low_z, &aw_high_x, &aw_high_y, &aw_high_z, aw.getPointer(),
             &an_low_x, &an_low_y, &an_low_z, &an_high_x, &an_high_y, &an_high_z, an.getPointer(),
             &as_low_x, &as_low_y, &as_low_z, &as_high_x, &as_high_y, &as_high_z, as.getPointer(),
             &at_low_x, &at_low_y, &at_low_z, &at_high_x, &at_high_y, &at_high_z, at.getPointer(),
             &ab_low_x, &ab_low_y, &ab_low_z, &ab_high_x, &ab_high_y, &ab_high_z, ab.getPointer(),
             &vv_low_x, &vv_low_y, &vv_low_z, &vv_high_x, &vv_high_y, &vv_high_z, const_cast<double*>(vv.getPointer()),
             &ww_low_x, &ww_low_y, &ww_low_z, &ww_high_x, &ww_high_y, &ww_high_z, const_cast<double*>(ww.getPointer()),
             &den_low_x, &den_low_y, &den_low_z, &den_high_x, &den_high_y, &den_high_z, const_cast<double*>(den.getPointer()),
             &vis_low_x, &vis_low_y, &vis_low_z, &vis_high_x, &vis_high_y, &vis_high_z, const_cast<double*>(vis.getPointer()),
             &den_ref_low_x, &den_ref_low_y, &den_ref_low_z, &den_ref_high_x, &den_ref_high_y, &den_ref_high_z, const_cast<double*>(den_ref.getPointer()),
             &SU_low_x, &SU_low_y, &SU_low_z, &SU_high_x, &SU_high_y, &SU_high_z, SU.getPointer(),
             &old_den_low_x, &old_den_low_y, &old_den_low_z, &old_den_high_x, &old_den_high_y, &old_den_high_z, const_cast<double*>(old_den.getPointer()),
             &old_UU_low_x, &old_UU_low_y, &old_UU_low_z, &old_UU_high_x, &old_UU_high_y, &old_UU_high_z, const_cast<double*>(old_UU.getPointer()),
             &eps_low_x, &eps_low_y, &eps_low_z, &eps_high_x, &eps_high_y, &eps_high_z, const_cast<double*>(eps.getPointer()),
             &deltat,
             &grav,
             &lcend,
             &ceeu_low, &ceeu_high, ceeu.get_objs(),
             &cweu_low, &cweu_high, cweu.get_objs(),
             &cwwu_low, &cwwu_high, cwwu.get_objs(),
             &cnn_low, &cnn_high, cnn.get_objs(),
             &csn_low, &csn_high, csn.get_objs(),
             &css_low, &css_high, css.get_objs(),
             &ctt_low, &ctt_high, ctt.get_objs(),
             &cbt_low, &cbt_high, cbt.get_objs(),
             &cbb_low, &cbb_high, cbb.get_objs(),
             &sewu_low, &sewu_high, sewu.get_objs(),
             &sew_low, &sew_high, sew.get_objs(),
             &sns_low, &sns_high, sns.get_objs(),
             &stb_low, &stb_high, stb.get_objs(),
             &dxepu_low, &dxepu_high, dxepu.get_objs(),
             &dxpwu_low, &dxpwu_high, dxpwu.get_objs(),
             &dxpw_low, &dxpw_high, dxpw.get_objs(),
             &dynp_low, &dynp_high, dynp.get_objs(),
             &dyps_low, &dyps_high, dyps.get_objs(),
             &dztp_low, &dztp_high, dztp.get_objs(),
             &dzpb_low, &dzpb_high, dzpb.get_objs(),
             &fac1u_low, &fac1u_high, fac1u.get_objs(),
             &fac2u_low, &fac2u_high, fac2u.get_objs(),
             &fac3u_low, &fac3u_high, fac3u.get_objs(),
             &fac4u_low, &fac4u_high, fac4u.get_objs(),
             &iesdu_low, &iesdu_high, iesdu.get_objs(),
             &iwsdu_low, &iwsdu_high, iwsdu.get_objs(),
             &nfac_low, &nfac_high, nfac.get_objs(),
             &sfac_low, &sfac_high, sfac.get_objs(),
             &tfac_low, &tfac_high, tfac.get_objs(),
             &bfac_low, &bfac_high, bfac.get_objs(),
             &fac1ns_low, &fac1ns_high, fac1ns.get_objs(),
             &fac2ns_low, &fac2ns_high, fac2ns.get_objs(),
             &fac3ns_low, &fac3ns_high, fac3ns.get_objs(),
             &fac4ns_low, &fac4ns_high, fac4ns.get_objs(),
             &n_shift_low, &n_shift_high, n_shift.get_objs(),
             &s_shift_low, &s_shift_high, s_shift.get_objs(),
             &fac1tb_low, &fac1tb_high, fac1tb.get_objs(),
             &fac2tb_low, &fac2tb_high, fac2tb.get_objs(),
             &fac3tb_low, &fac3tb_high, fac3tb.get_objs(),
             &fac4tb_low, &fac4tb_high, fac4tb.get_objs(),
             &t_shift_low, &t_shift_high, t_shift.get_objs(),
             &b_shift_low, &b_shift_high, b_shift.get_objs(),
             idxLoU.get_pointer(),
             idxHiU.get_pointer() );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine uvelcoef(uu_low_x, uu_low_y, uu_low_z, uu_high_x, 
     & uu_high_y, uu_high_z, uu, cesav_low_x, cesav_low_y, cesav_low_z,
     &  cesav_high_x, cesav_high_y, cesav_high_z, cesav, cwsav_low_x, 
     & cwsav_low_y, cwsav_low_z, cwsav_high_x, cwsav_high_y, 
     & cwsav_high_z, cwsav, cnsav_low_x, cnsav_low_y, cnsav_low_z, 
     & cnsav_high_x, cnsav_high_y, cnsav_high_z, cnsav, cssav_low_x, 
     & cssav_low_y, cssav_low_z, cssav_high_x, cssav_high_y, 
     & cssav_high_z, cssav, ctsav_low_x, ctsav_low_y, ctsav_low_z, 
     & ctsav_high_x, ctsav_high_y, ctsav_high_z, ctsav, cbsav_low_x, 
     & cbsav_low_y, cbsav_low_z, cbsav_high_x, cbsav_high_y, 
     & cbsav_high_z, cbsav, ap_low_x, ap_low_y, ap_low_z, ap_high_x, 
     & ap_high_y, ap_high_z, ap, ae_low_x, ae_low_y, ae_low_z, 
     & ae_high_x, ae_high_y, ae_high_z, ae, aw_low_x, aw_low_y, 
     & aw_low_z, aw_high_x, aw_high_y, aw_high_z, aw, an_low_x, 
     & an_low_y, an_low_z, an_high_x, an_high_y, an_high_z, an, 
     & as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, as_high_z, 
     & as, at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z, at, ab_low_x, ab_low_y, ab_low_z, ab_high_x, 
     & ab_high_y, ab_high_z, ab, vv_low_x, vv_low_y, vv_low_z, 
     & vv_high_x, vv_high_y, vv_high_z, vv, ww_low_x, ww_low_y, 
     & ww_low_z, ww_high_x, ww_high_y, ww_high_z, ww, den_low_x, 
     & den_low_y, den_low_z, den_high_x, den_high_y, den_high_z, den, 
     & vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z, vis, den_ref_low_x, den_ref_low_y, den_ref_low_z, 
     & den_ref_high_x, den_ref_high_y, den_ref_high_z, den_ref, 
     & SU_low_x, SU_low_y, SU_low_z, SU_high_x, SU_high_y, SU_high_z, 
     & SU, old_den_low_x, old_den_low_y, old_den_low_z, old_den_high_x,
     &  old_den_high_y, old_den_high_z, old_den, old_UU_low_x, 
     & old_UU_low_y, old_UU_low_z, old_UU_high_x, old_UU_high_y, 
     & old_UU_high_z, old_UU, eps_low_x, eps_low_y, eps_low_z, 
     & eps_high_x, eps_high_y, eps_high_z, eps, deltat, grav, lcend, 
     & ceeu_low, ceeu_high, ceeu, cweu_low, cweu_high, cweu, cwwu_low, 
     & cwwu_high, cwwu, cnn_low, cnn_high, cnn, csn_low, csn_high, csn,
     &  css_low, css_high, css, ctt_low, ctt_high, ctt, cbt_low, 
     & cbt_high, cbt, cbb_low, cbb_high, cbb, sewu_low, sewu_high, sewu
     & , sew_low, sew_high, sew, sns_low, sns_high, sns, stb_low, 
     & stb_high, stb, dxepu_low, dxepu_high, dxepu, dxpwu_low, 
     & dxpwu_high, dxpwu, dxpw_low, dxpw_high, dxpw, dynp_low, 
     & dynp_high, dynp, dyps_low, dyps_high, dyps, dztp_low, dztp_high,
     &  dztp, dzpb_low, dzpb_high, dzpb, fac1u_low, fac1u_high, fac1u, 
     & fac2u_low, fac2u_high, fac2u, fac3u_low, fac3u_high, fac3u, 
     & fac4u_low, fac4u_high, fac4u, iesdu_low, iesdu_high, iesdu, 
     & iwsdu_low, iwsdu_high, iwsdu, nfac_low, nfac_high, nfac, 
     & sfac_low, sfac_high, sfac, tfac_low, tfac_high, tfac, bfac_low, 
     & bfac_high, bfac, fac1ns_low, fac1ns_high, fac1ns, fac2ns_low, 
     & fac2ns_high, fac2ns, fac3ns_low, fac3ns_high, fac3ns, fac4ns_low
     & , fac4ns_high, fac4ns, n_shift_low, n_shift_high, n_shift, 
     & s_shift_low, s_shift_high, s_shift, fac1tb_low, fac1tb_high, 
     & fac1tb, fac2tb_low, fac2tb_high, fac2tb, fac3tb_low, fac3tb_high
     & , fac3tb, fac4tb_low, fac4tb_high, fac4tb, t_shift_low, 
     & t_shift_high, t_shift, b_shift_low, b_shift_high, b_shift, 
     & idxLoU, idxHiU)

      implicit none
      integer uu_low_x, uu_low_y, uu_low_z, uu_high_x, uu_high_y, 
     & uu_high_z
      double precision uu(uu_low_x:uu_high_x, uu_low_y:uu_high_y, 
     & uu_low_z:uu_high_z)
      integer cesav_low_x, cesav_low_y, cesav_low_z, cesav_high_x, 
     & cesav_high_y, cesav_high_z
      double precision cesav(cesav_low_x:cesav_high_x, cesav_low_y:
     & cesav_high_y, cesav_low_z:cesav_high_z)
      integer cwsav_low_x, cwsav_low_y, cwsav_low_z, cwsav_high_x, 
     & cwsav_high_y, cwsav_high_z
      double precision cwsav(cwsav_low_x:cwsav_high_x, cwsav_low_y:
     & cwsav_high_y, cwsav_low_z:cwsav_high_z)
      integer cnsav_low_x, cnsav_low_y, cnsav_low_z, cnsav_high_x, 
     & cnsav_high_y, cnsav_high_z
      double precision cnsav(cnsav_low_x:cnsav_high_x, cnsav_low_y:
     & cnsav_high_y, cnsav_low_z:cnsav_high_z)
      integer cssav_low_x, cssav_low_y, cssav_low_z, cssav_high_x, 
     & cssav_high_y, cssav_high_z
      double precision cssav(cssav_low_x:cssav_high_x, cssav_low_y:
     & cssav_high_y, cssav_low_z:cssav_high_z)
      integer ctsav_low_x, ctsav_low_y, ctsav_low_z, ctsav_high_x, 
     & ctsav_high_y, ctsav_high_z
      double precision ctsav(ctsav_low_x:ctsav_high_x, ctsav_low_y:
     & ctsav_high_y, ctsav_low_z:ctsav_high_z)
      integer cbsav_low_x, cbsav_low_y, cbsav_low_z, cbsav_high_x, 
     & cbsav_high_y, cbsav_high_z
      double precision cbsav(cbsav_low_x:cbsav_high_x, cbsav_low_y:
     & cbsav_high_y, cbsav_low_z:cbsav_high_z)
      integer ap_low_x, ap_low_y, ap_low_z, ap_high_x, ap_high_y, 
     & ap_high_z
      double precision ap(ap_low_x:ap_high_x, ap_low_y:ap_high_y, 
     & ap_low_z:ap_high_z)
      integer ae_low_x, ae_low_y, ae_low_z, ae_high_x, ae_high_y, 
     & ae_high_z
      double precision ae(ae_low_x:ae_high_x, ae_low_y:ae_high_y, 
     & ae_low_z:ae_high_z)
      integer aw_low_x, aw_low_y, aw_low_z, aw_high_x, aw_high_y, 
     & aw_high_z
      double precision aw(aw_low_x:aw_high_x, aw_low_y:aw_high_y, 
     & aw_low_z:aw_high_z)
      integer an_low_x, an_low_y, an_low_z, an_high_x, an_high_y, 
     & an_high_z
      double precision an(an_low_x:an_high_x, an_low_y:an_high_y, 
     & an_low_z:an_high_z)
      integer as_low_x, as_low_y, as_low_z, as_high_x, as_high_y, 
     & as_high_z
      double precision as(as_low_x:as_high_x, as_low_y:as_high_y, 
     & as_low_z:as_high_z)
      integer at_low_x, at_low_y, at_low_z, at_high_x, at_high_y, 
     & at_high_z
      double precision at(at_low_x:at_high_x, at_low_y:at_high_y, 
     & at_low_z:at_high_z)
      integer ab_low_x, ab_low_y, ab_low_z, ab_high_x, ab_high_y, 
     & ab_high_z
      double precision ab(ab_low_x:ab_high_x, ab_low_y:ab_high_y, 
     & ab_low_z:ab_high_z)
      integer vv_low_x, vv_low_y, vv_low_z, vv_high_x, vv_high_y, 
     & vv_high_z
      double precision vv(vv_low_x:vv_high_x, vv_low_y:vv_high_y, 
     & vv_low_z:vv_high_z)
      integer ww_low_x, ww_low_y, ww_low_z, ww_high_x, ww_high_y, 
     & ww_high_z
      double precision ww(ww_low_x:ww_high_x, ww_low_y:ww_high_y, 
     & ww_low_z:ww_high_z)
      integer den_low_x, den_low_y, den_low_z, den_high_x, den_high_y, 
     & den_high_z
      double precision den(den_low_x:den_high_x, den_low_y:den_high_y, 
     & den_low_z:den_high_z)
      integer vis_low_x, vis_low_y, vis_low_z, vis_high_x, vis_high_y, 
     & vis_high_z
      double precision vis(vis_low_x:vis_high_x, vis_low_y:vis_high_y, 
     & vis_low_z:vis_high_z)
      integer den_ref_low_x, den_ref_low_y, den_ref_low_z, 
     & den_ref_high_x, den_ref_high_y, den_ref_high_z
      double precision den_ref(den_ref_low_x:den_ref_high_x, 
     & den_ref_low_y:den_ref_high_y, den_ref_low_z:den_ref_high_z)
      integer SU_low_x, SU_low_y, SU_low_z, SU_high_x, SU_high_y, 
     & SU_high_z
      double precision SU(SU_low_x:SU_high_x, SU_low_y:SU_high_y, 
     & SU_low_z:SU_high_z)
      integer old_den_low_x, old_den_low_y, old_den_low_z, 
     & old_den_high_x, old_den_high_y, old_den_high_z
      double precision old_den(old_den_low_x:old_den_high_x, 
     & old_den_low_y:old_den_high_y, old_den_low_z:old_den_high_z)
      integer old_UU_low_x, old_UU_low_y, old_UU_low_z, old_UU_high_x, 
     & old_UU_high_y, old_UU_high_z
      double precision old_UU(old_UU_low_x:old_UU_high_x, old_UU_low_y:
     & old_UU_high_y, old_UU_low_z:old_UU_high_z)
      integer eps_low_x, eps_low_y, eps_low_z, eps_high_x, eps_high_y, 
     & eps_high_z
      double precision eps(eps_low_x:eps_high_x, eps_low_y:eps_high_y, 
     & eps_low_z:eps_high_z)
      double precision deltat
      double precision grav
      logical*1 lcend
      integer ceeu_low
      integer ceeu_high
      double precision ceeu(ceeu_low:ceeu_high)
      integer cweu_low
      integer cweu_high
      double precision cweu(cweu_low:cweu_high)
      integer cwwu_low
      integer cwwu_high
      double precision cwwu(cwwu_low:cwwu_high)
      integer cnn_low
      integer cnn_high
      double precision cnn(cnn_low:cnn_high)
      integer csn_low
      integer csn_high
      double precision csn(csn_low:csn_high)
      integer css_low
      integer css_high
      double precision css(css_low:css_high)
      integer ctt_low
      integer ctt_high
      double precision ctt(ctt_low:ctt_high)
      integer cbt_low
      integer cbt_high
      double precision cbt(cbt_low:cbt_high)
      integer cbb_low
      integer cbb_high
      double precision cbb(cbb_low:cbb_high)
      integer sewu_low
      integer sewu_high
      double precision sewu(sewu_low:sewu_high)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer dxepu_low
      integer dxepu_high
      double precision dxepu(dxepu_low:dxepu_high)
      integer dxpwu_low
      integer dxpwu_high
      double precision dxpwu(dxpwu_low:dxpwu_high)
      integer dxpw_low
      integer dxpw_high
      double precision dxpw(dxpw_low:dxpw_high)
      integer dynp_low
      integer dynp_high
      double precision dynp(dynp_low:dynp_high)
      integer dyps_low
      integer dyps_high
      double precision dyps(dyps_low:dyps_high)
      integer dztp_low
      integer dztp_high
      double precision dztp(dztp_low:dztp_high)
      integer dzpb_low
      integer dzpb_high
      double precision dzpb(dzpb_low:dzpb_high)
      integer fac1u_low
      integer fac1u_high
      double precision fac1u(fac1u_low:fac1u_high)
      integer fac2u_low
      integer fac2u_high
      double precision fac2u(fac2u_low:fac2u_high)
      integer fac3u_low
      integer fac3u_high
      double precision fac3u(fac3u_low:fac3u_high)
      integer fac4u_low
      integer fac4u_high
      double precision fac4u(fac4u_low:fac4u_high)
      integer iesdu_low
      integer iesdu_high
      integer iesdu(iesdu_low:iesdu_high)
      integer iwsdu_low
      integer iwsdu_high
      integer iwsdu(iwsdu_low:iwsdu_high)
      integer nfac_low
      integer nfac_high
      double precision nfac(nfac_low:nfac_high)
      integer sfac_low
      integer sfac_high
      double precision sfac(sfac_low:sfac_high)
      integer tfac_low
      integer tfac_high
      double precision tfac(tfac_low:tfac_high)
      integer bfac_low
      integer bfac_high
      double precision bfac(bfac_low:bfac_high)
      integer fac1ns_low
      integer fac1ns_high
      double precision fac1ns(fac1ns_low:fac1ns_high)
      integer fac2ns_low
      integer fac2ns_high
      double precision fac2ns(fac2ns_low:fac2ns_high)
      integer fac3ns_low
      integer fac3ns_high
      double precision fac3ns(fac3ns_low:fac3ns_high)
      integer fac4ns_low
      integer fac4ns_high
      double precision fac4ns(fac4ns_low:fac4ns_high)
      integer n_shift_low
      integer n_shift_high
      integer n_shift(n_shift_low:n_shift_high)
      integer s_shift_low
      integer s_shift_high
      integer s_shift(s_shift_low:s_shift_high)
      integer fac1tb_low
      integer fac1tb_high
      double precision fac1tb(fac1tb_low:fac1tb_high)
      integer fac2tb_low
      integer fac2tb_high
      double precision fac2tb(fac2tb_low:fac2tb_high)
      integer fac3tb_low
      integer fac3tb_high
      double precision fac3tb(fac3tb_low:fac3tb_high)
      integer fac4tb_low
      integer fac4tb_high
      double precision fac4tb(fac4tb_low:fac4tb_high)
      integer t_shift_low
      integer t_shift_high
      integer t_shift(t_shift_low:t_shift_high)
      integer b_shift_low
      integer b_shift_high
      integer b_shift(b_shift_low:b_shift_high)
      integer idxLoU(3)
      integer idxHiU(3)
#endif /* __cplusplus */

#endif /* fspec_uvelcoef */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
