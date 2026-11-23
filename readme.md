# Spin_Splitting_Ge_ABS <br>
Follow this repository to see the codes related to simulating the spin splitting of Andreev Bound Spectrum with germanium as the weak link. <br>

Following files in src folder are important and contain the models <br>
1. src/Operators2D.py <br>
2. src/Minimal_model_1D.py <br>
3. src/Minimal_model_1D_1sub.py <br>
4. src/Minimal_model_2D.py(has disorder included in it)<br>
5. src/charge_trap_pot_model.py<br>
6. src/charge_trap_pot_model_neg_charge.py<br>
7. src/charge_trap_pot_model_4_layers.py<br>
8. src/Eff_heavyhole_model_CSOC_parameters.py (is used to find out Ge parameters using eff heavy hole model)  <br>

All the required libraries and global constants are imported in each of the above scipt using  <br>
9. src/config.py file <br>

10. Figure_jupyter_notebooks/Eff_mass_and_SO_vs_Lz_v2.ipynb - The actual parameters used in the paper are found in this file.  <br>

These are the analysis files for the manuscript.  <br>
11. Fig1 - Figure_jupyter_notebooks/Andreev_Bound_Spectrum.ipynb <br>
12. Fig2 - Figure_jupyter_notebooks/Variation_ss_soc_linear_cubic_2d_together_75nm.ipynb <br>
13. Fig3 - Figure_jupyter_notebooks/lsoc_csoc_spectrum.ipynb, Figure_jupyter_notebooks/\lsoc_csoc_spectrum_zoom.ipynb<br>
14. Fig4 - Figure_jupyter_notebooks/Es_vs_omega_mu__zero_gradient_csoc_lsoc_together_final.ipynb, <br>Figure_jupyter_notebooks/cubic_linear_spin_Splitting_vs_yct_analysis_together_4_layers.ipynb <br>
Files for dataset generation:Figure_jupyter_notebooks/cubic_spin_splitting_charge_trap_effect_image_charge_model_4_layers_v4.ipynb, Figure_jupyter_notebooks/linear_spin_splitting_charge_trap_effect_image_charge_model_4_layers_v4.ipynb <br>
15. Appendix D - Figure_jupyter_notebooks/sheet_of_charge_v2.ipynb, src/Image_charge_3d.py <br>

While running file, pathing may need to be changed, as it was set as per original directory. 