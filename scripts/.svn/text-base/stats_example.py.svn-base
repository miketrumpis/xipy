#!/usr/bin/env python
from nutmeg.core.tfbeam import tfbeam_from_file
from nutmeg.stats import beam_stats as bstat
from nutmeg.vis import ortho_plot_window_qt4 as plotter
import glob
import os, sys

def perform_contrast_ttest(dir=None, ptr=None):
    if dir is not None:
        glob_str = os.path.join(dir, 's_beam*spatnorm*mat')
        mat_files = glob.glob(glob_str)
        # get all beams in dir, with subject and condition labels for each beam
        beam_list = [tfbeam_from_file(mfile, fixed_comparison='f db')
                     for mfile in mat_files]
        subj_labels = range(1,len(beam_list)+1)
        cond_labels = [1] * len(subj_labels)

        a_comp = bstat.BeamContrastAverager(beam_list,
                                            subj_labels,
                                            cond_labels)
    elif ptr is not None:
        a_comp = bstat.BeamContrastAverager.from_matlab_ptr_file(
            ptr,
            fixed_comparison='f db'
            )

    # do an activation test on condition 1
    one_samp_t = bstat.SnPMOneSampT(a_comp, [1], force_full_perms=True)
    _, avg_beams = a_comp.compare(conditions=[1])
    t_maps, pvals_pos, pvals_neg = one_samp_t.test(correct_tpts=True,
                                                   correct_fpts=True)
    return avg_beams, t_maps, pvals_pos, pvals_neg

def perform_activation_ttest(dir=None, ptr=None):
    if dir is not None:
        glob_str = os.path.join(dir, 's_beam*spatnorm*mat')
        mat_files = glob.glob(glob_str)
        # get all beams in dir, with subject and condition labels for each beam
        beam_list = [tfbeam_from_file(mfile, fixed_comparison='f db')
                     for mfile in mat_files]
        subj_labels = range(1,len(beam_list)+1)
        cond_labels = [1] * len(subj_labels)

        a_comp = bstat.BeamActivationAverager(beam_list,
                                              subj_labels,
                                              cond_labels)
    elif ptr is not None:
        a_comp = bstat.BeamActivationAverager.from_matlab_ptr_file(ptr,
                                                            ratio_type='f db')

    # do an activation test on condition 1
    one_samp_t = bstat.SnPMOneSampT(a_comp, [1], force_full_perms=True)
    _, avg_beams = a_comp.compare(conditions=[1])
    t_maps, pvals_pos, pvals_neg = one_samp_t.test(correct_tpts=True,
                                                   correct_fpts=True)
    return avg_beams, t_maps, pvals_pos, pvals_neg

if __name__=='__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]

    if os.path.splitext(arg)[1] == '.mat':
        activ, t_maps, pvals_pos, pvals_neg = perform_activation_ttest(ptr=arg)
        dir = os.environ['PWD']
    else:
        activ, t_maps, pvals_pos, pvals_neg = perform_activation_ttest(dir=arg)
    
    #plotter.view_beam(t_maps[0])
    activ[0]._to_recarray('avg_activation')
    t_maps[0]._to_recarray('tmaps')
    pvals_pos[0]._to_recarray('pvals_pos')
    pvals_neg[0]._to_recarray('pvals_neg')

