# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from the following repository:
# https://github.com/yuvalatzmon/COSMO/
#
# The license for the original version of this file can be
# found in this directory (LICENSE_COSMO). The modifications
# to this file are subject to the License
# located at the root directory.
# ---------------------------------------------------------------



import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc


def _AUSUC(Acc_tr, Acc_ts):

    """ Calc area under seen-unseen curve
        Source: https://github.com/yuvalatzmon/COSMO/blob/master/src/metrics.py
    """

    # Sort by X axis
    X_sorted_arg = np.argsort(Acc_tr)
    sorted_X = np.array(Acc_tr)[X_sorted_arg]
    sorted_Y = np.array(Acc_ts)[X_sorted_arg]

    # zero pad
    leftmost_X, leftmost_Y = 0, sorted_Y[0]
    rightmost_X, rightmost_Y = sorted_X[-1], 0
    sorted_X = np.block([np.array([leftmost_X]), sorted_X, np.array([rightmost_X])])
    sorted_Y = np.block([np.array([leftmost_Y]), sorted_Y, np.array([rightmost_Y])])

    # eval AUC
    AUSUC = auc(sorted_X, sorted_Y)

    return AUSUC


def calc_cs_ausuc(pred, y_gt, seen_classses, unseen_classses, use_balanced_accuracy=True, gamma_range=None, verbose=True):
    """ Calc area under seen-unseen curve, according to calibrated stacking (Chao et al. 2016)
        Adapted from: https://github.com/yuvalatzmon/COSMO/blob/master/src/metrics.py

        NOTE: pred cannot accept -np.inf values
    """

    assert(pred.min() != -np.inf)

    # make numbers positive with min = 0
    # pred = pred.copy() - pred.min()

    if gamma_range is None:
        # make a log spaced search range
        gamma = abs((pred[:, unseen_classses].max(dim=1)[0] - pred[:, seen_classses].max(dim=1)[0]).min()) + abs(
            (pred[:, unseen_classses].max(dim=1)[0] - pred[:, seen_classses].max(dim=1)[0]).max())
        gamma = gamma.item()
        pos_range = (-(np.logspace(0, np.log10(gamma + 1), 50) - 1)).tolist()
        neg_range = (np.logspace(0, np.log10(gamma +1), 50) - 1).tolist()
        gamma_range = sorted(pos_range + neg_range)

    # torch.cuda.empty_cache()
    Acc_tr_values, Acc_ts_values = calc_acc_tr_ts_over_gamma_range(gamma_range, pred, y_gt, seen_classses, unseen_classses, use_balanced_accuracy)
    # torch.cuda.empty_cache()

    cs_ausuc = _AUSUC(Acc_tr=Acc_tr_values, Acc_ts=Acc_ts_values)
    if min(Acc_tr_values) > 0.01:
        print(f'CS AUSUC ERROR: Increase gamma range (add low values), because min(Acc_tr_values) equals {min(Acc_tr_values)}')
        return np.nan
    if min(Acc_ts_values) > 0.01:
        print(f'CS AUSUC ERROR: Increase gamma range (add high values), because min(Acc_ts_values) equals {min(Acc_ts_values)}')
        return np.nan
    if verbose:
        print('AUSUC: max(acc_seen) = ', max(Acc_tr_values))
        print('AUSUC: max(acc_unseen)', max(Acc_ts_values))
        print(f'AUSUC (by Calibrated Stacking) = {cs_ausuc:.3f}')
    return cs_ausuc


def calc_acc_tr_ts_over_gamma_range(gamma_range, pred, y_gt, seen_classses, unseen_classses, use_balanced_accuracy):
    Acc_tr_values = []
    Acc_ts_values = []

    for gamma in gamma_range:
        params = gamma, pred, seen_classses, unseen_classses, use_balanced_accuracy, y_gt
        Acc_tr, Acc_ts = cs_at_single_operating_point(params)
        Acc_tr_values.append(Acc_tr)
        Acc_ts_values.append(Acc_ts)

    return Acc_tr_values, Acc_ts_values


def cs_at_single_operating_point(params):
    gamma, pred, seen_classses, unseen_classses, use_balanced_accuracy, y_gt = params
    if isinstance(pred, torch.Tensor):
        cs_pred = pred.clone()
    else:
        cs_pred = pred.copy()

    cs_pred[:, seen_classses] -= gamma
    zs_metrics = ZSL_Metrics(seen_classses, unseen_classses)
    Acc_ts, Acc_tr, H = zs_metrics.generlized_scores(y_gt.cpu().numpy(), cs_pred, num_class_according_y_true=True,
                                                     use_balanced_accuracy=use_balanced_accuracy)
    cs_pred = None # release memory
    # torch.cuda.empty_cache()  # release memory

    return Acc_tr, Acc_ts


class ZSL_Metrics():
    def __init__(self, seen_classes, unseen_classes, report_entropy=False):
        self._seen_classes = np.sort(seen_classes)
        self._unseen_classes = np.sort(unseen_classes)
        self._n_seen = len(seen_classes)
        self._n_unseen = len(unseen_classes)
        self._report_entropy = report_entropy

        assert(self._n_seen == len(np.unique(seen_classes))) # sanity check
        assert(self._n_unseen == len(np.unique(unseen_classes))) # sanity check


    def unseen_balanced_accuracy(self, y_true, pred_softmax):
        Acc_zs, Ent_zs =  self._subset_classes_balanced_accuracy(y_true, pred_softmax,
                                                      self._unseen_classes)
        if self._report_entropy:
            return Acc_zs, Ent_zs
        else:
            return Acc_zs

    def seen_balanced_accuracy(self, y_true, pred_softmax):
        Acc_seen, Ent_seen = self._subset_classes_balanced_accuracy(y_true,
                                                                    pred_softmax,
                                                      self._seen_classes)
        if self._report_entropy:
            return Acc_seen, Ent_seen
        else:
            return Acc_seen

    def generlized_scores(self, y_true_cpu, pred_softmax, num_class_according_y_true=True, use_balanced_accuracy=True):

        Acc_ts, Ent_ts = self._generalized_unseen_accuracy(y_true_cpu, pred_softmax, num_class_according_y_true,
                                                           use_balanced_accuracy)
        Acc_tr, Ent_tr = self._generalized_seen_accuracy(y_true_cpu, pred_softmax, num_class_according_y_true,
                                                         use_balanced_accuracy)
        H = 2*Acc_tr*Acc_ts/(Acc_tr + Acc_ts + 1e-8)
        Ent_H = 2*Ent_tr*Ent_ts/(Ent_tr + Ent_ts + 1e-8)

        if self._report_entropy:
            return Acc_ts, Acc_tr, H, Ent_ts, Ent_tr, Ent_H
        else:
            return Acc_ts, Acc_tr, H

    def _generalized_unseen_accuracy(self, y_true_cpu, pred_softmax, num_class_according_y_true, use_balanced_accuracy):
        return self._generalized_subset_accuracy(y_true_cpu, pred_softmax,
                                                 self._unseen_classes, num_class_according_y_true, use_balanced_accuracy)

    def _generalized_seen_accuracy(self, y_true_cpu, pred_softmax, num_class_according_y_true, use_balanced_accuracy):
        return self._generalized_subset_accuracy(y_true_cpu, pred_softmax,
                                                 self._seen_classes, num_class_according_y_true, use_balanced_accuracy)

    def _generalized_subset_accuracy(self, y_true_cpu, pred_softmax, subset_classes, num_class_according_y_true,
                                     use_balanced_accuracy):
        is_member = np.in1d # np.in1d is like MATLAB's ismember
        ix_subset_samples = is_member(y_true_cpu, subset_classes)

        y_true_subset = y_true_cpu[ix_subset_samples]
        all_classes = np.sort(np.block([self._seen_classes, self._unseen_classes]))

        if isinstance(pred_softmax, torch.Tensor):
            amax = (pred_softmax[:, all_classes]).argmax(1).cpu()
        else:
            amax = (pred_softmax[:, all_classes]).argmax(1)

        y_pred = all_classes[amax]
        y_pred_subset = y_pred[ix_subset_samples]

        if use_balanced_accuracy:

            num_class = len(subset_classes)
            # infer number of classes according to unique(y_true_subset)
            if num_class_according_y_true:
                num_class = len(np.unique(y_true_subset))
            Acc = float(per_class_balanced_accuracy(y_true_subset, y_pred_subset, num_class))
        else:
            Acc = (y_true_subset == y_pred_subset).mean()

        # Ent = float(entropy2(pred_softmax[ix_subset_samples, :][:, all_classes]).mean())
        Ent = 0*Acc + 1e-3 # disabled because its too slow
        return Acc, Ent

    def _subset_classes_balanced_accuracy(self, y_true, pred_softmax, subset_classes):
        is_member = np.in1d # np.in1d is like MATLAB's ismember
        ix_subset_samples = is_member(y_true, subset_classes)

        y_true_zs = y_true[ix_subset_samples]
        y_pred = subset_classes[(pred_softmax[:, subset_classes]).argmax(dim=1)]
        y_pred_zs = y_pred[ix_subset_samples]

        Acc = float(per_class_balanced_accuracy(y_true_zs, y_pred_zs, len(subset_classes)))
        # Ent = float(entropy2(pred_softmax[:, subset_classes]).mean())
        Ent = 0*Acc + 1e-3 # disabled because its too slow
        return Acc, Ent


def per_class_balanced_accuracy(y_true, y_pred, num_class=None):
    """ A balanced accuracy metric as in Xian (CVPR 2017). Accuracy is
        evaluated individually per class, and then uniformly averaged between
        classes.
    """
    if len(y_true) == 0 or num_class == 0:
        return np.nan

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.flatten().cpu().numpy().astype('int32')
    else:
        y_true = y_true.flatten().astype('int32')

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()


    if num_class is None:
        num_class = len(np.unique(np.block([y_true, y_pred])))

    max_class_id = 1+max([num_class, y_true.max(), y_pred.max()])

    counts_per_class_s = pd.Series(y_true).value_counts()
    counts_per_class = np.zeros((max_class_id,))
    counts_per_class[counts_per_class_s.index] = counts_per_class_s.values

    accuracy = (1.*(y_pred == y_true) / counts_per_class[y_true]).sum() / num_class
    return accuracy.astype('float32')



def run_bash(cmd, raise_on_err=True, raise_on_warning=False, versbose=True, return_exist_code=False, err_ind_by_exitcode=False):
    """ This function takes Bash commands and return their stdout
    Returns: string (stdout)
    :type cmd: string
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, executable='/bin/bash')
    # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = out.strip().decode('utf-8')
    err = err.strip().decode('utf-8')
    exit_code = p.returncode
    is_err = err != ''
    if err_ind_by_exitcode:
        is_err = (exit_code != 0)

    if is_err and raise_on_err:
        do_raise = True
        if 'warning' in err.lower():
            do_raise = raise_on_warning
            if versbose and not raise_on_warning:
                print('command was: {}'.format(cmd))
            print(err, file=sys.stderr)

        if do_raise or 'error' in err.lower():
            if versbose:
                print('command was: {}'.format(cmd))
            raise RuntimeError(err)

    if return_exist_code:
        return out, exit_code
    else:
        return out  # This is the stdout from the shell command


@contextmanager
def temporary_random_numpy_seed(seed) -> object:
    """ From https://github.com/yuvalatzmon/COSMO/blob/master/src/utils/ml_utils.py#L701
        A context manager for a temporary random seed (only within context)
        When leaving the context the numpy random state is restored

        This function is inspired by http://stackoverflow.com/q/32679403, which is shared according to https://creativecommons.org/licenses/by-sa/3.0/

        License
        THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.

        BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO BE BOUND BY THE TERMS OF THIS LICENSE. TO THE EXTENT THIS LICENSE MAY BE CONSIDERED TO BE A CONTRACT, THE LICENSOR GRANTS YOU THE RIGHTS CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND CONDITIONS.

        1. Definitions

        "Adaptation" means a work based upon the Work, or upon the Work and other pre-existing works, such as a translation, adaptation, derivative work, arrangement of music or other alterations of a literary or artistic work, or phonogram or performance and includes cinematographic adaptations or any other form in which the Work may be recast, transformed, or adapted including in any form recognizably derived from the original, except that a work that constitutes a Collection will not be considered an Adaptation for the purpose of this License. For the avoidance of doubt, where the Work is a musical work, performance or phonogram, the synchronization of the Work in timed-relation with a moving image ("synching") will be considered an Adaptation for the purpose of this License.
        "Collection" means a collection of literary or artistic works, such as encyclopedias and anthologies, or performances, phonograms or broadcasts, or other works or subject matter other than works listed in Section 1(f) below, which, by reason of the selection and arrangement of their contents, constitute intellectual creations, in which the Work is included in its entirety in unmodified form along with one or more other contributions, each constituting separate and independent works in themselves, which together are assembled into a collective whole. A work that constitutes a Collection will not be considered an Adaptation (as defined below) for the purposes of this License.
        "Creative Commons Compatible License" means a license that is listed at https://creativecommons.org/compatiblelicenses that has been approved by Creative Commons as being essentially equivalent to this License, including, at a minimum, because that license: (i) contains terms that have the same purpose, meaning and effect as the License Elements of this License; and, (ii) explicitly permits the relicensing of adaptations of works made available under that license under this License or a Creative Commons jurisdiction license with the same License Elements as this License.
        "Distribute" means to make available to the public the original and copies of the Work or Adaptation, as appropriate, through sale or other transfer of ownership.
        "License Elements" means the following high-level license attributes as selected by Licensor and indicated in the title of this License: Attribution, ShareAlike.
        "Licensor" means the individual, individuals, entity or entities that offer(s) the Work under the terms of this License.
        "Original Author" means, in the case of a literary or artistic work, the individual, individuals, entity or entities who created the Work or if no individual or entity can be identified, the publisher; and in addition (i) in the case of a performance the actors, singers, musicians, dancers, and other persons who act, sing, deliver, declaim, play in, interpret or otherwise perform literary or artistic works or expressions of folklore; (ii) in the case of a phonogram the producer being the person or legal entity who first fixes the sounds of a performance or other sounds; and, (iii) in the case of broadcasts, the organization that transmits the broadcast.
        "Work" means the literary and/or artistic work offered under the terms of this License including without limitation any production in the literary, scientific and artistic domain, whatever may be the mode or form of its expression including digital form, such as a book, pamphlet and other writing; a lecture, address, sermon or other work of the same nature; a dramatic or dramatico-musical work; a choreographic work or entertainment in dumb show; a musical composition with or without words; a cinematographic work to which are assimilated works expressed by a process analogous to cinematography; a work of drawing, painting, architecture, sculpture, engraving or lithography; a photographic work to which are assimilated works expressed by a process analogous to photography; a work of applied art; an illustration, map, plan, sketch or three-dimensional work relative to geography, topography, architecture or science; a performance; a broadcast; a phonogram; a compilation of data to the extent it is protected as a copyrightable work; or a work performed by a variety or circus performer to the extent it is not otherwise considered a literary or artistic work.
        "You" means an individual or entity exercising rights under this License who has not previously violated the terms of this License with respect to the Work, or who has received express permission from the Licensor to exercise rights under this License despite a previous violation.
        "Publicly Perform" means to perform public recitations of the Work and to communicate to the public those public recitations, by any means or process, including by wire or wireless means or public digital performances; to make available to the public Works in such a way that members of the public may access these Works from a place and at a place individually chosen by them; to perform the Work to the public by any means or process and the communication to the public of the performances of the Work, including by public digital performance; to broadcast and rebroadcast the Work by any means including signs, sounds or images.
        "Reproduce" means to make copies of the Work by any means including without limitation by sound or visual recordings and the right of fixation and reproducing fixations of the Work, including storage of a protected performance or phonogram in digital form or other electronic medium.
        2. Fair Dealing Rights. Nothing in this License is intended to reduce, limit, or restrict any uses free from copyright or rights arising from limitations or exceptions that are provided for in connection with the copyright protection under copyright law or other applicable laws.

        3. License Grant. Subject to the terms and conditions of this License, Licensor hereby grants You a worldwide, royalty-free, non-exclusive, perpetual (for the duration of the applicable copyright) license to exercise the rights in the Work as stated below:

        to Reproduce the Work, to incorporate the Work into one or more Collections, and to Reproduce the Work as incorporated in the Collections;
        to create and Reproduce Adaptations provided that any such Adaptation, including any translation in any medium, takes reasonable steps to clearly label, demarcate or otherwise identify that changes were made to the original Work. For example, a translation could be marked "The original work was translated from English to Spanish," or a modification could indicate "The original work has been modified.";
        to Distribute and Publicly Perform the Work including as incorporated in Collections; and,
        to Distribute and Publicly Perform Adaptations.
        For the avoidance of doubt:

        Non-waivable Compulsory License Schemes. In those jurisdictions in which the right to collect royalties through any statutory or compulsory licensing scheme cannot be waived, the Licensor reserves the exclusive right to collect such royalties for any exercise by You of the rights granted under this License;
        Waivable Compulsory License Schemes. In those jurisdictions in which the right to collect royalties through any statutory or compulsory licensing scheme can be waived, the Licensor waives the exclusive right to collect such royalties for any exercise by You of the rights granted under this License; and,
        Voluntary License Schemes. The Licensor waives the right to collect royalties, whether individually or, in the event that the Licensor is a member of a collecting society that administers voluntary licensing schemes, via that society, from any exercise by You of the rights granted under this License.
        The above rights may be exercised in all media and formats whether now known or hereafter devised. The above rights include the right to make such modifications as are technically necessary to exercise the rights in other media and formats. Subject to Section 8(f), all rights not expressly granted by Licensor are hereby reserved.

        4. Restrictions. The license granted in Section 3 above is expressly made subject to and limited by the following restrictions:

        You may Distribute or Publicly Perform the Work only under the terms of this License. You must include a copy of, or the Uniform Resource Identifier (URI) for, this License with every copy of the Work You Distribute or Publicly Perform. You may not offer or impose any terms on the Work that restrict the terms of this License or the ability of the recipient of the Work to exercise the rights granted to that recipient under the terms of the License. You may not sublicense the Work. You must keep intact all notices that refer to this License and to the disclaimer of warranties with every copy of the Work You Distribute or Publicly Perform. When You Distribute or Publicly Perform the Work, You may not impose any effective technological measures on the Work that restrict the ability of a recipient of the Work from You to exercise the rights granted to that recipient under the terms of the License. This Section 4(a) applies to the Work as incorporated in a Collection, but this does not require the Collection apart from the Work itself to be made subject to the terms of this License. If You create a Collection, upon notice from any Licensor You must, to the extent practicable, remove from the Collection any credit as required by Section 4(c), as requested. If You create an Adaptation, upon notice from any Licensor You must, to the extent practicable, remove from the Adaptation any credit as required by Section 4(c), as requested.
        You may Distribute or Publicly Perform an Adaptation only under the terms of: (i) this License; (ii) a later version of this License with the same License Elements as this License; (iii) a Creative Commons jurisdiction license (either this or a later license version) that contains the same License Elements as this License (e.g., Attribution-ShareAlike 3.0 US)); (iv) a Creative Commons Compatible License. If you license the Adaptation under one of the licenses mentioned in (iv), you must comply with the terms of that license. If you license the Adaptation under the terms of any of the licenses mentioned in (i), (ii) or (iii) (the "Applicable License"), you must comply with the terms of the Applicable License generally and the following provisions: (I) You must include a copy of, or the URI for, the Applicable License with every copy of each Adaptation You Distribute or Publicly Perform; (II) You may not offer or impose any terms on the Adaptation that restrict the terms of the Applicable License or the ability of the recipient of the Adaptation to exercise the rights granted to that recipient under the terms of the Applicable License; (III) You must keep intact all notices that refer to the Applicable License and to the disclaimer of warranties with every copy of the Work as included in the Adaptation You Distribute or Publicly Perform; (IV) when You Distribute or Publicly Perform the Adaptation, You may not impose any effective technological measures on the Adaptation that restrict the ability of a recipient of the Adaptation from You to exercise the rights granted to that recipient under the terms of the Applicable License. This Section 4(b) applies to the Adaptation as incorporated in a Collection, but this does not require the Collection apart from the Adaptation itself to be made subject to the terms of the Applicable License.
        If You Distribute, or Publicly Perform the Work or any Adaptations or Collections, You must, unless a request has been made pursuant to Section 4(a), keep intact all copyright notices for the Work and provide, reasonable to the medium or means You are utilizing: (i) the name of the Original Author (or pseudonym, if applicable) if supplied, and/or if the Original Author and/or Licensor designate another party or parties (e.g., a sponsor institute, publishing entity, journal) for attribution ("Attribution Parties") in Licensor's copyright notice, terms of service or by other reasonable means, the name of such party or parties; (ii) the title of the Work if supplied; (iii) to the extent reasonably practicable, the URI, if any, that Licensor specifies to be associated with the Work, unless such URI does not refer to the copyright notice or licensing information for the Work; and (iv) , consistent with Ssection 3(b), in the case of an Adaptation, a credit identifying the use of the Work in the Adaptation (e.g., "French translation of the Work by Original Author," or "Screenplay based on original Work by Original Author"). The credit required by this Section 4(c) may be implemented in any reasonable manner; provided, however, that in the case of a Adaptation or Collection, at a minimum such credit will appear, if a credit for all contributing authors of the Adaptation or Collection appears, then as part of these credits and in a manner at least as prominent as the credits for the other contributing authors. For the avoidance of doubt, You may only use the credit required by this Section for the purpose of attribution in the manner set out above and, by exercising Your rights under this License, You may not implicitly or explicitly assert or imply any connection with, sponsorship or endorsement by the Original Author, Licensor and/or Attribution Parties, as appropriate, of You or Your use of the Work, without the separate, express prior written permission of the Original Author, Licensor and/or Attribution Parties.
        Except as otherwise agreed in writing by the Licensor or as may be otherwise permitted by applicable law, if You Reproduce, Distribute or Publicly Perform the Work either by itself or as part of any Adaptations or Collections, You must not distort, mutilate, modify or take other derogatory action in relation to the Work which would be prejudicial to the Original Author's honor or reputation. Licensor agrees that in those jurisdictions (e.g. Japan), in which any exercise of the right granted in Section 3(b) of this License (the right to make Adaptations) would be deemed to be a distortion, mutilation, modification or other derogatory action prejudicial to the Original Author's honor and reputation, the Licensor will waive or not assert, as appropriate, this Section, to the fullest extent permitted by the applicable national law, to enable You to reasonably exercise Your right under Section 3(b) of this License (right to make Adaptations) but not otherwise.
        5. Representations, Warranties and Disclaimer

        UNLESS OTHERWISE MUTUALLY AGREED TO BY THE PARTIES IN WRITING, LICENSOR OFFERS THE WORK AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE WORK, EXPRESS, IMPLIED, STATUTORY OR OTHERWISE, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTIBILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OF ABSENCE OF ERRORS, WHETHER OR NOT DISCOVERABLE. SOME JURISDICTIONS DO NOT ALLOW THE EXCLUSION OF IMPLIED WARRANTIES, SO SUCH EXCLUSION MAY NOT APPLY TO YOU.

        6. Limitation on Liability. EXCEPT TO THE EXTENT REQUIRED BY APPLICABLE LAW, IN NO EVENT WILL LICENSOR BE LIABLE TO YOU ON ANY LEGAL THEORY FOR ANY SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR EXEMPLARY DAMAGES ARISING OUT OF THIS LICENSE OR THE USE OF THE WORK, EVEN IF LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

        7. Termination

        This License and the rights granted hereunder will terminate automatically upon any breach by You of the terms of this License. Individuals or entities who have received Adaptations or Collections from You under this License, however, will not have their licenses terminated provided such individuals or entities remain in full compliance with those licenses. Sections 1, 2, 5, 6, 7, and 8 will survive any termination of this License.
        Subject to the above terms and conditions, the license granted here is perpetual (for the duration of the applicable copyright in the Work). Notwithstanding the above, Licensor reserves the right to release the Work under different license terms or to stop distributing the Work at any time; provided, however that any such election will not serve to withdraw this License (or any other license that has been, or is required to be, granted under the terms of this License), and this License will continue in full force and effect unless terminated as stated above.
        8. Miscellaneous

        Each time You Distribute or Publicly Perform the Work or a Collection, the Licensor offers to the recipient a license to the Work on the same terms and conditions as the license granted to You under this License.
        Each time You Distribute or Publicly Perform an Adaptation, Licensor offers to the recipient a license to the original Work on the same terms and conditions as the license granted to You under this License.
        If any provision of this License is invalid or unenforceable under applicable law, it shall not affect the validity or enforceability of the remainder of the terms of this License, and without further action by the parties to this agreement, such provision shall be reformed to the minimum extent necessary to make such provision valid and enforceable.
        No term or provision of this License shall be deemed waived and no breach consented to unless such waiver or consent shall be in writing and signed by the party to be charged with such waiver or consent.
        This License constitutes the entire agreement between the parties with respect to the Work licensed here. There are no understandings, agreements or representations with respect to the Work not specified here. Licensor shall not be bound by any additional provisions that may appear in any communication from You. This License may not be modified without the mutual written agreement of the Licensor and You.
        The rights granted under, and the subject matter referenced, in this License were drafted utilizing the terminology of the Berne Convention for the Protection of Literary and Artistic Works (as amended on September 28, 1979), the Rome Convention of 1961, the WIPO Copyright Treaty of 1996, the WIPO Performances and Phonograms Treaty of 1996 and the Universal Copyright Convention (as revised on July 24, 1971). These rights and subject matter take effect in the relevant jurisdiction in which the License terms are sought to be enforced according to the corresponding provisions of the implementation of those treaty provisions in the applicable national law. If the standard suite of rights granted under applicable copyright law includes additional rights not granted under this License, such additional rights are deemed to be included in the License; this License is not intended to restrict the license of any rights under applicable law.




    """
    state = np.random.get_state()
    np.random.seed(seed)
    yield None
    np.random.set_state(state)


