import torch
import torch.nn.functional as F


class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), 0, dtype=torch.long, device=device)]
        self.next_ys[0][0] = 2

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        # print("print:{}....{}".format(best_scores_id,num_words))
        # print("prev_k:{}".format(best_scores_id/ num_words))
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == 3:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            # print("self.prev_ks:{}\n".format(self.prev_ks))
            # print("keys :{} type:{}\n".format(keys, type(keys)))
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[2] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            # print("j :{} k:{}\n".format(j,k))
            # print("k_pre_ks :{} type:{} \n".format(self.prev_ks[j][k],type(self.prev_ks[j][k])))
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


class Translator(object):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.device = torch.device(cfg.test.device)
        self.model = model
        self.keep_aspect_ratio = cfg.keep_aspect_ratio
        self.stages = {'TPS': cfg.tps_block, 'Feat': cfg.feature_block}

    def translate_batch(self, images):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.contiguous().view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            # active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc.permute(1, 0, 2), active_inst_idx, n_prev_active_inst, n_bm).permute(1, 0, 2)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm, memory_key_padding_mask):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                '''
                prepare beam_tgt for decoder
                :param inst_dec_beams: class beam(num:batch_size)
                :param len_dec_seq: max_len(beam search len)
                :return: beam_tgt
                '''
                # init beams as 2 for start
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_memory_key_padding_mask(inst_dec_beams, memory_key_padding_mask, n_bm):
                keep = []
                for idx, each in enumerate(memory_key_padding_mask):
                    if not inst_dec_beams[idx].done:
                        keep.append(idx)
                memory_key_padding_mask = memory_key_padding_mask[torch.tensor(keep)]
                len_s = memory_key_padding_mask.shape[-1]
                n_inst = memory_key_padding_mask.shape[0]
                memory_key_padding_mask = memory_key_padding_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
                return memory_key_padding_mask


            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, enc_output, n_active_inst, n_bm, memory_key_padding_mask):
# ------ decoder predict word-------
                sem_seq,sem_mask,sem_key_padding_mask = self.model.semantic_branch(dec_seq)
                pos_seq = self.model.positional_branch(sem_seq)
                dec_output = self.model.mdcdp(sem_seq,enc_output,pos_seq,
                                tgt_mask=sem_mask,
                                tgt_key_padding_mask=sem_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                ).permute(1, 0, 2)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

# --- beam decoder step start ---
        #     dec_seq : decoder word num for one tgt
            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            if self.keep_aspect_ratio:
                memory_key_padding_mask = prepare_beam_memory_key_padding_mask(inst_dec_beams, memory_key_padding_mask, n_bm)
            else:
                memory_key_padding_mask = None
            # dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, enc_output, n_active_inst, n_bm, memory_key_padding_mask)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]
                # print("best :{} type:{}\n".format(tail_idxs[:n_best]))
                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

# ---start predict---
        with torch.no_grad():
            #-- Encode
            images = images.to(self.device)
            src_enc,memory_key_padding_mask  = self.model.visual_branch(images)
            # print(src_enc.shape)
            # -------- note delete ------
            #-- Repeat data for beam search

            n_bm = self.cfg.beam_size

            src_enc = src_enc.permute(1, 0, 2)
            n_inst, len_s, d_h = src_enc.size()
            # src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h).permute(1, 0, 2)
            # memory_key_padding_mask = memory_key_padding_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            #-- Prepare beams
            # n_inst == batch_size
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, 50):
                # word iter for sentense

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, n_bm, memory_key_padding_mask)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_enc, inst_idx_to_position_map = collate_active_info(
                    src_enc, inst_idx_to_position_map, active_inst_idx_list)
        # each decoder word transform to vocab
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.cfg.n_best)

        return batch_hyp, batch_scores

