import torch
import random
import numpy as np
from A4.src.grader import setup, LARGE_EMBED_SIZE, LARGE_HIDDEN_SIZE, NONZERO_DROPOUT_RATE, test_encoding_hiddens, \
    BATCH_SIZE, Vocab, EMBED_SIZE, HIDDEN_SIZE, DROPOUT_RATE, reinitialize_layers
import submission


class Debug_1d():
    def setUp(self):
        # Set Seeds
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

            # Create Inputs
        input = setup()
        self.vocab = input[-1]

        # Initialize student model
        self.model = submission.NMT(
            embed_size=LARGE_EMBED_SIZE,
            hidden_size=LARGE_HIDDEN_SIZE,
            dropout_rate=NONZERO_DROPOUT_RATE,
            vocab=self.vocab
        )

        # Initialize soln model
        random.seed(35436)
        np.random.seed(4355)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        self.soln_model = self.run_with_solution_if_possible(submission, lambda sub_or_sol: sub_or_sol).NMT(
            embed_size=LARGE_EMBED_SIZE,
            hidden_size=LARGE_HIDDEN_SIZE,
            dropout_rate=NONZERO_DROPOUT_RATE,
            vocab=self.vocab
        )

        self.source_lengths = [len(s) for s in input[0]]
        self.source_padded = self.soln_model.vocab.src.to_input_tensor(input[0], device=self.soln_model.device)
        self.enc_hidden, self.decode_hidden, self.decode_cell = test_encoding_hiddens(self.source_padded,
                                                                                      self.source_lengths,
                                                                                      self.model, self.soln_model,
                                                                                      self.vocab)

    def test_0(self):
        """1d-0-basic:  Sanity check for Encode.  Compares student output to that of model with dummy data."""
        # Seed the Random Number Generators
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed * 13 // 7)

        # Load training data & vocabulary
        train_data_src = submission.read_corpus('./sanity_check_en_es_data/train_sanity_check.es', 'src')
        train_data_tgt = submission.read_corpus('./sanity_check_en_es_data/train_sanity_check.en', 'tgt')
        train_data = list(zip(train_data_src, train_data_tgt))

        for src_sents, tgt_sents in submission.batch_iter(train_data, batch_size=BATCH_SIZE, shuffle=True):
            src_sents = src_sents
            tgt_sents = tgt_sents
            break
        vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json')

        # Create NMT Model
        model = submission.NMT(
            embed_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            dropout_rate=DROPOUT_RATE,
            vocab=vocab)
        # Configure for Testing
        reinitialize_layers(model)
        source_lengths = [len(s) for s in src_sents]
        source_padded = model.vocab.src.to_input_tensor(src_sents, device=model.device)

        # Load Outputs
        enc_hiddens_target = torch.load('./sanity_check_en_es_data/enc_hiddens.pkl')
        dec_init_state_target = torch.load('./sanity_check_en_es_data/dec_init_state.pkl')

        # Test
        with torch.no_grad():
            enc_hiddens_pred, dec_init_state_pred = model.encode(source_padded, source_lengths)
        assert (np.allclose(enc_hiddens_target.numpy(),
                                    enc_hiddens_pred.numpy())), "enc_hiddens is incorrect: it should be:\n {} but is:\n{}".format(
            enc_hiddens_target, enc_hiddens_pred)
        print("enc_hiddens Sanity Checks Passed!")
        assert (np.allclose(dec_init_state_target[0].numpy(), dec_init_state_pred[
            0].numpy())), "dec_init_state[0] is incorrect: it should be:\n {} but is:\n{}".format(
            dec_init_state_target[0],
            dec_init_state_pred[0])
        print("dec_init_state[0] Sanity Checks Passed!")
        assert (np.allclose(dec_init_state_target[1].numpy(), dec_init_state_pred[
            1].numpy())), "dec_init_state[1] is incorrect: it should be:\n {} but is:\n{}".format(
            dec_init_state_target[1],
            dec_init_state_pred[1])
        print("dec_init_state[1] Sanity Checks Passed!")


if __name__ == '__main__':
    d = Debug_1d()
    a = d.test_0()
    pass