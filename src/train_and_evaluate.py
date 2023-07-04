from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import numpy as np

USE_CUDA = torch.cuda.is_available()

class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            if (len(num_stack) == 0):
                target[i] = num_start
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str



def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_prefix_tree_results(test_res, test_tar, output_lang, num_list, num_stack):
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack): ##
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, exp_encoder, predict, generate, merge, output_lang, num_pos, exp_num_pos, tarexp_num_lengths, exp_num_index, numslist_batch, adj_tar_batch, alpha, beam_size, epsilon, temp, hard, english=False):

    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]


    input_var = torch.LongTensor(input_batch)
    target = torch.LongTensor(target_batch).transpose(0, 1)
    adj_tar_graph = torch.LongTensor(adj_tar_batch)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    exp_encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        adj_tar_graph = adj_tar_graph.cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(input_var, 1 - seq_mask.to(torch.long))

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    target_exp_ids = copy.deepcopy(target)

    true_exp_encoder_outputs = exp_encoder(target_exp_ids.cuda(), adj_tar_graph.float())
    true_exp_encoder_outputs = true_exp_encoder_outputs.view(batch_size, -1, true_exp_encoder_outputs.size(2))

    exp_num_hidden = []
    for i in range(batch_size):
        exp_num_hidden.append(true_exp_encoder_outputs[i][exp_num_pos[i]])
    exp_num_hiddens = torch.stack(exp_num_hidden)

    exp_num_index = torch.Tensor(exp_num_index)
    question_num_blank_hidden = all_nums_encoder_outputs
    new_batch_numlen = []
    new_batch_num_hidden = []
    question_num_id = []
    question_num_len = []
    for i in range(batch_size):
        expression_num_lenth = tarexp_num_lengths[i]
        num_label = exp_num_index[i][:expression_num_lenth]
        if expression_num_lenth == 0:
            new_batch_num_hidden.append(torch.zeros(1, exp_num_hiddens.size(2)))
            new_batch_numlen.append(0)
            question_num_id.append(0)
            question_num_len.append(0)
            continue
        idxs = getIdx(num_label)
        new_num_len = len(idxs)
        new_batch_numlen.append(new_num_len)

        index_flag = torch.unique(num_label)
        question_num_id.append(index_flag)
        question_num_len.append(index_flag.size(0))

        per_batch_feature = []
        for j, index in enumerate(idxs):
            numberfeatures = exp_num_hiddens[i][index]
            per_batch_feature.append(torch.mean(numberfeatures, dim=0))  #

        per_batch_features = torch.stack(per_batch_feature)
        new_batch_num_hidden.append(per_batch_features)

    expnum_hiddens = torch.zeros(exp_num_hiddens.size(0), max(new_batch_numlen),
                                 exp_num_hiddens.size(2)).cuda()
    batch_lables = torch.zeros(exp_num_hiddens.size(0), max(new_batch_numlen)).cuda()
    question_num_ids = torch.zeros(batch_size, max(question_num_len))

    for i in range(exp_num_hiddens.size(0)):
        expnum_hiddens[i][:new_batch_numlen[i]] = new_batch_num_hidden[i]
        batch_lables[i][:new_batch_numlen[i]] = torch.arange(0, new_batch_numlen[i])
        question_num_ids[i][:question_num_len[i]] = question_num_id[i]

    ques_num_hiddens = []
    for i in range(batch_size):
        ques_num_hiddens.append(question_num_blank_hidden[i][question_num_ids[i].type(torch.long)])
    ques_num_hiddens = torch.stack(ques_num_hiddens)  # b n_q h
    query = encoder.question_linear(ques_num_hiddens).unsqueeze(2)  # b n_q 1 h


    pred_ids = []
    pred_lengths = []
    pred_num_pos = []
    pred_num_lengths = []
    pred_num_index = []
    pred_exp_adjs = []
    for i in range(batch_size):
        inputlen = input_length[i]
        inputid = input_batch[i][:inputlen]
        targetlen = target_length[i]
        targetid = target_batch[i][:targetlen]
        numpos = num_pos[i]
        numlist = numslist_batch[i]
        test_res, _, q_list, _ = evaluate_tree_duringtrain(inputid, inputlen, generate_nums, encoder, predict, generate,
                             merge, output_lang, numpos, temp, hard, beam_size=1, analyze=True)
        _, _, pred_exp, _ = compute_prefix_tree_results(test_res, targetid, output_lang,
                                                      numlist, nums_stack_batch[i])
        index = 0
        for j in range(len(numlist)):###
            number = numlist[j]

            for k in range(len(pred_exp)):
                if pred_exp[k] == number:
                    pred_exp[k] = "N" + str(index)
                if type(pred_exp[k]) == int:
                    pred_exp[k] = str(pred_exp[k])
            index = index + 1
        pred_exp = [str(x) for x in pred_exp]
        predexp = copy.deepcopy(pred_exp)
        try:
            exp_tree, index_tree = PreExpTree(predexp)
        except:
            predexp = ["UNK"]
            pred_exp = ["UNK"]
            exp_tree, index_tree = PreExpTree(predexp)

        pre_order, adj_index = preorder_adj(index_tree)
        diag_ele = np.zeros(len(adj_index))
        for i in range(len(adj_index)):
            diag_ele[i] = 1
        adj_matrix = np.diag(diag_ele)
        for i in range(len(adj_index)):
            adj_node_list = adj_index[i]
            if adj_node_list == []:
                continue
            for j in range(len(adj_node_list)):
                adj_matrix[i][adj_node_list[j]] = 1

        pred_id = indexes_from_sentence(output_lang, pred_exp, tree=True)
        posi = 0
        expnum_pos = []
        expnum_index = []
        for id in pred_id:
            token = output_lang.index2word[id]
            if token[0] == 'N':
                expnum_pos.append(posi)
                expnum_index.append(int(token[-1]))
            posi = posi + 1

        pred_ids.append(pred_id)
        pred_lengths.append(len(pred_id))
        pred_num_pos.append(expnum_pos)
        pred_num_lengths.append(len(expnum_pos))
        pred_num_index.append(expnum_index)
        pred_exp_adjs.append(adj_matrix)

    pred_len_max = max(pred_lengths)
    expnum_len_max = max(pred_num_lengths)

    pred_out_batch = []
    pred_num_pos_batch = []
    pred_num_index_batch = []
    pred_adj_graph_batch = []
    for i in range(batch_size):
        pred_out_batch.append(pad_seq(pred_ids[i], pred_lengths[i], pred_len_max))
        pred_num_pos_batch.append(pred_num_pos[i] + [0] * (expnum_len_max - pred_num_lengths[i])) #
        pred_num_index_batch.append(pred_num_index[i] + [0] * (expnum_len_max - pred_num_lengths[i]))
        pred_adj_graph_batch.append(pad_adj(pred_exp_adjs[i], pred_len_max))

    adj_pred_graph = torch.LongTensor(np.array(pred_adj_graph_batch))

    pred_num_index_batch = torch.Tensor(pred_num_index_batch)


    prediction = torch.LongTensor(pred_out_batch).transpose(0, 1)
    pred_exp_ids = copy.deepcopy(prediction)

    pred_exp_encoder_outputs = exp_encoder(pred_exp_ids.cuda(), adj_pred_graph.float().cuda())
    pred_exp_encoder_outputs = pred_exp_encoder_outputs.view(batch_size, -1, pred_exp_encoder_outputs.size(2))

    pred_num_hidden = []
    for i in range(batch_size):
        pred_num_hidden.append(pred_exp_encoder_outputs[i][pred_num_pos_batch[i]])
    pred_num_hiddens = torch.stack(pred_num_hidden)


    new_pred_num_hidden = []
    for i in range(batch_size):
        equation_num_lenth = pred_num_lengths[i]
        target_num_lenth = tarexp_num_lengths[i]

        if target_num_lenth == 0:
            new_pred_num_hidden.append(torch.zeros(1, pred_num_hiddens.size(2)))
            continue

        if equation_num_lenth == 0:
            new_pred_num_hidden.append(torch.zeros(new_batch_numlen[i], pred_num_hiddens.size(2)))
            continue

        else:
            perbatch_feature = []
            pred_num_label = pred_num_index_batch[i][:equation_num_lenth]
            target_num_label = exp_num_index[i][:target_num_lenth]
            pred_idxs = getIdx(pred_num_label)
            pred_index_flag = torch.unique(pred_num_label)
            pred_index_flag_np = pred_index_flag.cpu().numpy()
            tar_index_flag = torch.unique(target_num_label)
            for j in range(tar_index_flag.size(0)):
                mark = tar_index_flag[j]

                if mark in pred_index_flag:
                    index_pre = np.argwhere(pred_index_flag_np == mark.cpu().numpy())
                    index_pre = index_pre[0][0]
                    index = pred_idxs[index_pre]
                    prednum_features = pred_num_hiddens[i][index]
                    perbatch_feature.append(torch.mean(prednum_features, dim=0))
                else:
                    perbatch_feature.append(torch.zeros(pred_num_hiddens.size(2)).cuda())

            perbatch_features = torch.stack(perbatch_feature) ###

        new_pred_num_hidden.append(perbatch_features)

    prednum_hiddens = torch.zeros(pred_num_hiddens.size(0), max(new_batch_numlen),
                                  pred_num_hiddens.size(2)).cuda()

    for i in range(batch_size):
        prednum_hiddens[i][:new_batch_numlen[i]] = new_pred_num_hidden[i]


    equa_num_hiddens = epsilon * expnum_hiddens + (1 - epsilon) * prednum_hiddens  # b n_e h

    key = encoder.equation_linear(equa_num_hiddens).unsqueeze(1)  # b 1 n_e h
    e = torch.tanh(query + key)  # b n_q n_e h
    # b n_q n_e
    e = encoder.elinear(e).squeeze(-1)

    pointed_mask = e.new_zeros(e.size(0), e.size(1), e.size(2)).bool()
    pointed_mask_by_target = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(2))
    target_mask = pointed_mask.new_zeros(pointed_mask.size(0), pointed_mask.size(1))
    for b in range(pointed_mask_by_target.size(0)):
        pointed_mask_by_target[b, :new_batch_numlen[b]] = 1
        target_mask[b, :question_num_len[b]] = 1
    pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)
    e.masked_fill_(pointed_mask_by_target == 0, -1e9)

    logp = F.log_softmax(e, dim=-1)
    logp = logp.view(-1, logp.size(-1))
    critic = nn.NLLLoss(reduction='none')
    loss = critic(logp, batch_lables.type(torch.long).contiguous().view(-1))
    target_mask = target_mask.view(-1)
    loss.masked_fill_(target_mask == 0, 0)

    transform_loss = loss.reshape(e.size(0), e.size(1))
    for i, item in enumerate(question_num_len):
        if item == 0:
            question_num_len[i] = 1
    transform_loss_sample = torch.sum(transform_loss, -1) / (
        torch.tensor(question_num_len).float().cuda())
    loss_infill = transform_loss_sample.sum() / batch_size

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    loss_math = masked_cross_entropy(all_node_outputs, target, target_length)
    loss = loss_math + alpha*loss_infill

    loss.backward()

    return loss.item(), loss_math.item(), loss_infill.item()


def getIdx(a):
    co = a.unsqueeze(0)-a.unsqueeze(1)
    uniquer = co.unique(dim=0)
    #print(uniquer)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0))
        mask = r==0
        idx = cover[mask]
        out.append(idx)
    outs = out[::-1]
    return outs

def train_tree_core(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, output_lang, num_pos, english=False, no_expr_loss=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    encoder_outputs, problem_output = encoder(input_var, 1 - seq_mask.to(torch.long))
    if (no_expr_loss):
        return None, None, None, problem_output
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    q_list = []
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
        q_list.append(current_embeddings)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    return all_node_outputs, target, target_length, problem_output, q_list

def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH, analyze=False):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    with torch.no_grad():
        encoder.eval()
        predict.eval()
        generate.eval()
        merge.eval()
        padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

        batch_size = 1

        if USE_CUDA:
            input_var = input_var.cuda()
            seq_mask = seq_mask.cuda()
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()
        # Run words through encoder
        if analyze:
            encoder_outputs, problem_output, hidden = encoder(input_var, analyze=analyze)
        else:
            encoder_outputs, problem_output = encoder(input_var, analyze=analyze)

        # Prepare input and output variables
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

        num_size = len(num_pos)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                                encoder.hidden_size)
        num_start = output_lang.num_start
        # B x P x N
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        q_list = []
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                    b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                    seq_mask, num_mask)
                q_list.append(current_embeddings)
                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if USE_CUDA:
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
    if analyze:
        return beams[0].out, problem_output, q_list, hidden
    return beams[0].out


def evaluate_tree_duringtrain(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang,
                              num_pos, temp, hard,
                              beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH, analyze=False):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(0)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder
    if analyze:
        encoder_outputs, problem_output, hidden = encoder(input_var, analyze=analyze)
    else:
        encoder_outputs, problem_output = encoder(input_var, analyze=analyze)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    q_list = []

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)
            q_list.append(current_embeddings)
            out_score = F.gumbel_softmax(torch.cat((op, num_score), dim=1), tau=temp, hard=hard)
            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))

        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        # print(beams)
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    if analyze:
        return beams[0].out, problem_output, q_list, hidden
    return beams[0].out