import transformers
import torch
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig,T5Tokenizer, T5ForConditionalGeneration,GPT2Tokenizer, GPT2Model
import scipy.stats as ss
import json
import random
import torch.nn.functional as F
import argparse
import numpy as np
import math
from scipy.stats.stats import kendalltau,spearmanr,pearsonr
from scipy.spatial import distance
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="")
parser.add_argument("--save", type=str, default="")
parser.add_argument("--input", type=str, default="")


args = parser.parse_args()


def mand(P,Q):
    P = np.array(P)
    Q = np.array(Q)
    return(np.sum(np.abs(P-Q))/len(P))



generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=5,
    num_return_sequences=1,
    output_scores=True,
    output_hidden_states=False,
    output_attentions=False,
    return_dict_in_generate=True
)

checkpoint = args.model_dir


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")



with open(args.input, 'r', encoding='utf8') as f:
    lines = f.readlines()
prompts = json.loads(lines[0])['relations']

out_f = open(args.save, 'w', encoding='utf8')
punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']
ACC = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
ACC_probs = [[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0]]

ACC = np.array(ACC)
ACC_probs = np.array(ACC_probs)

PID_mand, IRD_mand, MONIITOR_mand= 0,0,0



count,count_threshold,count_scale = 0,0,0
flag_threshold, flag_scale = False, False
for line in lines[1:]:
    line = json.loads(line)
    X = line['subject'].strip()
    M = line['object'].strip()
    out_f.flush()
    flag_threshold, flag_scale = False, False
    res_pos,res_neg = [],[]
    taxonomy = [M] + line['taxonomy']

    mand_avg_positive, mand_avg_negative = 0, 0

    mand_negative,  mand_positive = [], []

    po,neg, ori = 0,0,0
    M_random = taxonomy[0]
    prompt = '[M_disturb]. ' + prompts[0]

    prompt = prompt.replace("[X]", X)
    prompt = prompt.replace("[M_disturb]", M_random)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, generation_config=generation_config, max_new_tokens=15)
    output = tokenizer.decode(outputs[0][0], skip_special_tokens=True).strip()
    if output.startswith(prompt):
        ans = output.split(prompt)[-1].strip()
    else:
        ans = output
    for p in punc:
        ans = ans.replace(p, "")

    if M == ans:
        count += 1
        po += 1

        logit = 0
        scale = 1
        length = len(outputs.scores)
        probs1 = outputs.scores[-1][0].softmax(-1)
        probs2 = outputs.scores[-2][0].softmax(-1)

        if tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == '</s>':
            length -= 1
        elif tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == ' </s>':
            length -= 1
        elif tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == '.':
            length -= 1
        if tokenizer.decode(torch.argsort(probs2, dim=-1, descending=True)[0]) == '.':
            length -= 1
        logit_anchor = []
        pos_anchor = []
        for k in range(length):
            if k != 0 and tokenizer.decode(torch.argsort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[0]) == '\n':
                break
            if k != 0 and tokenizer.decode(torch.argsort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[0]) == '</s>':
                    break
            pos = torch.sort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[1][0] #top-1 position
            pos_anchor.append(pos)
            logit_anchor.append(outputs.scores[k][0].softmax(-1)[pos].cpu().numpy())
        res_neg.append(logit_anchor)
        ACC_probs[0][1] += np.sum(logit_anchor)/len(logit_anchor)
        ACC[0][1] += 1



        for i in range(len(prompts)):
            pro = prompts[i]
            prompt = pro #计算没有干扰信息的情况
            prompt = prompt.replace("[X]", X)
            inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, generation_config=generation_config, max_new_tokens=15)
            output = tokenizer.decode(outputs[0][0], skip_special_tokens=True).strip()
            if output.startswith(prompt):
                ans = output.split(prompt)[-1].strip()
            else:
                ans = output
            for p in punc:
                ans = ans.replace(p, "")

            length = len(outputs.scores)
            probs1 = outputs.scores[-1][0].softmax(-1)
            probs2 = outputs.scores[-2][0].softmax(-1)

            if tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == '</s>':
                length -= 1
            elif tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == ' </s>':
                length -= 1
            elif tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == '.':
                length -= 1
            if tokenizer.decode(torch.argsort(probs2, dim=-1, descending=True)[0]) == '.':
                length -= 1
            len_pos = 0
            logit_pos = []
            for k in range(length):
                if k != 0 and tokenizer.decode(torch.argsort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[0]) == '\n':
                    break
                if k != 0 and tokenizer.decode(torch.argsort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[0]) == '</s>':
                    break
                if len_pos < len(pos_anchor):
                    pos = pos_anchor[len_pos]
                    logit_pos.append(outputs.scores[k][0].softmax(-1)[pos].cpu().numpy())
                else:
                    break
                len_pos += 1

            res_pos.append(logit_pos)


            if M == ans:
                original_acc += 1
                ori += 1
                ACC_probs[i][0] += np.sum(logit_pos)/len(logit_pos)
                ACC[i][0] += 1
            if i == 0:
                for j in range(1,len(taxonomy)):
                    M_random = taxonomy[j]
                    prompt = '[M_disturb]. ' + prompts[0]

                    prompt = prompt.replace("[X]", X)
                    prompt = prompt.replace("[M_disturb]", M_random)
                    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                    outputs = model.generate(inputs, generation_config=generation_config, max_new_tokens=15)
                    output = tokenizer.decode(outputs[0][0], skip_special_tokens=True).strip()
                    if output.startswith(prompt):
                        ans = output.split(prompt)[-1].strip()
                    else:
                        ans = output
                    for p in punc:
                        ans = ans.replace(p, "")

                    length = len(outputs.scores)
                    probs1 = outputs.scores[-1][0].softmax(-1)
                    probs2 = outputs.scores[-2][0].softmax(-1)

                    if tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == '</s>':
                        length -= 1
                    elif tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == ' </s>':
                        length -= 1
                    elif tokenizer.decode(torch.argsort(probs1, dim=-1, descending=True)[0]) == '.':
                        length -= 1
                    if tokenizer.decode(torch.argsort(probs2, dim=-1, descending=True)[0]) == '.':
                        length -= 1

                    len_neg = 0
                    logit_neg = []
                    for k in range(length):
                        if k != 0 and tokenizer.decode(torch.argsort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[0]) == '\n':
                            break
                        if k != 0 and tokenizer.decode(torch.argsort(outputs.scores[k][0].softmax(-1), dim=-1, descending=True)[0]) == '</s>':
                            break
                        if len_neg < len(pos_anchor):
                            pos = pos_anchor[len_neg]
                            logit_neg.append(outputs.scores[k][0].softmax(-1)[pos].cpu().numpy())
                        else:
                            break
                        len_neg += 1
                    res_neg.append(logit_neg)


                    if M == ans:
                        if M_random == M:
                            po += 1
                            ACC[i][1] += 1
                            ACC_probs[i][1] += np.sum(logit_neg)/len(logit_neg)
                        else:
                            neg += 1
                            ACC_probs[i][1+j] += np.sum(logit_neg)/len(logit_neg)
                            ACC[i][1+j] += 1



                for k in range(1, len(taxonomy)):
                    if len(res_neg[0]) > len(res_neg[k]):
                        pad_length = len(res_neg[0]) - len(res_neg[k])
                        pad = np.zeros(pad_length)
                        foreign_negative = np.append(np.array(res_neg[k]), pad)
                    else:
                        foreign_negative = res_neg[k]

                    mind_negative.append(mind(res_neg[0], foreign_negative))
                mand_avg_negative = sum(mand_negative) / 5

        for m in range(len(prompts)):
            if len(res_neg[0]) > len(res_pos[m]):
                pad_length = len(res_neg[0]) - len(res_pos[m])
                pad = np.zeros(pad_length)
                foreign_positive = np.append(np.array(res_pos[m]), pad)
            else:
                foreign_positive = res_pos[m]

            mand_positive.append(mand(res_neg[0], foreign_positive))
        mand_avg_positive = sum(mand_positive) / 7

        mand_avg_square = math.sqrt((mand_avg_negative *mand_avg_negative + mand_avg_positive* mand_avg_positive+mand_avg_negative * mand_avg_positive)/3)

        out_f.write("%s\t" % str(ori))
        out_f.write("%s\t" % str(po))
        out_f.write("%s\t" % str(neg))
        out_f.write("mand\t")
        out_f.write("%s\t" % str(mand_avg_positive))
        out_f.write("%s\n" % str(mand_avg_negative))


        out_f.flush()
        PID_mand += mand_avg_positive
        IRD_mand += mand_avg_negative

        MONIITOR_mand += mand_avg_square


if count != 0:
    print("ACC score", ACC / count)
    print("ACC probs score", ACC_probs/count)


    out_f.write("acc range %s\n" % str(ACC / count))
    out_f.write("acc probs range %s\n" % str(ACC_probs / count))

    out_f.write("avg mand CBD_avg square %s\n" % str(MONIITOR_mand/ ACC_probs[0][1]))

out_f.close()
