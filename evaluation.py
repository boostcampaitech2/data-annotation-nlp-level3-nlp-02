import pickle as pickle
import os
import json
import pandas as pd
import argparse
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

def klue_re_micro_f1(preds, labels, average):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
      'no_relation',
      'product',
      'location',
      'poh:type',
      'related',
      'org:members',
      'poh:start_date',
      'per:member_of',
      'alternate',
      'org:field',
      'org:event',
      'per:title',
      'org:founded',
      'poh:end_date'
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    labels = np.eye(14)[labels]

    score = np.zeros((14,))
    for c in range(14):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def label_to_num(label):
  num_label = []

  dict_label_to_num = {'no_relation' : 0,
      'product' : 1,
      'location' : 2,
      'poh:type' : 3,
      'related' : 4,
      'org:members' : 5,
      'poh:start_date' : 6,
      'per:member_of' : 7,
      'alternate' : 8,
      'org:field' : 9,
      'org:event' : 10,
      'per:title' : 11,
      'org:founded' : 12,
      'poh:end_date' : 13
	  }
  for v in label:
    num_label.append(dict_label_to_num[v])
  return num_label

def make_probs(probs):
  prob_temp = []
  probs = probs.apply(lambda x: x[1:-1].split(','))
  for i in probs:
    prob_temp.append(list(map(float, i)))
  return prob_temp

def evaluation(gt_path, pred_path):
  pred = pd.read_csv(pred_path) # model이 예측한 정답 csv [id,pred_label, probs]
  answer = pd.read_csv(gt_path) # test dataset의 정답 csv [id, label]

  pre_pred = pred.loc[pred['id'][answer['id']]]

  micro_f1 = klue_re_micro_f1(label_to_num(pre_pred["pred_label"]), label_to_num(answer["label"]), average='micro') 
  auprc = klue_re_auprc(np.array(make_probs(pre_pred['probs'])), np.array(label_to_num(answer["label"])))

  results = {}
  results['micro_f1'] = {
        'value': f'{micro_f1:.04f}',
        'rank': True,
        'decs': True,
    }
  results['auprc'] = {
      'value': f'{auprc:.04f}',
      'rank': False,
      'decs': True,
  }

  return json.dumps(results)

def main(args):
  print(evaluation(args.public_dataset_dir, args.pred_answer_dir))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # data dir
  parser.add_argument('--pred_answer_dir', type=str, default="./submission.csv")
  parser.add_argument('--public_dataset_dir', type=str, default="./ground_truth/private_gt.csv")
  args = parser.parse_args()
  main(args)



