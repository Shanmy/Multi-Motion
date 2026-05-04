# TODO: update this code with recent changes?

import os
from datetime import datetime
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_laion import load_img_tar
from viz_utils import viz_image

import sys
sys.path.append('./open_clip/src/')
from open_clip.factory import create_model_and_transforms


def get_embeddings(args, df):
    # load open clip
    clip_model, _, clip_transforms = create_model_and_transforms(
        args.open_clip_type,
        args.open_clip_path,
        precision='amp',
        device=torch.device(0),
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )

    clip_embeddings = []

    # get embeddings (and maybe viz)
    for i in range(len(df)):
        print(f'Processing row {i}')
        df_row = df.iloc[i]
        image = load_img_tar(f"{df_row['ids']:09d}", args.laion_path)

        image_pil = Image.fromarray(image)
        image_clip = clip_transforms(image_pil)
        image_clip = image_clip.unsqueeze(0).cuda()

        with torch.no_grad():
            out_dict = clip_model(image_clip, None)
        clip_embeddings.append(out_dict['image_features'].cpu())

        if args.viz_images:
            for col_name in df_row.drop('ids').keys():
                if df_row[col_name] == 1:
                    viz_image(image, "", os.path.join(args.out_path, col_name), id=df_row['ids'])

    clip_embeddings = np.concatenate(clip_embeddings, 0)
    if args.save_clip_embeddings:
        np.save(os.path.join(args.out_path, 'clip_embeddings.npy'), clip_embeddings)

    return clip_embeddings


def train_linear_classifier(X_train, y_train, X_test, y_test):
    C_vals = np.exp(np.linspace(np.log(0.01), np.log(100), 100))
    best_acc = 0
    best_C = 0
    best_clf = None
    for C_val in C_vals:
        clf = LogisticRegression(C=C_val, max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        #print(f'Logistic Regression Accuracy: {acc:.04f}. C: {C_val}')
        if acc >= best_acc:
            best_acc = acc
            best_C = C_val
            best_clf = clf
    print(f'Best Logistic Regression Accuracy: {best_acc}. Best C: {best_C}')
    return best_clf


def train_svm(X_train, y_train, X_test, y_test):
    C_vals = np.exp(np.linspace(np.log(0.0001), np.log(1), 100))
    best_acc_svm = 0
    best_C_svm = 0
    for C_val in C_vals:
        clf_svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=C_val))
        clf_svm.fit(X_train, y_train)
        y_pred_svm = clf_svm.predict(X_test)
        acc = (y_pred_svm == y_test).mean()
        #print(f'SVM Accuracy: {acc:.04f}. C: {C_val}')
        if acc >= best_acc_svm:
            best_acc_svm = acc
            best_C_svm = C_val
    print(f'Best SVM Accuracy: {best_acc_svm}. Best C: {best_C_svm}')


class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        h = self.lin2(self.relu(self.lin1(x)))
        return self.relu(x + h)

class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers=4):
        super().__init__()

        mlp_layers = []
        for _ in range(num_layers):
            mlp_layers.append(ResBlock(hidden_size))
        self.mlp_layers = nn.Sequential(*mlp_layers)
        self.mlp_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.mlp_out(self.mlp_layers(x))


def train_mlp(args, X_train, y_train, X_test, y_test):

    mlp = MLP(hidden_size=X_train.shape[1])
    mlp = mlp.cuda()
    optim = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0
    for j in range(args.train_steps):
        #print(f'Training batch {j + 1} of {args.train_steps}')

        batch_inds = np.random.permutation(X_train.shape[0])[0:args.batch_size]
        X_batch = torch.tensor(X_train[batch_inds]).cuda()
        y_batch = torch.tensor(y_train[batch_inds]).cuda().float()

        batch_out = mlp(X_batch).squeeze(1)
        loss = loss_fn(batch_out, y_batch)

        optim.zero_grad()
        loss.backward()
        optim.step()

        test_out = mlp(torch.tensor(X_test).cuda()).squeeze(1)
        acc = ((test_out > 0).long().cpu().numpy() == y_test).mean()
        #print(f"MLP Acc: {acc}")
        if acc > best_acc:
            best_acc = acc

    print(f"Best MLP Acc: {best_acc}")


df = pd.read_csv('/home/us000240/pose_caption_dataset/filtering-23-10-13-20-03-53.csv')

parser = argparse.ArgumentParser()
parser.add_argument("--annotation_file", type=str, default='/home/us000240/pose_caption_dataset/filtering-23-10-13-20-03-53.csv')
parser.add_argument("--laion_path", type=str, default="/nfs/USRCSEA/IVA/Datasets/Text2Image/laion400m/data4/")
parser.add_argument("--open_clip_path", type=str, default='/nfs/USRCSEA/IVA/Models/PoseCaptionData/open_clip/vit_h_14_open_clip_pytorch_model.bin')
parser.add_argument("--open_clip_type", type=str, default='ViT-H-14')

# threshold for binary classifier
parser.add_argument("--thresh", type=float, default=0.75)

# settings for training mlp classifier using open clip features
parser.add_argument("--train_steps", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--clip_embedding_path", type=str, default="/home/us000240/pose_caption_dataset/out_laion_ref_viz/23-10-18-01-14-17/clip_embeddings.npy")
parser.add_argument("--load_clip_embeddings", type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument("--save_clip_embeddings", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--viz_images", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--out_path", type=str, default="out_laion_ref_viz/")
args = parser.parse_args()

# folder to save viz results
args.out_path = os.path.join(args.out_path, datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
os.makedirs(args.out_path)

df = pd.read_csv(args.annotation_file)

# make folders for each column
if args.viz_images:
    for col_name in df.drop('ids', axis=1).keys():
        os.makedirs(os.path.join(args.out_path, col_name))

# check that data is valid
df_np = df.to_numpy()
# only 0 or 1 values
assert df_np[:, 1:].size == (df_np[:, 1:] == 1).sum() + (df_np[:, 1:] == 0).sum()
# each positive label is has no negative labels
assert df_np[df_np[:, 1] == 1, 2:].nonzero()[0].size == 0
# each negative label has at least one negative sublabel
assert (df_np[df_np[:, 1] == 0, 2:].sum(axis=1) == 0).nonzero()[0].size == 0


if args.load_clip_embeddings:
    clip_embeddings = np.load(args.clip_embedding_path)
else:
    clip_embeddings = get_embeddings(args, df)

X_train, X_test, y_train, y_test, keys_train, keys_test = \
    train_test_split(clip_embeddings, df_np[:, 1], df_np[:, 0], test_size=0.25)

#train_svm(X_train, y_train, X_test, y_test)
#train_mlp(args, X_train, y_train, X_test, y_test)
clf = train_linear_classifier(X_train, y_train, X_test, y_test)
y_pred_probs = clf.predict_proba(X_test)[:, 1]

thresh_list = np.linspace(0.5, 0.95, 100)
for thresh in thresh_list:
    y_pred_thresh = (y_pred_probs > thresh).astype(np.int64)
    print(
            f'Thresh {thresh:.03f}. ' +
            f'Precision: {precision_score(y_test, y_pred_thresh):.04f}. '
            f'Recall: {recall_score(y_test, y_pred_thresh):.04f}.'
        )

# four outcomes
os.makedirs(os.path.join(args.out_path, 'true_positive'))
os.makedirs(os.path.join(args.out_path, 'true_negative'))
os.makedirs(os.path.join(args.out_path, 'false_positive'))
os.makedirs(os.path.join(args.out_path, 'false_negative'))

for i, key in enumerate(keys_test):
    print(f'Processing row {i}')
    image = load_img_tar(f"{key:09d}", args.laion_path)
    if y_test[i] == 1:
        if y_pred_probs[i] >= args.thresh:
            out_folder = 'true_positive'
        else:
            out_folder = 'false_negative'
    else:
        if y_pred_probs[i] >= args.thresh:
            out_folder = 'false_positive'
        else:
            out_folder = 'true_negative'
    viz_image(image, "", os.path.join(args.out_path, out_folder), id=key)
