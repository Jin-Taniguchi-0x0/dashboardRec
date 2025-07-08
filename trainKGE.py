import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from pykeen.triples import TriplesFactory
# ★★★ 比較対象のモデルをインポート ★★★
from pykeen.models import TransE, DistMult, ComplEx, RotatE
from pykeen.training import SLCWATrainingLoop
from pykeen.losses import MarginRankingLoss, NSSALoss
from pykeen.evaluation import RankBasedEvaluator
import time

# --- 設定 ---
TRIPLES_FILE = 'triple_viewsize_try.csv'
EMBEDDING_DIM = 200
NUM_EPOCHS = 500
LEARNING_RATE = 0.01
BATCH_SIZE = 256

# --- 1. データの読み込み ---
try:
    df = pd.read_csv(TRIPLES_FILE)
except FileNotFoundError:
    print(f"エラー: トリプルファイル '{TRIPLES_FILE}' が見つかりません。")
    exit()

df = df.astype(str)

# --- 2. データ分割 ---
relation_to_split = 'belongTo'
belong_to_df = df[df['predicate'] == relation_to_split].copy()
attribute_df = df[df['predicate'] != relation_to_split].copy()

train_triples, validation_triples, test_triples = [], [], []

for dashboard_name, group in belong_to_df.groupby('subject'):
    n_views = len(group)
    if n_views <= 3:
        train_triples.append(group)
    else:
        test_sample = group.sample(n=1, random_state=42)
        test_triples.append(test_sample)
        remaining_after_test = group.drop(test_sample.index)
        val_sample = remaining_after_test.sample(n=1, random_state=42)
        validation_triples.append(val_sample)
        train_sample = remaining_after_test.drop(val_sample.index)
        train_triples.append(train_sample)

if not train_triples:
    raise ValueError(f"学習用のトリプルリストが空です。")

train_df = pd.concat(train_triples, ignore_index=True)
validation_df = pd.concat(validation_triples, ignore_index=True) if validation_triples else pd.DataFrame(columns=belong_to_df.columns)
test_df = pd.concat(test_triples, ignore_index=True) if test_triples else pd.DataFrame(columns=belong_to_df.columns)
final_train_df = pd.concat([train_df, attribute_df], ignore_index=True)

print("データ分割の結果:")
print(f"学習用トリプル数: {len(final_train_df)}")
print(f"検証用トリプル数: {len(validation_df)}")
print(f"テスト用トリプル数: {len(test_df)}")

# --- 3. TriplesFactoryの作成 ---
training_factory = TriplesFactory.from_labeled_triples(triples=final_train_df.values)
validation_factory = TriplesFactory.from_labeled_triples(
    triples=validation_df.values, entity_to_id=training_factory.entity_to_id, relation_to_id=training_factory.relation_to_id
)
testing_factory = TriplesFactory.from_labeled_triples(
    triples=test_df.values, entity_to_id=training_factory.entity_to_id, relation_to_id=training_factory.relation_to_id
)

# --- 4. ★★★ 複数モデルの学習と評価ループ ★★★ ---

# 評価するモデルのリスト
# ComplExとDistMultはNSSALossという損失関数と相性が良いとされています
models_to_evaluate = [
    {'class': TransE, 'loss': MarginRankingLoss(margin=1.0)},
    {'class': DistMult, 'loss': NSSALoss(margin=1.0)},
    {'class': ComplEx, 'loss': NSSALoss(margin=1.0)},
    {'class': RotatE, 'loss': None}, # RotatEは内部で損失関数を定義
]

all_results = []
evaluator = RankBasedEvaluator()

for model_config in models_to_evaluate:
    model_class = model_config['class']
    loss_function = model_config['loss']
    model_name = model_class.__name__
    
    print("\n" + "="*50)
    print(f"--- モデル: {model_name} の学習と評価を開始 ---")
    print("="*50)
    
    # モデルの初期化
    model_kwargs = {'triples_factory': training_factory, 'embedding_dim': EMBEDDING_DIM, 'random_seed': 42}
    if loss_function:
        model_kwargs['loss'] = loss_function

    model = model_class(**model_kwargs)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training_loop = SLCWATrainingLoop(model=model, triples_factory=training_factory, optimizer=optimizer)

    # 学習
    start_time = time.time()
    _ = training_loop.train(triples_factory=training_factory, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, use_tqdm=True)
    end_time = time.time()
    print(f"{model_name} の学習が完了しました。(所要時間: {end_time - start_time:.2f}秒)")
    
    # 評価
    if not test_df.empty:
        test_results = evaluator.evaluate(model=model, mapped_triples=testing_factory.mapped_triples, batch_size=1024, use_tqdm=True)
        
        # 結果の抽出
        results_df = test_results.to_df()
        tail_realistic = results_df[(results_df['Side'] == 'tail') & (results_df['Rank_type'] == 'realistic')]
        
        mrr = tail_realistic[tail_realistic['Metric'] == 'inverse_harmonic_mean_rank']['Value'].iloc[0]
        hits_at_10 = tail_realistic[tail_realistic['Metric'] == 'hits_at_10']['Value'].iloc[0]
        hits_at_3 = tail_realistic[tail_realistic['Metric'] == 'hits_at_3']['Value'].iloc[0]
        
        all_results.append({
            'Model': model_name,
            'MRR': mrr,
            'Hits@10': hits_at_10,
            'Hits@3': hits_at_3,
        })

# --- 5. 最終結果の比較 ---
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values(by='MRR', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print("--- 全モデルの性能比較結果 ---")
    print("="*50)
    print(summary_df)
    summary_df.to_csv('results/model_comparison_results.csv', index=False)
    print("\n比較結果を 'results/model_comparison_results.csv' に保存しました。")
else:
    print("評価対象のテストデータがなかったため、比較結果はありません。")