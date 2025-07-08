import pandas as pd
import torch
from pykeen.triples import TriplesFactory
from pykeen.models import RotatE
from pykeen.hpo import hpo_pipeline
import os

# --- 設定 ---
TRIPLES_FILE = 'triple_viewsize_try.csv'

# --- 1. データの読み込み ---
try:
    df = pd.read_csv(TRIPLES_FILE)
except FileNotFoundError:
    print(f"エラー: トリプルファイル '{TRIPLES_FILE}' が見つかりません。")
    exit()

df = df.astype(str)

# --- 2. データ分割（前回と同じロジック） ---
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
    raise ValueError("学習用トリプルがありません。")
train_df = pd.concat(train_triples, ignore_index=True)
validation_df = pd.concat(validation_triples, ignore_index=True) if validation_triples else pd.DataFrame(columns=belong_to_df.columns)
test_df = pd.concat(test_triples, ignore_index=True) if test_triples else pd.DataFrame(columns=belong_to_df.columns)
final_train_df = pd.concat([train_df, attribute_df], ignore_index=True)

# --- 3. TriplesFactoryの作成 ---
training_factory = TriplesFactory.from_labeled_triples(triples=final_train_df.values)
validation_factory = TriplesFactory.from_labeled_triples(
    triples=validation_df.values, entity_to_id=training_factory.entity_to_id, relation_to_id=training_factory.relation_to_id
)
testing_factory = TriplesFactory.from_labeled_triples(
    triples=test_df.values, entity_to_id=training_factory.entity_to_id, relation_to_id=training_factory.relation_to_id
)

# --- 4. HPOパイプラインの実行 ---
output_directory = 'hpo_results'
os.makedirs(output_directory, exist_ok=True)

print("\n" + "="*50)
print("--- ハイパーパラメータ最適化(HPO)を開始（早期終了あり）---")
print("="*50)

hpo_result = hpo_pipeline(
    n_trials=30,
    training=training_factory,
    validation=validation_factory,
    testing=testing_factory,
    model=RotatE,
    model_kwargs_ranges=dict(
        embedding_dim=dict(type='int', low=64, high=256, step=32),
    ),
    optimizer='Adam',
    optimizer_kwargs_ranges=dict(
        lr=dict(type='float', low=1e-4, high=1e-2, log=True),
    ),
    training_loop='SLCWATrainingLoop',
    training_kwargs_ranges=dict(
        num_epochs=dict(type='int', low=200, high=800, step=100),
        batch_size=dict(type='categorical', choices=[128, 256, 512]),
    ),
    evaluator='RankBasedEvaluator',
    evaluator_kwargs=dict(
        filtered=True,
    ),
    # ★★★ ここからが早期終了の追加設定 ★★★
    stopper='early',  # 早期終了を有効化
    stopper_kwargs=dict(
        metric='inverse_harmonic_mean_rank', # 監視する指標 (検証MRR)
        patience=5,          # 性能が5回連続で改善しなかったら停止
        frequency=25,        # 25エポックごとに検証を実施
        relative_delta=0.002 # 0.2%以上の改善を「改善」とみなす
    ),
    # ★★★ ここまでが追加設定 ★★★
    metric='inverse_harmonic_mean_rank',
    direction='maximize',
    study_name='rotate_hpo_study_with_early_stopping',
    storage=f'sqlite:///{output_directory}/hpo_study.db',
    sampler='TPE', # TPEサンプラーを使用
)

hpo_result.save_to_directory(output_directory)

print("\n" + "="*50)
print("--- HPOが完了しました ---")
print(f"結果は '{output_directory}' ディレクトリに保存されました。")
print("="*50)

print("\n--- 最適なハイパーパラメータと性能 ---")
print(f"Validation MRR: {hpo_result.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in hpo_result.best_params.items():
    print(f"  {key}: {value}")

print("\n--- 最適モデルでの最終テスト結果 ---")
print(hpo_result.best_trial.user_attrs['final_results'].to_df())