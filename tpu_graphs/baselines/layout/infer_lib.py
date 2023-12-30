import tensorflow as tf
import tensorflow_gnn as tfgnn
from absl import flags
import os
import tqdm
from tpu_graphs.baselines.layout import data
from tpu_graphs.baselines.layout import infer_args

from tpu_graphs.baselines.layout import models  # 假设 ResModel 定义在这里
import tensorflow_ranking as tfr

_DATA_ROOT = flags.DEFINE_string(
    'data_root', './data/tpugraphs/npz/layout',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, valid}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', './data/tpugraphs/cache/layout',
    'If given, dataset tensors will be cached here for faster loading. Files '
    'with name "<hash>.npz" will be written, where <hash> is a hash of the '
    'filepattern of training data, i.e., it depends on the collection e.g., '
    '{xla:default} and partition {train, test, valid}.')
_PDB = flags.DEFINE_integer(
    'debug', -1, 'If >0, pdb debugger will be entered after this many epochs.')


def run_inference(model, test_dataset, results_csv_path, source, search, inference_batch_size):
    test_rankings = []
    for graph in tqdm.tqdm(test_dataset.test.iter_graph_tensors(),
                           total=test_dataset.test.graph_id.shape[-1],
                           desc='Inference'):
        num_configs = graph.node_sets['g']['runtimes'].shape[-1]
        all_scores = []
        for i in range(0, num_configs, inference_batch_size):
            end_i = min(i + inference_batch_size, num_configs)
            # Take a cut of the configs.
            node_set_g = graph.node_sets['g']
            subconfigs_graph = tfgnn.GraphTensor.from_pieces(
                edge_sets=graph.edge_sets,
                node_sets={
                    'op': graph.node_sets['op'],
                    'nconfig': tfgnn.NodeSet.from_fields(
                        sizes=graph.node_sets['nconfig'].sizes,
                        features={
                            'feats': graph.node_sets['nconfig']['feats'][:, i:end_i],
                        }),
                    'g': tfgnn.NodeSet.from_fields(
                        sizes=tf.constant([1]),
                        features={
                            'graph_id': node_set_g['graph_id'],
                            'runtimes': node_set_g['runtimes'][:, i:end_i],
                            'kept_node_ratio': node_set_g['kept_node_ratio'],
                        })
                })
            h = model.forward(subconfigs_graph, num_configs=(end_i - i),
                              backprop=False)
            all_scores.append(h[0])
        all_scores = tf.concat(all_scores, axis=0)
        graph_id = graph.node_sets['g']['graph_id'][0].numpy().decode()
        sorted_indices = tf.strings.join(
            tf.strings.as_string(tf.argsort(all_scores)), ';').numpy().decode()
        test_rankings.append((graph_id, sorted_indices))

    # 保存推理结果到 CSV 文件
    with tf.io.gfile.GFile(results_csv_path, 'w') as fout:
        fout.write('ID,TopConfigs\n')
        for graph_id, ranks in test_rankings:
            fout.write(f'layout:{source}:{search}:{graph_id},{ranks}\n')

    print(f'\n\n   ***  Wrote {results_csv_path} \n\n')


def infer(args: infer_args.TrainArgs):

    model_path = 'E:/aaai/out/tpugraphs_layout/model_7ca2bd8ccb2d4c464f2c315d300fd52e/variables/variables'
    # 需要替换为实际参数
    data_root_dir = os.path.join(
        os.path.expanduser(_DATA_ROOT.value), args.source, args.search)
    num_configs = args.configs

    test_dataset = data.get_npz_dataset(
        data_root_dir, min_train_configs=num_configs,
        max_train_configs=args.max_configs,
        cache_dir=os.path.expanduser(_CACHE_DIR.value))
    model = models.ResModel(num_configs, test_dataset.num_ops)
    # 加载模型
    model.load_weights(model_path)
    # model = tf.keras.models.load_model(model_path)

    results_csv_path = 'E:/aaai/out/results.csv'
    source = 'xla'
    search = 'default'
    inference_batch_size = 500


    run_inference(model, test_dataset, results_csv_path, source, search, inference_batch_size)


if __name__ == '__main__':
    model_path = 'E:/aaai/out/tpugraphs_layout/model_7ca2bd8ccb2d4c464f2c315d300fd52f'
    # model = models.ResModel(num_configs, test_dataset.num_ops)
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    model.summary()
