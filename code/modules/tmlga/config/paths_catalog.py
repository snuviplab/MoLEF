import os

from .settings import (
    ANET_FEATURES_PATH,
    CHARADES_FEATURES_PATH,
    EMBEDDINGS_PATH,
    ANNOTATIONS_PATH)


class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "anet_cap_train": {
            "feature_path": '/data/projects/VT_localization/tsgv_data/data/activity/org',
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'activitynet/activitynet_tmlga_train.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },

        "anet_cap_test": {
            "feature_path": '/data/projects/VT_localization/tsgv_data/data/activity/org',
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'activitynet/activitynet_tmlga_test.json'),
            "embeddings_path":
                os.path.join(
                    EMBEDDINGS_PATH, 'glove.840B.300d.txt'),
        },

        "charades_sta_train": {
            "feature_path": os.path.join(
                CHARADES_FEATURES_PATH, 'rgb'),
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'charades-sta/charades_sta_train_tokens.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt')
        },

        "charades_sta_test": {
            "feature_path": os.path.join(
                CHARADES_FEATURES_PATH, 'rgb'),
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'charades-sta/charades_sta_test_tokens.json'),
            "embeddings_path": os.path.join(
                EMBEDDINGS_PATH, 'glove.840B.300d.txt')
        },

         "didemo_train": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/didemo/org",
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'didemo/didemo_tmlga_train.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },

        "didemo_test": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/didemo/org",
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'didemo/didemo_tmlga_test.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },

         "tacos_train": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/tacos/org",
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'tacos/tacos_tmlga_train.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },

        "tacos_test": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/tacos/org",
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'tacos/tacos_tmlga_test.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },
         "tvr_train": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/tvr/concat",
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'tvr/tvr_tmlga_train.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },
        "tvr_test": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/tvr/concat",
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'tvr/tvr_tmlga_test.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },
        "youcook2_train": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/youcook2/coot",
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'youcook2/youcook2_tmlga_train.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },

        "youcook2_test": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/youcook2/coot",
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'youcook2/youcook2_tmlga_test.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },
        "msrvtt_train": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/msr_vtt/org",
            "ann_file_path":
                os.path.join(
                    ANNOTATIONS_PATH, 'msrvtt/msrvtt_tmlga_train.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },

        "msrvtt_test": {
            "feature_path": "/data/projects/VT_localization/tsgv_data/data/msr_vtt/org",
            "ann_file_path": os.path.join(
                ANNOTATIONS_PATH, 'msrvtt/msrvtt_tmlga_test.json'),
            "embeddings_path": os.path.join(
                '/data/projects/VT_localization/tsgv_data/data', 'glove.840B.300d.bin')
        },
    }

    @staticmethod
    def get(name):
        if "charades_sta" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(data_dir, attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(data_dir, attrs["embeddings_path"]),
            )
            return dict(
                factory="CHARADES_STA",
                args=args,
            )

        if "anet_cap" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(attrs["embeddings_path"]),
            )
            return dict(
                factory="ANET_CAP",
                args=args,
            )

        if "didemo" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(attrs["embeddings_path"]),
            )
            return dict(
                factory="DIDEMO",
                args=args,
            )
        if "tacos" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(attrs["embeddings_path"]),
            )
            return dict(
                factory="TACOS",
                args=args,
            )
        if "youcook2" in name :
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(attrs["embeddings_path"]),
            )
            return dict(
                factory="Youcook2",
                args=args,
            )
        if "tvr" in name :
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                features_path=os.path.join(attrs["feature_path"]),
                ann_file_path=os.path.join(data_dir, attrs["ann_file_path"]),
                embeddings_path=os.path.join(attrs["embeddings_path"]),
            )
            return dict(
                factory="Tvr",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
