"""Schema defaults for config-driven CLI commands."""

DEFAULTS = {
    "model": {
        "name_or_path": "facebook/wav2vec2-xls-r-300m",
    },
    "dataset": {
        "backend": "hf",  # hf | local
        "hf_name": "pengyizhou/nsc-imda-part6",
        "hf_config": None,
        "splits": {
            "train": "train",
            "val": "validation",
            "test": "test",
        },
        "columns": {
            "audio": "audio",
            "transcript": "text",
            "test_audio": "audio",
            "test_transcript": None,
            "id": "id",
        },
        "local": {
            "manifests": {
                "train": None,
                "val": None,
                "test": None,
            },
            "audio_root": None,
        },
    },
    "transcript": {
        "source": "inline",  # inline | jsonl
        "join": {
            "dataset_key": "id",
            "json_key": "id",
            "strict": True,
        },
        "jsonl": {
            "type": "pseudolabel",  # ground_truth | pseudolabel
            "json_path": "examples/pseudolabels/IMDA_pseudolabels.jsonl",
            "dev_json_path": None,
            "text_col": "text",
            "score_col": "score",
            "min_score": 0.6,
            "audio_path_col": "audio_path",
            "hf_id_col": "id",
            "hf_audio_splits": "train,validation",
            "prevent_test_leakage": True,
        },
    },
    "normalization": {
        "use_text_normalizer": True,
        "normalizer_yaml": None,
    },
    "vocab": {
        "out_path": "artifacts/vocab/vocab_shared.json",
        "mode": "shared_hf_plus_pseudolabel",  # dataset_only | shared_hf_plus_pseudolabel
        "score_col": "score",
        "min_score": 0.0,
        "aux_score_col": "score",
        "aux_min_score": 0.6,
    },
    "training": {
        "out_dir": "runs/xlsr300m_gt",
        "max_sec": 30.0,
        "num_proc": 8,
        "seed": 42,
        "val_size": 0.1,
        "bs": 16,
        "grad_accum": 2,
        "lr": 3e-5,
        "epochs": 20.0,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "eval_steps": 1200,
        "save_steps": 1200,
        "logging_steps": 50,
        "save_total_limit": 4,
        "best_metric": "wer_decoded_norm",
    },
    "eval": {
        "split": "test",
        "columns": {
            "audio": None,
            "transcript": None,
            "id": "id",
        },
        "local_manifest": None,
        "batch_size": 16,
        "num_workers": 4,
        "num_samples": 0,
        "max_sec": 0.0,
        "discard_number_samples": False,
        "use_text_normalizer": True,
        "normalizer_yaml": None,
        "space_chinese_chars": True,
        "verbalize_numbers": False,
        "device": "auto",
        "model_dir": None,
        "out_json": "eval_outputs/metrics.json",
        "out_jsonl": "eval_outputs/preds.jsonl",
    },
    "plot": {
        "ground_truth_run": None,
        "pseudolabel_run": None,
        "ground_truth_label": "Ground truth",
        "pseudolabel_label": "Pseudolabel",
        "wer_variant": "norm",  # norm | raw
        "out_dir": "runs/compare_plots",
    },
}
