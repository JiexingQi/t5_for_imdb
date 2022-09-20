"""IMDB metrics."""

from typing import Optional, Union
import datasets
from seq2seq.metrics.imdb.imdb_accuracy import compute_accuracy_metric


_DESCRIPTION = """
IMDB metrics.
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class IMDB(datasets.Metric):
    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs
    ):
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=cache_dir,
            num_process=num_process,
            process_id=process_id,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=max_concurrent_cache_files,
            timeout=timeout,
            **kwargs
        )
        self.test_suite_db_dir: Optional[str] = kwargs.pop("test_suite_db_dir", None)

    def _info(self):
        if self.config_name not in [
            "accuracy",
        ]:
            raise KeyError(
                "You should supply a configuration name selected in " '["exact_match", "test_suite", "both"]'
            )
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": {
                        "text": datasets.Value("string"), 
                        "context": datasets.Value("string"),
                        "label": datasets.Value("string"), 
                        # "label": datasets.features.ClassLabel(names=["neg", "pos"]),
                        # "sentiment": datasets.Value("string")
                    },
                }
            ),
        )

    def _compute(self, predictions, references):
        if self.config_name == "accuracy":
            accuracy = compute_accuracy_metric(predictions=predictions, references=references)
        return {**accuracy}
