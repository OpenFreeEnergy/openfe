import pytest
from openfe.setup.scorer import Scorer


class ConcreteScorer(Scorer):
    """Test implementation of Scorer with a score"""
    def _score(self, atommapping):
        return 3.14


class ConcreteAnnotator(Scorer):
    """Test implementation of Scorer with a custom annotation"""
    def _score(self, atommapping):
        return None

    def _annotation(self, atommapping):
        return {'annotation': 'data'}


class TestScorer:
    def test_abstract_error(self, simple_mapping):
        scorer = Scorer()
        match_re = "'Scorer'.*abstract.*_score"
        with pytest.raises(NotImplementedError, match=match_re):
            scorer(simple_mapping)

    def test_concrete_scorer(self, simple_mapping):
        # The ConcreteScorer class should give the implemented value for the
        # score and the default empty dict for the annotation.
        scorer = ConcreteScorer()
        result = scorer(simple_mapping)
        assert result.score == 3.14
        assert result.annotation == {}

    def test_concrete_annotator(self, simple_mapping):
        # The ConcreteAnnotator class should give the implemented (None)
        # value for the score and the implemented value of the annotation.
        scorer = ConcreteAnnotator()
        result = scorer(simple_mapping)
        assert result.score is None
        assert result.annotation == {'annotation': 'data'}
