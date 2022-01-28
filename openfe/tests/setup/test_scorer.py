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
    def test_abstract_error(self, mock_atommapping):
        scorer = Scorer()
        with pytest.raises(NotImplementedError, match="'Scorer'.*abstract"):
            scorer(mock_atommapping)

    def test_concrete_scorer(self, mock_atommapping):
        # The ConcreteScorer class should give the implemented value for the
        # score and the default empty dict for the annotation.
        scorer = ConcreteScorer()
        result = scorer(mock_atommapping)
        assert result.score == 3.14
        assert result.annotation == {}

    def test_concrete_annotator(self, mock_atommapping):
        # The ConcreteAnnotator class should give the implemented (None)
        # value for the score and the implemented value of the annotation.
        scorer = ConcreteAnnotator()
        result = scorer(mock_atommapping)
        assert result.score is None
        assert result.annotation == {'annotation': 'data'}
