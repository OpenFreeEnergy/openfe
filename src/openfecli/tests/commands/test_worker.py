from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from openfecli.commands.worker import worker


class _SuccessfulResult:
    def ok(self):
        return True


class _FailedResult:
    def ok(self):
        return False


def test_worker_requires_task_database():
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("warehouse").mkdir()
        result = runner.invoke(worker, ["warehouse"])
        assert result.exit_code == 1
        assert "Task database not found at" in result.output


def test_worker_no_available_task_exits_zero():
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = Path("warehouse")
        warehouse_path.mkdir()
        (warehouse_path / "tasks.db").touch()

        mock_worker = mock.Mock()
        mock_worker.execute_unit.return_value = None

        with mock.patch(
            "openfecli.commands.worker._build_worker", return_value=mock_worker
        ) as build_worker:
            result = runner.invoke(worker, ["warehouse"])

        assert result.exit_code == 0
        assert "No available task in task graph." in result.output
        build_worker.assert_called_once_with(warehouse_path, warehouse_path / "tasks.db")
        kwargs = mock_worker.execute_unit.call_args.kwargs
        assert kwargs["scratch"] == Path.cwd()


def test_worker_executes_one_task_and_reports_completion():
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = Path("warehouse")
        warehouse_path.mkdir()
        (warehouse_path / "tasks.db").touch()

        mock_worker = mock.Mock()
        mock_worker.execute_unit.return_value = (
            "Transformation-abc:ProtocolUnit-def",
            _SuccessfulResult(),
        )

        with mock.patch("openfecli.commands.worker._build_worker", return_value=mock_worker):
            result = runner.invoke(worker, ["warehouse", "--scratch", "scratch"])

        assert result.exit_code == 0
        assert "Completed task: Transformation-abc:ProtocolUnit-def" in result.output
        assert Path("scratch").is_dir()
        kwargs = mock_worker.execute_unit.call_args.kwargs
        assert kwargs["scratch"] == Path("scratch")


def test_worker_raises_when_result_is_failure():
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = Path("warehouse")
        warehouse_path.mkdir()
        (warehouse_path / "tasks.db").touch()

        mock_worker = mock.Mock()
        mock_worker.execute_unit.return_value = (
            "Transformation-abc:ProtocolUnit-def",
            _FailedResult(),
        )

        with mock.patch("openfecli.commands.worker._build_worker", return_value=mock_worker):
            result = runner.invoke(worker, ["warehouse"])

        assert result.exit_code == 1
        assert "returned a failure result" in result.output


def test_worker_raises_when_execution_throws():
    runner = CliRunner()
    with runner.isolated_filesystem():
        warehouse_path = Path("warehouse")
        warehouse_path.mkdir()
        (warehouse_path / "tasks.db").touch()

        mock_worker = mock.Mock()
        mock_worker.execute_unit.side_effect = RuntimeError("boom")

        with mock.patch("openfecli.commands.worker._build_worker", return_value=mock_worker):
            result = runner.invoke(worker, ["warehouse"])

        assert result.exit_code == 1
        assert "Task execution failed: boom" in result.output
