import logging
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pytest


@pytest.mark.optuna
class TestOptunaConfiguration(unittest.TestCase):
    def test_log_level(self):
        import optuna
        from timeeval.integration.optuna import OptunaConfiguration

        # test setting log level on init
        old_logging_level = optuna.logging.get_verbosity()
        config = OptunaConfiguration(default_storage="journal-file", log_level=logging.DEBUG)
        new_logging_level = optuna.logging.get_verbosity()
        self.assertNotEqual(old_logging_level, new_logging_level)
        self.assertEqual(new_logging_level, logging.DEBUG)

        # test setting log level on change
        config.log_level = logging.ERROR
        new_logging_level2 = optuna.logging.get_verbosity()
        self.assertNotEqual(new_logging_level, new_logging_level2)
        self.assertEqual(new_logging_level2, logging.ERROR)

    # @pytest.mark.usefixtures("capfd")
    def test_update_log_stream(self):
        import optuna
        from timeeval.integration.optuna import OptunaConfiguration

        # set up logging
        logfile = StringIO()
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(logging.StreamHandler(logfile))

        # because we test global options, make sure that Optuna defaults are still set:
        optuna.logging.enable_default_handler()
        optuna.logging.disable_propagation()

        test_logger = optuna.logging.get_logger("optuna.test-logger")

        # Optuna logs to stderr by default
        test_logger.error("logging-test")
        self.assertNotIn("logging-test", logfile.getvalue())
        # self.assertIn("logging-test", capfd.readouterr().err)

        # within TimeEval, Optuna logs to root logger from logging configuration
        config = OptunaConfiguration(default_storage="journal-file", log_level=logging.DEBUG)
        test_logger.info("logging-test2")
        self.assertIn("logging-test2", logfile.getvalue())
        # self.assertNotIn("logging-test2", capfd.readouterr().err)

        # test whether changes on the config are reflected
        config.use_default_logging = True
        test_logger.error("logging-test3")
        self.assertNotIn("logging-test3", logfile.getvalue())


@pytest.mark.optuna
class TestOptunaModule(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_timeeval = Mock(spec=["distributed", "remote", "exps", "results_path"])
        self.mock_timeeval.distributed = False
        self.mock_timeeval.remote = Mock(spec=["config", "run_on_scheduler"])
        self.mock_timeeval.remote.config = Mock(spec=["scheduler_host"])
        self.mock_timeeval.exps = Mock(algorithms=[])
        self.full_storage_url = "postgresql://postgres:hairy_bumblebee@test-host:5432/postgres"

    def test_missing_or_wrong_storage(self):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        with self.assertRaises(ValueError):
            OptunaModule(OptunaConfiguration(default_storage=None))

    @patch("timeeval.integration.optuna.module.OptunaModule._check_docker_available")
    def test_checks_docker(self, mock_check):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        OptunaModule(OptunaConfiguration(default_storage="journal-file", dashboard=True))
        mock_check.assert_called_with("start the Optuna dashboard")

        OptunaModule(OptunaConfiguration(default_storage="postgresql", dashboard=False))
        mock_check.assert_called_with("start the Optuna storage postgresql")

    @patch("socket.gethostname")
    @patch("timeeval.integration.optuna.module._start_postgres_container")
    @patch("timeeval.integration.optuna.module.OptunaModule._check_docker_available")
    def test_prepare_start_storage_container(self, mock_check, mock_start_postgres, mock_hostname):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        module = OptunaModule(OptunaConfiguration(default_storage="postgresql", dashboard=False))
        mock_hostname.return_value = "test-host"
        self.mock_timeeval.distributed = False

        module.prepare(self.mock_timeeval)

        self.assertEqual(module.storage_url, self.full_storage_url)
        mock_check.assert_called_once()
        mock_start_postgres.assert_called_with(password="hairy_bumblebee", port=5432)

    @patch("socket.gethostname")
    @patch("timeeval.integration.optuna.module.OptunaModule._check_docker_available")
    def test_prepare_not_starting_dashboard(self, mock_check, mock_hostname):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        module = OptunaModule(OptunaConfiguration(default_storage="journal-file", dashboard=True))
        mock_hostname.return_value = "test-host"
        self.mock_timeeval.distributed = False

        with tempfile.TemporaryDirectory() as tempdir:
            self.mock_timeeval.results_path = Path(tempdir)
            with self.assertLogs(level="WARNING") as logs:
                module.prepare(self.mock_timeeval)
            self.assertRegex("\n".join(logs.output), "[C|c]ould not find dashboard connection URL")

        mock_check.assert_called_once()
        self.assertIsNone(module.storage_url)
        # mock_start_dashboard.assert_called_with(storage="http://localhost:8080")

    @patch("socket.gethostname")
    @patch("timeeval.integration.optuna.module._start_dashboard_container")
    @patch("timeeval.integration.optuna.module._start_postgres_container")
    @patch("timeeval.integration.optuna.module.OptunaModule._check_docker_available")
    def test_prepare_start_storage_and_dashboard_container(self, mock_check, mock_start_postgres, mock_start_dashboard, mock_hostname):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        module = OptunaModule(OptunaConfiguration(default_storage="postgresql", dashboard=True))
        mock_hostname.return_value = "test-host"
        self.mock_timeeval.distributed = False

        module.prepare(self.mock_timeeval)

        self.assertTrue(callable(module.config.default_storage))
        self.assertEqual(module.storage_url, self.full_storage_url)
        mock_check.assert_called_once()
        mock_start_postgres.assert_called_with(password="hairy_bumblebee", port=5432)
        mock_start_dashboard.assert_called_with(storage=self.full_storage_url)

    @patch("socket.gethostname")
    @patch("timeeval.integration.optuna.module._start_postgres_container")
    @patch("timeeval.integration.optuna.module.OptunaModule._check_docker_available")
    def test_prepare_start_on_scheduler(self, mock_check, mock_start_postgres, mock_hostname):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        module = OptunaModule(OptunaConfiguration(default_storage="postgresql", dashboard=False))
        self.mock_timeeval.distributed = True

        module.prepare(self.mock_timeeval)

        self.mock_timeeval.remote.run_on_scheduler.assert_called_once_with(
            [(mock_start_postgres, [], {"password": "hairy_bumblebee", "port": 5432})],
            msg="Starting Optuna containers on scheduler"
        )

    @patch("timeeval.integration.optuna.module._stop_containers")
    def test_finalize_local(self, mock_stop_containers):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        module = OptunaModule(OptunaConfiguration(default_storage="postgresql", dashboard=False))
        self.mock_timeeval.distributed = False

        # without remove
        module.finalize(self.mock_timeeval)
        mock_stop_containers.assert_called_with(remove=False)
        self.mock_timeeval.remote.run_on_scheduler.assert_not_called()

        # with remove
        module.config.remove_managed_containers = True
        module.finalize(self.mock_timeeval)
        mock_stop_containers.assert_called_with(remove=True)

    @patch("timeeval.integration.optuna.module._stop_containers")
    def test_finalize_distributed(self, mock_stop_containers):
        from timeeval.integration.optuna import OptunaModule, OptunaConfiguration

        module = OptunaModule(OptunaConfiguration(default_storage="postgresql", dashboard=False))
        self.mock_timeeval.distributed = True

        # without remove
        module.finalize(self.mock_timeeval)
        self.mock_timeeval.remote.run_on_scheduler.assert_called_with(
            [(mock_stop_containers, [], {"remove": False})],
            msg="Stopping Optuna containers on scheduler"
        )

        # with remove
        module.config.remove_managed_containers = True
        module.finalize(self.mock_timeeval)
        self.mock_timeeval.remote.run_on_scheduler.assert_called_with(
            [(mock_stop_containers, [], {"remove": True})],
            msg="Stopping Optuna containers on scheduler"
        )
