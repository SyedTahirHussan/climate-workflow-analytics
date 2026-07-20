"""End-to-end CLI tests: each subcommand runs the real pipeline on a tiny file."""

import matplotlib
import pytest

matplotlib.use("Agg")  # headless-safe before any pyplot import

from cwa.cli import main  # noqa: E402


@pytest.fixture()
def small_nc(tmp_path):
    """Generate a small dataset once per test via the real `gen` command."""
    path = tmp_path / "demo.nc"
    main(["gen", "--path", str(path), "--n-time", "120"])
    return path


def test_version_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_gen_reports_file(tmp_path, capsys):
    path = tmp_path / "gen.nc"
    main(["gen", "--path", str(path), "--n-time", "24"])
    out = capsys.readouterr().out
    assert "wrote" in out
    assert path.exists()


def test_stream_reports_index(small_nc, capsys):
    main(["stream", "--path", str(small_nc)])
    out = capsys.readouterr().out
    assert "in-transit spatial-mean reduction" in out
    assert "steps processed : 120" in out


def test_forecast_reports_skill(small_nc, capsys):
    main(["forecast", "--path", str(small_nc), "--horizon", "6"])
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "skill score" in out


def test_forecast_plot_writes_png(small_nc, tmp_path, capsys):
    fig = tmp_path / "forecast.png"
    main(["forecast", "--path", str(small_nc), "--horizon", "6", "--plot", str(fig)])
    out = capsys.readouterr().out
    assert fig.exists() and fig.stat().st_size > 1000  # a real image, not a stub
    assert str(fig) in out  # CLI tells the user where the figure went
