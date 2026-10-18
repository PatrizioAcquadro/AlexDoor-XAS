"""Desktop PID churn cannot become a false competing-compute stop."""

import pytest

from alexdoor_xas.resources import is_desktop_graphics


@pytest.mark.parametrize("pid", [100, 200])
def test_known_graphics_renderer_is_classified_by_role_across_PID_changes(pid):
    assert is_desktop_graphics(
        {"pid": pid, "kind": "G", "name": "/opt/google/chrome/chrome --type=gpu-process"}
    )


@pytest.mark.parametrize(
    "process",
    [
        {"kind": "C", "name": "/opt/google/chrome/chrome"},
        {"kind": "C+G", "name": "/opt/google/chrome/chrome"},
        {"kind": "C", "name": "/usr/share/rustdesk/rustdesk"},
        {"kind": "G", "name": "/other/renderer"},
    ],
)
def test_compute_and_unknown_graphics_remain_unexempted(process):
    assert not is_desktop_graphics(process)
