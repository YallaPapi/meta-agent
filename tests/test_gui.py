"""Tests for the GUI module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestGUIImports:
    """Test that GUI module imports correctly."""

    def test_gui_module_imports(self):
        """Test that GUI module can be imported."""
        from metaagent import gui
        assert hasattr(gui, 'MetaAgentGUI')
        assert hasattr(gui, 'launch_gui')
        assert hasattr(gui, 'main')

    def test_gui_class_exists(self):
        """Test that MetaAgentGUI class exists."""
        from metaagent.gui import MetaAgentGUI
        assert MetaAgentGUI is not None


class TestGUILicenseIntegration:
    """Test GUI integration with license module."""

    def test_gui_imports_license_constants(self):
        """Test that GUI imports license constants."""
        from metaagent.gui import FREE_TIER_LIMIT, PURCHASE_URL
        assert FREE_TIER_LIMIT == 5
        assert 'gumroad' in PURCHASE_URL


class TestLaunchGUI:
    """Test launch_gui function."""

    @patch('metaagent.gui.tk.Tk')
    @patch('metaagent.gui.MetaAgentGUI')
    def test_launch_gui_creates_window(self, mock_gui_class, mock_tk):
        """Test that launch_gui creates a Tk window."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root

        from metaagent.gui import launch_gui
        launch_gui()

        mock_tk.assert_called_once()
        mock_gui_class.assert_called_once_with(mock_root)
        mock_root.mainloop.assert_called_once()
