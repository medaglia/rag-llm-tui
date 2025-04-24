from unittest.mock import MagicMock, patch

import pytest
from d7.dice_expression import DiceExpression

from tools.dice import DiceTool, Roller


class TestRoller:
    def test_init(self):
        roller = Roller("1d6")
        assert roller.text == "1d6"
        assert type(roller.expression) is DiceExpression

        # Ignores whitespace
        roller = Roller("1 d 10")
        assert roller.text == "1d10"

    def test_roll(self):
        roller = Roller("1d6")
        assert roller.roll() >= 1 and roller.roll() <= 6

    def test_str(self):
        roller = Roller("1d6")
        assert str(roller) == "1d6"

    def test_repr(self):
        roller = Roller("1d6")
        assert repr(roller) == roller.expression.toJSON()


class MockRoller:
    def roll(self):
        return 5


class TestDiceTool:
    @patch.object(Roller, "roll", return_value=5)
    @patch.object(DiceTool, "formatted")
    def test_run(self, mock_formatted, mock_roll):
        DiceTool()._run("1d6")
        mock_formatted.assert_called_once_with(5)

    @patch.object(Roller, "roll", return_value=5)
    @patch.object(DiceTool, "formatted")
    @pytest.mark.asyncio
    async def test_arun(self, mock_formatted, mock_roll):
        await DiceTool()._arun("1d6", MagicMock())
        mock_formatted.assert_called_once_with(5)
