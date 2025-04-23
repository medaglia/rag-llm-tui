import random
from typing import Optional

from d7 import dice_expression
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from utils import get_logger

logger = get_logger(__name__)

DICE_TOOL_NAME = "dice"

DICE_TOOL_DESCRIPTION = """
A tool to roll dice using dice notation.

The number and size of the dice are represented in simple, math-like expressions. Using the different notations below,
on top of the traditional XdY, you can form both simple or complex dice-rolling expressions. The maximum sided die is 100.

Examples:
1d6+2 - roll one six-sided die, adding two to the result;
2d4rr1+1 - roll two, four-sided dice, re-rolling the value one, adding one to the result;
3d6ro<2kh2 - roll three, six-sided dice, re-rolling the value two at most once, keeping the highest two rolls;
6d8rr1mi3kh3!+4 - roll six exploding, eight-sided dice, whose minimum value is three, re-rolling the value one, keeping the highest three, and adding four to the result.
"""


class Roller:
    """
    A dice notation interpreter and roller for use in tabletop role-playing games (TTRPGs).
    """

    def __init__(self, text: str = None, *args, **kwargs):
        self.text = text.replace(" ", "")
        self.expression = dice_expression.DiceExpression(self.text)

    def roll(self) -> int:
        """Roll the dice"""
        result = self.expression.roll()
        logger.debug(f"Rolling {self.text} = {result}")
        logger.debug(self.expression.dice)
        return result

    def __str__(self) -> str:
        """Return the dice expression"""
        return self.text

    def __repr__(self) -> str:
        """Return the dice expression in JSON format"""
        return self.expression.toJSON()


class DiceToolInput(BaseModel):
    text: str = Field(
        description="dice expression", examples=["1d100", "2d6", "3d10+1"]
    )


class DiceTool(BaseTool):
    """
    Roll dice using dice notation
    """

    name: str = DICE_TOOL_NAME
    description: str = DICE_TOOL_DESCRIPTION
    args_schema: Optional[ArgsSchema] = DiceToolInput
    run_manager: Optional[AsyncCallbackManagerForToolRun] = (None,)
    return_direct: bool = False

    def _run(
        self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self.format(Roller(text).roll())

    async def _arun(
        self, text: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return self._run(text, run_manager=run_manager.get_sync())

    def format(self, text: str) -> str:
        formats = [
            "(╯°□°）╯ .✧∘˳°     {text}",
            "(੭˃ᴗ˂)⊃━ .✧.*･    {text}",
            "(੭⁰ᴗ⁰)⊃━ ✧.*･    {text}",
            "(ಠ ༻࿓ ಠ)⊃━ ✧.*･    {text}",
            "(∩^o^)⊃━ ✧.*･    {text}",
            "(๑╹ڡ╹)╭━✧°˖    {text}",
            "༼ ಠДಠ ༽╭o͡͡͡━ ✧.*･｡ﾟ    {text}",
            "(⊃ ° ͜ʖ͡° )⊃━ .*･｡ﾟ    {text}",
        ]
        return "Dice Roll!  \n\n" + random.choice(formats).format(text=text)
