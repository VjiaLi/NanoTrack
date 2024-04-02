import os
from abc import abstractmethod, ABC
from enum import Enum, unique


class BaseUtil(ABC):
    """
    抽取数据基类
    """

    @abstractmethod
    def init(self):
        """
        工具类初始化
        :return:
        """
        pass
